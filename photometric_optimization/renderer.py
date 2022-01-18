"""
Author: Yao Feng
Copyright (c) 2020, Yao Feng
All rights reserved.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.io import imread
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
import util


class Pytorch3dRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.

    Notice:
        x,y,z are in image space
    """

    def __init__(self, image_size=224):
        """
        Args:
            raster_settings: the parameters for rasterization. This should be a
                named tuple.
        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        raster_settings = util.dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        fixed_vetices = vertices.clone()
        fixed_vetices[..., :2] = -fixed_vetices[..., :2]
        meshes_screen = Meshes(verts=fixed_vetices.float(), faces=faces.long())
        raster_settings = self.raster_settings

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )

        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1  # []
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        # import ipdb; ipdb.set_trace()
        return pixel_vals


class Renderer(nn.Module):
    def __init__(self, image_size, obj_filename, uv_size=256):
        super(Renderer, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size

        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        faces = faces.verts_idx[None, ...]
        self.rasterizer = Pytorch3dRasterizer(image_size)
        self.uv_rasterizer = Pytorch3dRasterizer(uv_size)

        # faces
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coordsw
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        # shape colors
        colors = torch.tensor([74, 120, 168])[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.
        face_colors = util.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)

        ## lighting
        pi = np.pi
        constant_factor = torch.tensor(
            [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), \
             ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), \
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))])
        self.register_buffer('constant_factor', constant_factor)



    def forward(self, vertices, transformed_vertices, albedos, lights=None, light_type='point'):
        '''
        lihgts:
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
        vertices: [N, V, 3], vertices in work space, for calculating normals, then shading
        transformed_vertices: [N, V, 3], range(-1, 1), projected vertices, for rendering
        '''
        batch_size = vertices.shape[0]
        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        # Attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))

        # render
        attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1), transformed_face_normals.detach(),
                                face_vertices.detach(), face_normals.detach()], -1)
        # import ipdb;ipdb.set_trace()
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]
        grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]

        albedo_images = F.grid_sample(albedos, grid, align_corners=False)

        # remove inner mouth region
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

        # shading
        if lights is not None:
            normal_images = rendering[:, 9:12, :, :].detach()
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type == 'point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(vertice_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                  normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                  lights)
                    shading_images = shading.reshape(
                        [batch_size, lights.shape[1], albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 1,
                                                                                                                  4, 2,
                                                                                                                  3)
                    shading_images = shading_images.mean(1)
                else:
                    shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                      lights)
                    shading_images = shading.reshape(
                        [batch_size, lights.shape[1], albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 1,
                                                                                                                  4, 2,
                                                                                                                  3)
                    shading_images = shading_images.mean(1)
            images = albedo_images * shading_images
        else:
            images = albedo_images
            shading_images = images.detach() * 0.

        outputs = {
            'images': images * alpha_images,
            'albedo_images': albedo_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals
        }

        return outputs

    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        N = normal_images
        sh = torch.stack([
            N[:, 0] * 0. + 1., N[:, 0], N[:, 1], \
            N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2],
            N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * (N[:, 2] ** 2) - 1
        ],
            1)  # [bz, 9, h, w]
        sh = sh * self.constant_factor[None, :, None, None]
        # import ipdb; ipdb.set_trace()
        shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
        return shading

    def add_pointlight(self, vertices, normals, lights):
        '''
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_positions = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_positions[:,:,None,:] - vertices[:,None,:,:], dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nlgiht, nv, 3]
        '''
        light_direction = lights[:, :, :3];
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(light_direction[:, :, None, :].expand(-1, -1, normals.shape[1], -1), dim=3)
        normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading

    def render_shape(self, vertices, transformed_vertices, images=None, lights=None):
        batch_size = vertices.shape[0]
        if lights is None:
            light_positions = torch.tensor([[-0.1, -0.1, 0.2],
                                            [0, 0, 1]]
                                           )[None, :, :].expand(batch_size, -1, -1).float()
            light_intensities = torch.ones_like(light_positions).float()
            lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)

        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        # Attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1));
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1));
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        # render
        attributes = torch.cat(
            [self.face_colors.expand(batch_size, -1, -1, -1), transformed_face_normals.detach(), face_vertices.detach(),
             face_normals.detach()], -1)
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        # albedo
        albedo_images = rendering[:, :3, :, :]
        # shading
        normal_images = rendering[:, 9:12, :, :].detach()

        if lights.shape[1] == 9:
            shading_images = self.add_SHlight(normal_images, lights)
        else:
            print('directional')
            shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)

            shading_images = shading.reshape(
                [batch_size, lights.shape[1], albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 1, 4, 2, 3)
            shading_images = shading_images.mean(1)
        images = albedo_images * shading_images

        return images

    def render_normal(self, transformed_vertices, normals):
        '''
        -- rendering normal
        '''
        batch_size = normals.shape[0]

        # Attributes
        attributes = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        normal_images = rendering[:, :3, :, :]
        return normal_images

    def world2uv(self, vertices):
        '''
        sample vertices from world space to uv space
        uv_vertices: [bz, 3, h, w]
        '''
        batch_size = vertices.shape[0]
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1)).clone().detach()
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                         self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]

        return uv_vertices

    def save_obj(self, filename, vertices, textures):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        util.save_obj(filename, vertices, self.faces[0], textures=textures, uvcoords=self.raw_uvcoords[0],
                          uvfaces=self.uvfaces[0])