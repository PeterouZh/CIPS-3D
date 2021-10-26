import tqdm
import random
import time

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from tl2.proj.fvcore import MODEL_REGISTRY, build_model
from tl2.proj.stylegan2_ada import persistence
from tl2.launch.launch_utils import global_cfg
from tl2.proj.pytorch.pytorch_hook import VerboseModel
from tl2.proj.pytorch import torch_utils

from exp.pigan.models.volumetric_rendering import *
from exp.pigan.models.siren import \
    (FiLMLayer, CustomMappingNetwork, frequency_init, first_layer_film_sine_init, UniformBoxWarp)


@MODEL_REGISTRY.register(name_prefix=__name__)
@persistence.persistent_class
class SPATIALSIRENBASELINE(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(self,
                 input_dim=2,
                 z_dim=100,
                 hidden_dim=256,
                 output_dim=1,
                 device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid())

        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1) * hidden_dim * 2)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

        # Don't worry about this, it was added to ensure compatibility with another model.
        # Shouldn't affect performance.
        self.gridwarper = UniformBoxWarp(0.24)
        pass

    def forward(self,
                input,
                z,
                ray_directions,
                **kwargs):
        # mapping network
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.mapping_network,
                                         submodels=['network', ],
                                         inputs_args=(z,),
                                         name_prefix="mapping.",
                                         input_padding=35,
                                         output_padding=32)
            torch_utils.print_number_params({'mapping_network': self.mapping_network})
        frequencies, phase_shifts = self.mapping_network(z)

        out = self.forward_with_frequencies_phase_shifts(
            input,
            frequencies,
            phase_shifts,
            ray_directions,
            **kwargs)

        if global_cfg.tl_debug:
            torch_utils.print_number_params({'generator': self})
        return out

    def print_number_params(self):
        print()
        torch_utils.print_number_params({'mapping_network': self.mapping_network,
                                         'network': self.network,
                                         'final_layer': self.final_layer,
                                         'color_layer_sine': self.color_layer_sine,
                                         'color_layer_linear': self.color_layer_linear,
                                         'generator': self})
        pass

    def forward_with_frequencies_phase_shifts(self,
                                              input,
                                              frequencies,
                                              phase_shifts,
                                              ray_directions,
                                              **kwargs):
        frequencies = frequencies * 15 + 30

        input = self.gridwarper(input)
        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim

            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(layer,
                                             inputs_args=(x,
                                                          frequencies[..., start:end],
                                                          phase_shifts[..., start:end]),
                                             name_prefix=f"FiLM.{index}",
                                             register_children=False,
                                             register_itself=True)
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        if global_cfg.tl_debug:
            torch_utils.print_number_params(models_dict={'synthesis network': self.network})

        if global_cfg.tl_debug:
            print("")
            VerboseModel.forward_verbose(self.final_layer,
                                         inputs_args=(x, ),
                                         name_prefix="sigma_out",
                                         register_itself=True)
        sigma = self.final_layer(x)

        if global_cfg.tl_debug:
            print("")
            VerboseModel.forward_verbose(self.color_layer_sine,
                                         inputs_args=(torch.cat([ray_directions, x], dim=-1),
                                                      frequencies[..., -self.hidden_dim:],
                                                      phase_shifts[..., -self.hidden_dim:]),
                                         name_prefix=f"color_fc_sine",
                                         register_children=False,
                                         register_itself=True)
        rbg = self.color_layer_sine(
            torch.cat([ray_directions, x], dim=-1),
            frequencies[..., -self.hidden_dim:],
            phase_shifts[..., -self.hidden_dim:])

        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.color_layer_linear,
                                         inputs_args=(rbg, ),
                                         name_prefix="color_out.")
        # rbg = torch.sigmoid(self.color_layer_linear(rbg))
        rbg = self.color_layer_linear(rbg)

        out = torch.cat([rbg, sigma], dim=-1)
        return out


@MODEL_REGISTRY.register(name_prefix=__name__)
@persistence.persistent_class
class ImplicitGenerator3d(nn.Module):
    def __init__(self,
                 z_dim,
                 siren_cfg,
                 **kwargs):
        super().__init__()
        self.z_dim = z_dim
        # self.siren = SIREN(output_dim=4, z_dim=self.z_dim, input_dim=3, device=None)
        self.siren = build_model(cfg=siren_cfg, output_dim=4, z_dim=self.z_dim, input_dim=3, device=None)
        self.print_number_params()

        self.epoch = 0
        self.step = 0
        pass

    def print_number_params(self):
        self.siren.print_number_params()
        pass

    def set_device(self, device):
        self.device = device
        self.siren.device = device

        self.generate_avg_frequencies()
        pass

    def get_xyz_range(self,
                      num_samples,
                      z,
                      img_size,
                      fov,
                      ray_start,
                      ray_end,
                      num_steps,
                      h_stddev,
                      v_stddev,
                      h_mean,
                      v_mean,
                      sample_dist=None,
                      **kwargs):
        """
        """

        batch_size = z.shape[0]

        xyz_minmax_mean = [[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]]

        for i in tqdm.tqdm(range(num_samples)):
            # Generate initial camera rays and sample points.
            with torch.no_grad():
                points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
                    batch_size,
                    num_steps,
                    resolution=(img_size, img_size),
                    device=self.device,
                    fov=fov,
                    ray_start=ray_start,
                    ray_end=ray_end)  # batch_size, pixels, num_steps, 1
                transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = \
                    transform_sampled_points(points_cam,
                                             z_vals,
                                             rays_d_cam,
                                             h_stddev=h_stddev,
                                             v_stddev=v_stddev,
                                             h_mean=h_mean,
                                             v_mean=v_mean,
                                             device=self.device,
                                             mode=sample_dist)
                xyz_list = transformed_points.unbind(3)
                for minmax_mean, axis_v in zip(xyz_minmax_mean, xyz_list):
                    min_v, max_v, mean_v = minmax_mean
                    minmax_mean[0] = min(min_v, axis_v.min().item())
                    minmax_mean[1] = max(max_v, axis_v.max().item())
                    minmax_mean[2] = minmax_mean[2] + axis_v.mean().item()
        for minmax_mean in xyz_minmax_mean:
            minmax_mean[2] = minmax_mean[2] / num_samples

        for minmax_mean, axis_name in zip(xyz_minmax_mean, 'xyz'):
            minmax_mean_str = f"{axis_name}: ({minmax_mean[0]:.2f}, {minmax_mean[1]:.2f}, {minmax_mean[2]:.2f})"
            print(minmax_mean_str)
        pass

    def forward(self,
                z,
                img_size,
                fov,
                ray_start,
                ray_end,
                num_steps,
                h_stddev,
                v_stddev,
                h_mean,
                v_mean,
                hierarchical_sample,
                sample_dist=None,
                lock_view_dependence=False,
                **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """

        batch_size = z.shape[0]

        # Generate initial camera rays and sample points.
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
                batch_size,
                num_steps,
                resolution=(img_size, img_size),
                device=self.device,
                fov=fov,
                ray_start=ray_start,
                ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = \
                transform_sampled_points(points_cam,
                                         z_vals,
                                         rays_d_cam,
                                         h_stddev=h_stddev,
                                         v_stddev=v_stddev,
                                         h_mean=h_mean,
                                         v_mean=v_mean,
                                         device=self.device,
                                         mode=sample_dist)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(
                batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        # Model prediction on course points
        coarse_output = self.siren(
            transformed_points,
            z,
            ray_directions=transformed_ray_directions_expanded)\
            .reshape(batch_size, img_size * img_size, num_steps, 4)

        # Re-sample fine points alont camera rays, as described in NeRF
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(
                    coarse_output,
                    z_vals,
                    device=self.device,
                    clamp_mode=kwargs['clamp_mode'],
                    noise_std=kwargs['nerf_noise'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                #### Start new importance sampling
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid,
                                         weights[:, 1:-1],
                                         num_steps,
                                         det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)


                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + \
                              transformed_ray_directions.unsqueeze(2).contiguous() * \
                              fine_z_vals.expand(-1,-1,-1,3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                #### end new importance sampling

            # Model prediction on re-sampled find points
            fine_output = self.siren(
                fine_points,
                z,
                ray_directions=transformed_ray_directions_expanded)\
                .reshape(batch_size, img_size * img_size, -1, 4)

            # Combine course and fine points
            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals


        # Create images with NeRF
        pixels, depth, weights = fancy_integration(
            all_outputs,
            all_z_vals,
            device=self.device,
            white_back=kwargs.get('white_back', False),
            last_back=kwargs.get('last_back', False),
            clamp_mode=kwargs['clamp_mode'],
            noise_std=kwargs['nerf_noise'])

        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

        return pixels, torch.cat([pitch, yaw], -1)

    def generate_avg_frequencies(self):
        """Calculates average frequencies and phase shifts"""

        z = torch.randn((10000, self.z_dim), device=self.siren.device)
        with torch.no_grad():
            frequencies, phase_shifts = self.siren.mapping_network(z)
        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
        return self.avg_frequencies, self.avg_phase_shifts


    def staged_forward(self,
                       z,
                       img_size,
                       fov,
                       ray_start,
                       ray_end,
                       num_steps,
                       h_stddev,
                       v_stddev,
                       h_mean,
                       v_mean,
                       psi=1,
                       lock_view_dependence=False,
                       max_batch_size=50000,
                       depth_map=False,
                       near_clip=0,
                       far_clip=2,
                       sample_dist=None,
                       hierarchical_sample=False,
                       **kwargs):
        """
        Similar to forward but used for inference.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """

        batch_size = z.shape[0]

        self.generate_avg_frequencies()

        with torch.no_grad():

            raw_frequencies, raw_phase_shifts = self.siren.mapping_network(z)

            truncated_frequencies = self.avg_frequencies + psi * (raw_frequencies - self.avg_frequencies)
            truncated_phase_shifts = self.avg_phase_shifts + psi * (raw_phase_shifts - self.avg_phase_shifts)


            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
                batch_size,
                num_steps,
                resolution=(img_size, img_size),
                device=self.device,
                fov=fov,
                ray_start=ray_start,
                ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = \
                transform_sampled_points(points_cam,
                                         z_vals,
                                         rays_d_cam,
                                         h_stddev=h_stddev,
                                         v_stddev=v_stddev,
                                         h_mean=h_mean,
                                         v_mean=v_mean,
                                         device=self.device,
                                         mode=sample_dist)


            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(
                batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # Sequentially evaluate siren with max_batch_size to avoid OOM
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
                        transformed_points[b:b+1, head:tail],
                        truncated_frequencies[b:b+1],
                        truncated_phase_shifts[b:b+1],
                        ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    head += max_batch_size

            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)


            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration(
                        coarse_output,
                        z_vals,
                        device=self.device,
                        clamp_mode=kwargs['clamp_mode'],
                        noise_std=kwargs['nerf_noise'])

                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                    #### Start new importance sampling
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid,
                                             weights[:, 1:-1],
                                             num_steps,
                                             det=False).detach().to(self.device)
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + \
                                  transformed_ray_directions.unsqueeze(2).contiguous() * \
                                  fine_z_vals.expand(-1,-1,-1,3).contiguous()
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

                # Sequentially evaluate siren with max_batch_size to avoid OOM
                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
                            fine_points[b:b+1, head:tail],
                            truncated_frequencies[b:b+1],
                            truncated_phase_shifts[b:b+1],
                            ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        head += max_batch_size

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4)

                all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals


            pixels, depth, weights = fancy_integration(
                all_outputs,
                all_z_vals,
                device=self.device,
                white_back=kwargs.get('white_back', False),
                clamp_mode = kwargs['clamp_mode'],
                last_back=kwargs.get('last_back', False),
                fill_mode=kwargs.get('fill_mode', None),
                noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()


            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1

        return pixels, depth_map

    # Used for rendering interpolations
    def staged_forward_with_frequencies(self,
                                        truncated_frequencies,
                                        truncated_phase_shifts,
                                        img_size,
                                        fov,
                                        ray_start,
                                        ray_end,
                                        num_steps,
                                        h_stddev,
                                        v_stddev,
                                        h_mean,
                                        v_mean,
                                        psi=0.7,
                                        lock_view_dependence=False,
                                        max_batch_size=50000,
                                        depth_map=False,
                                        near_clip=0,
                                        far_clip=2,
                                        sample_dist=None,
                                        hierarchical_sample=False,
                                        **kwargs):
        batch_size = truncated_frequencies.shape[0]

        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
                batch_size,
                num_steps,
                resolution=(img_size, img_size),
                device=self.device,
                fov=fov,
                ray_start=ray_start,
                ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = \
                transform_sampled_points(points_cam,
                                         z_vals,
                                         rays_d_cam,
                                         h_stddev=h_stddev,
                                         v_stddev=v_stddev,
                                         h_mean=h_mean,
                                         v_mean=v_mean,
                                         device=self.device,
                                         mode=sample_dist)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(
                batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # BATCHED SAMPLE
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
                        transformed_points[b:b+1, head:tail],
                        truncated_frequencies[b:b+1],
                        truncated_phase_shifts[b:b+1],
                        ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    head += max_batch_size

            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)
            # END BATCHED SAMPLE


            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration(
                        coarse_output,
                        z_vals,
                        device=self.device,
                        clamp_mode=kwargs['clamp_mode'],
                        noise_std=kwargs['nerf_noise'])

                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(
                        z_vals_mid,
                        weights[:, 1:-1],
                        num_steps, det=False).detach().to(self.device) # batch_size, num_pixels**2, num_steps
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + \
                                  transformed_ray_directions.unsqueeze(2).contiguous() * \
                                  fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                # fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)
                # BATCHED SAMPLE
                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
                            fine_points[b:b+1, head:tail],
                            truncated_frequencies[b:b+1],
                            truncated_phase_shifts[b:b+1],
                            ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        head += max_batch_size

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4)
                # END BATCHED SAMPLE

                all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals


            pixels, depth, weights = fancy_integration(
                all_outputs,
                all_z_vals,
                device=self.device,
                white_back=kwargs.get('white_back', False),
                clamp_mode = kwargs['clamp_mode'],
                last_back=kwargs.get('last_back', False),
                fill_mode=kwargs.get('fill_mode', None),
                noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()


            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1

        return pixels, depth_map


    def forward_with_frequencies(self,
                                 frequencies,
                                 phase_shifts,
                                 img_size,
                                 fov,
                                 ray_start,
                                 ray_end,
                                 num_steps,
                                 h_stddev,
                                 v_stddev,
                                 h_mean,
                                 v_mean,
                                 hierarchical_sample,
                                 sample_dist=None,
                                 lock_view_dependence=False,
                                 **kwargs):
        batch_size = frequencies.shape[0]

        points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
            batch_size,
            num_steps,
            resolution=(img_size, img_size),
            device=self.device,
            fov=fov,
            ray_start=ray_start,
            ray_end=ray_end) # batch_size, pixels, num_steps, 1
        transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = \
            transform_sampled_points(points_cam,
                                     z_vals,
                                     rays_d_cam,
                                     h_stddev=h_stddev,
                                     v_stddev=v_stddev,
                                     h_mean=h_mean,
                                     v_mean=v_mean,
                                     device=self.device,
                                     mode=sample_dist)


        transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(
            batch_size, img_size*img_size*num_steps, 3)
        transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

        if lock_view_dependence:
            transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
            transformed_ray_directions_expanded[..., -1] = -1
            
        coarse_output = self.siren.forward_with_frequencies_phase_shifts(
            transformed_points,
            frequencies,
            phase_shifts,
            ray_directions=transformed_ray_directions_expanded)\
            .reshape(batch_size, img_size * img_size, num_steps, 4)
        
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(
                    coarse_output,
                    z_vals,
                    device=self.device,
                    clamp_mode=kwargs['clamp_mode'],
                    noise_std=kwargs['nerf_noise'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                #### Start new importance sampling
                # RuntimeError: Sizes of tensors must match except in dimension 1.
                # Got 3072 and 6144 (The offending index is 0)
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                # We squash the dimensions here. This means we importance sample for every batch for every ray
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid,
                                         weights[:, 1:-1],
                                         num_steps,
                                         det=False).detach() # batch_size, num_pixels**2, num_steps
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)


                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + \
                              transformed_ray_directions.unsqueeze(2).contiguous() * \
                              fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                #### end new importance sampling
                
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

            fine_output = self.siren.forward_with_frequencies_phase_shifts(
                fine_points,
                frequencies,
                phase_shifts,
                ray_directions=transformed_ray_directions_expanded)\
                .reshape(batch_size, img_size * img_size, -1, 4)

            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            # Target sizes: [-1, -1, -1, 4].  Tensor sizes: [240, 512, 12]
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals


        pixels, depth, weights = fancy_integration(
            all_outputs,
            all_z_vals,
            device=self.device,
            white_back=kwargs.get('white_back', False),
            last_back=kwargs.get('last_back', False),
            clamp_mode=kwargs['clamp_mode'],
            noise_std=kwargs['nerf_noise'])

        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

        return pixels, torch.cat([pitch, yaw], -1)
