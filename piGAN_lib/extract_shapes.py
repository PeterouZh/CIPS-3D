import plyfile
import argparse
import torch
import numpy as np
import skimage.measure
import scipy
import mrcfile
import os


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3
                   
    return samples.unsqueeze(0), voxel_origin, voxel_size


def sample_generator(generator, z, max_batch=100000, voxel_resolution=256, voxel_origin=[0,0,0], cube_length=2.0, psi=0.5):
    head = 0
    samples, voxel_origin, voxel_size = create_samples(voxel_resolution, voxel_origin, cube_length)
    samples = samples.to(z.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
    
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
    transformed_ray_directions_expanded[..., -1] = -1
    
    
    generator.generate_avg_frequencies()
    with torch.no_grad():
        raw_frequencies, raw_phase_shifts = generator.siren.mapping_network(z)
        truncated_frequencies = generator.avg_frequencies + psi * (raw_frequencies - generator.avg_frequencies)
        truncated_phase_shifts = generator.avg_phase_shifts + psi * (raw_phase_shifts - generator.avg_phase_shifts)
    
    with torch.no_grad():
        while head < samples.shape[1]:
            coarse_output = generator.siren.forward_with_frequencies_phase_shifts(samples[:, head:head+max_batch], truncated_frequencies, truncated_phase_shifts, ray_directions=transformed_ray_directions_expanded[:, :samples.shape[1]-head]).reshape(samples.shape[0], -1, 4)
            sigmas[:, head:head+max_batch] = coarse_output[:, :, -1:]
            head += max_batch
    
    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
    
    return sigmas
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
    parser.add_argument('--cube_size', type=float, default=0.3)
    parser.add_argument('--voxel_resolution', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='shapes')
    opt = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = torch.load(opt.path, map_location=torch.device(device))
    ema = torch.load(opt.path.split('generator')[0] + 'ema.pth')
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()
    
    for seed in opt.seeds:
        torch.manual_seed(seed)
        
        z = torch.randn(1, 256, device=device)

        voxel_grid = sample_generator(generator, z, cube_length=opt.cube_size, voxel_resolution=opt.voxel_resolution)

        os.makedirs(opt.output_dir, exist_ok=True)
        with mrcfile.new_mmap(os.path.join(opt.output_dir, f'{seed}.mrc'), overwrite=True, shape=voxel_grid.shape, mrc_mode=2) as mrc:
            mrc.data[:] = voxel_grid