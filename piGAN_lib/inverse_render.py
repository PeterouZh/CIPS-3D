import argparse
import math
import os

from torchvision.utils import save_image

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
import skvideo.io
import curriculums
from torchvision import transforms

def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('generator_path', type=str)
parser.add_argument('image_path', type=str)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--num_frames', type=int, default=64)
parser.add_argument('--max_batch_size', type=int, default=2400000)


opt = parser.parse_args()

generator = torch.load(opt.generator_path, map_location=torch.device(device))
ema_file = opt.generator_path.split('generator')[0] + 'ema.pth'
ema = torch.load(ema_file, map_location=device)
ema.copy_to(generator.parameters())
generator.set_device(device)
generator.eval()

if opt.seed is not None:
    torch.manual_seed(opt.seed)

gt_image = Image.open(opt.image_path).convert('RGB')
transform = transforms.Compose(
                    [transforms.Resize(256), transforms.CenterCrop(256), transforms.Resize((opt.image_size, opt.image_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

gt_image = transform(gt_image).to(device)

    
options = {
    'img_size': opt.image_size,
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'num_steps': 24,
    'h_stddev': 0,
    'v_stddev': 0,
    'h_mean': torch.tensor(math.pi/2).to(device),
    'v_mean': torch.tensor(math.pi/2).to(device),
    'hierarchical_sample': False,
    'sample_dist': None,
    'clamp_mode': 'relu',
    'nerf_noise': 0,
}

render_options = {
    'img_size': 256,
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'num_steps': 48,
    'h_stddev': 0,
    'v_stddev': 0,
    'v_mean': math.pi/2,
    'hierarchical_sample': True,
    'sample_dist': None,
    'clamp_mode': 'relu',
    'nerf_noise': 0,
    'last_back': True,
}

z = torch.randn((10000, 256), device=device)
with torch.no_grad():
    frequencies, phase_shifts = generator.siren.mapping_network(z)
w_frequencies = frequencies.mean(0, keepdim=True)
w_phase_shifts = phase_shifts.mean(0, keepdim=True)

w_frequency_offsets = torch.zeros_like(w_frequencies)
w_phase_shift_offsets = torch.zeros_like(w_phase_shifts)
w_frequency_offsets.requires_grad_()
w_phase_shift_offsets.requires_grad_()

frames = []

n_iterations_pose = 0
n_iterations = 700

os.makedirs('debug', exist_ok=True)
save_image(gt_image, "debug/gt.jpg", normalize=True)

                
optimizer = torch.optim.Adam([w_frequency_offsets, w_phase_shift_offsets], lr=1e-2, weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.75)

for i in range(n_iterations):
    noise_w_frequencies = 0.03 * torch.randn_like(w_frequencies) * (n_iterations - i)/n_iterations
    noise_w_phase_shifts = 0.03 * torch.randn_like(w_phase_shifts) * (n_iterations - i)/n_iterations
    frame, _ = generator.forward_with_frequencies(w_frequencies + noise_w_frequencies + w_frequency_offsets, w_phase_shifts + noise_w_phase_shifts + w_phase_shift_offsets, **options)
    loss = torch.nn.MSELoss()(frame, gt_image)
    loss = loss.mean()
    

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    
    scheduler.step()

    if i % 100 == 0:
        save_image(frame, f"debug/{i}.jpg", normalize=True)
        with torch.no_grad():
            for angle in [-0.7, -0.5, -0.3, 0, 0.3, 0.5, 0.7]:
                img, _ = generator.staged_forward_with_frequencies(w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, h_mean=(math.pi/2 + angle), max_batch_size=opt.max_batch_size, lock_view_dependence=True, **render_options)
                save_image(img, f"debug/{i}_{angle}.jpg", normalize=True)

trajectory = [] 
for t in np.linspace(0, 1, 24):
    pitch = 0.2 * t
    yaw = 0
    trajectory.append((pitch, yaw))
for t in np.linspace(0, 1, opt.num_frames):
    pitch = 0.2 * np.cos(t * 2 * math.pi)
    yaw = 0.4 * np.sin(t * 2 * math.pi)
    trajectory.append((pitch, yaw))
        
output_name = 'reconstructed.mp4'
writer = skvideo.io.FFmpegWriter(os.path.join('debug', output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})

frames = []
depths = []


with torch.no_grad():
    for pitch, yaw in tqdm(trajectory):
        render_options['h_mean'] = yaw + 3.14/2
        render_options['v_mean'] = pitch + 3.14/2

        frame, depth_map = generator.staged_forward_with_frequencies(w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, max_batch_size=opt.max_batch_size, lock_view_dependence=True, **render_options)
        frames.append(tensor_to_PIL(frame))
        
        depths.append(depth_map.unsqueeze(0).expand(-1, 3, -1, -1).squeeze().permute(1, 2, 0).cpu().numpy())

for frame in frames:
    writer.writeFrame(np.array(frame))
writer.close()