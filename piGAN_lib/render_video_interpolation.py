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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
parser.add_argument('--output_dir', type=str, default='vids')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_batch_size', type=int, default=2400000)
parser.add_argument('--depth_map', action='store_true')
parser.add_argument('--lock_view_dependence', action='store_true')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--ray_step_multiplier', type=int, default=2)
parser.add_argument('--num_frames', type=int, default=36)
parser.add_argument('--curriculum', type=str, default='CelebA')
parser.add_argument('--trajectory', type=str, default='front')
parser.add_argument('--psi', type=float, default=0.7)
opt = parser.parse_args()

os.makedirs(opt.output_dir, exist_ok=True)

curriculum = getattr(curriculums, opt.curriculum)
curriculum['num_steps'] = curriculum[0]['num_steps'] * opt.ray_step_multiplier
curriculum['img_size'] = opt.image_size
curriculum['psi'] = opt.psi
curriculum['v_stddev'] = 0
curriculum['h_stddev'] = 0
curriculum['lock_view_dependence'] = opt.lock_view_dependence
curriculum['last_back'] = curriculum.get('eval_last_back', False)
curriculum['num_frames'] = opt.num_frames
curriculum['nerf_noise'] = 0
curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

class FrequencyInterpolator:
    def __init__(self, generator, z1, z2, psi=0.5):
        avg_frequencies, avg_phase_shifts = generator.generate_avg_frequencies()
        raw_frequencies1, raw_phase_shifts1 = generator.siren.mapping_network(z1)
        self.truncated_frequencies1 = avg_frequencies + psi * (raw_frequencies1 - avg_frequencies)
        self.truncated_phase_shifts1 = avg_phase_shifts + psi * (raw_phase_shifts1 - avg_phase_shifts)
        raw_frequencies2, raw_phase_shifts2 = generator.siren.mapping_network(z2)
        self.truncated_frequencies2 = avg_frequencies + psi * (raw_frequencies2 - avg_frequencies)
        self.truncated_phase_shifts2 = avg_phase_shifts + psi * (raw_phase_shifts2 - avg_phase_shifts)

    def forward(self, t):
        frequencies = self.truncated_frequencies1 * (1-t) + self.truncated_frequencies2 * t
        phase_shifts = self.truncated_phase_shifts1 * (1-t) + self.truncated_phase_shifts2 * t

        return frequencies, phase_shifts

def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

generator = torch.load(opt.path, map_location=torch.device(device))
ema_file = opt.path.split('generator')[0] + 'ema.pth'
ema = torch.load(ema_file)
ema.copy_to(generator.parameters())
generator.set_device(device)
generator.eval()

if opt.trajectory == 'front':
    trajectory = []
    for t in np.linspace(0, 1, curriculum['num_frames']):
        pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2
        yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2

        fov = curriculum['fov'] + 5 + np.sin(t * 2 * math.pi) * 5

        trajectory.append((t, pitch, yaw, fov))
elif opt.trajectory == 'orbit':
    trajectory = []
    for t in np.linspace(0, 1, curriculum['num_frames']):
        pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/4
        yaw = t * 2 * math.pi
        fov = curriculum['fov']

        trajectory.append((t, pitch, yaw, fov))


output_name = f'interp.mp4'
writer = skvideo.io.FFmpegWriter(os.path.join(opt.output_dir, output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})

print(opt.seeds)

for i, seed in enumerate(opt.seeds):
    frames = []
    depths = []

    torch.manual_seed(seed)
    z_current = torch.randn(1, 256, device=device)

    torch.manual_seed(opt.seeds[(i+1)%len(opt.seeds)])
    z_next = torch.randn(1, 256, device=device)

    frequencyInterpolator = FrequencyInterpolator(generator, z_current, z_next, psi=opt.psi)

    with torch.no_grad():
        for t, pitch, yaw, fov in tqdm(trajectory):
            curriculum['h_mean'] = yaw# + 3.14/2
            curriculum['v_mean'] = pitch# + 3.14/2
            curriculum['fov'] = fov
            curriculum['h_stddev'] = 0
            curriculum['v_stddev'] = 0
            frame, depth_map = generator.staged_forward_with_frequencies(*frequencyInterpolator.forward(t), max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
#             frame, depth_map = generator.staged_forward(z, max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
            frames.append(tensor_to_PIL(frame))

        for frame in frames:
            writer.writeFrame(np.array(frame))

writer.close()
