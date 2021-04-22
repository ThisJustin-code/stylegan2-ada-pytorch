import sys
sys.path.insert(0, "./")

import sys
sys.path.insert(0, "/content/stylegan2-ada-pytorch")
import pickle
import os
import numpy as np
import PIL.Image
from IPython.display import Image
import matplotlib.pyplot as plt
import IPython.display
import torch
import dnnlib
import legacy

def seed2vec(G, seed):
  return np.random.RandomState(seed).randn(1, G.z_dim)

def generate_image(G, z, truncation_psi):
    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    label = np.zeros([1] + G.input_shapes[1][1:])
    images = G.run(z, label, **G_kwargs) # [minibatch, height, width, channel]
    return images[0]

def get_label(G, device, class_idx):
  label = torch.zeros([1, G.c_dim], device=device)
  if G.c_dim != 0:
      if class_idx is None:
          ctx.fail('Must specify class label with --class when using a conditional network')
      label[:, class_idx] = 1
  else:
      if class_idx is not None:
          print ('warn: --class=lbl ignored when running on an unconditional network')
  return label

def generate_image(device, G, z, truncation_psi=1.0, noise_mode='const', class_idx=None):
  z = torch.from_numpy(z).to(device)
  label = get_label(G, device, class_idx)
  img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
  img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
  #PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
  return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

def expand_seed(seeds, vector_size):
  result = []

  for seed in seeds:
    rnd = np.random.RandomState(seed)
    result.append( rnd.randn(1, vector_size) )
  return result

print("\nLATENT VECTOR WALK VIDEO GENERATOR made by JUSTIN GALLAGHER")
print("DO NOT INCLUDE QUOTATIONS IN ANY FILE OR DIRECTORY PATH NAMES!\n")
URL = input('What is the .pkl file path: ')

print(f'Loading networks from "{URL}"...')
device = torch.device('cuda')
with dnnlib.util.open_url(URL) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

vector_size = G.z_dim
# range(8192,8300)
seeds = expand_seed( [8192+1,8192+9], vector_size)
#generate_images(Gs, seeds,truncation_psi=0.5)
print(seeds[0].shape)

# Choose your seeds to morph through and the number of steps to take to get to each.

SEEDS = []
SEEDS_NUM = int(input("How many seeds would you like to input: "))
for i in range(0, SEEDS_NUM):
    SEED = int(input(f'Seed #{i+1}: '))
    SEEDS.append(SEED)  # adding the individual seeds to seed list

STEPS = int(input("Number of steps between seeds: "))
save_path = input('Save to which directory: ')

temp = os.path.join(save_path, "temp-images")
if os.path.isdir(temp):
    print('Temp directory already exists. Deleting images in:')
    os.system('del '+os.path.join(temp, "*"))
else:
    os.mkdir(temp)

# Generate the images for the video.
idx = 0
for i in range(len(SEEDS) - 1):
    v1 = seed2vec(G, SEEDS[i])
    v2 = seed2vec(G, SEEDS[i + 1])

    diff = v2 - v1
    step = diff / STEPS
    current = v1.copy()

    for j in range(STEPS):
        current = current + step
        img = generate_image(device, G, current)
        img.save(os.path.join(temp, f'frame-{idx}.png'))
        idx += 1

    print(f"Completed Seed {SEEDS[i]}")

# Link the images into a video.

cmd = 'ffmpeg -r 30 -i '+os.path.join(temp,'frame-%d.png')+' -vcodec mpeg4 -y '+os.path.join(save_path, "movie.mp4")+''
os.system(cmd)
print('Done.')