import sys
sys.path.insert(0, "./")
import os
import numpy as np
import PIL.Image
import torch
import dnnlib
import legacy

def seed_to_vector(G, seed):
  return np.random.RandomState(seed).randn(1, G.z_dim)

def generate_image(G, z, truncation_psi):
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    label = np.zeros([1] + G.input_shapes[1][1:])
    images = G.run(z, label, **G_kwargs)
    return images[0]

def grab_label(G, device, class_idx):
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')
    return label

def generate_image(device, G, z, truncation_psi=1.0, noise_mode='const', class_idx=None):
    z = torch.from_numpy(z).to(device)
    label = grab_label(G, device, class_idx)
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

def expand_seed(seeds, vector_size):
  result = []
  for seed in seeds:
    rnd = np.random.RandomState(seed)
    result.append(rnd.randn(1, vector_size))
  return result

print("\nLATENT VECTOR WALK VIDEO GENERATOR made by JUSTIN GALLAGHER")
print("DO NOT INCLUDE QUOTATIONS IN ANY FILE OR DIRECTORY PATH NAMES!\n")
pkl_file = input('What is the .pkl file path: ')
save_path = input('Save video to which directory: ')

print(f'Loading networks from "{pkl_file}"...')
device = torch.device('cuda')
with dnnlib.util.open_url(pkl_file) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

vector_size = G.z_dim
seeds = expand_seed( [8192+1,8192+9], vector_size)

seed_list = []
seed_num = int(input("How many seeds would you like to input: "))
print('Begin inputting seeds now...')
for i in range(0, seed_num):
    seed = int(input(f'Seed #{i+1}: '))
    seed_list.append(seed)

steps = int(input("Number of steps between seeds: "))

print('Creating temp directory for images...')
temp = os.path.join(save_path, "temp-images")
if os.path.isdir(temp):
    print('Temp directory already exists. Deleting images in:')
    os.system('del '+os.path.join(temp, "*"))
else:
    os.mkdir(temp)

print('Begin image generation...')
idx = 0
for i in range(len(seed_list) - 1):
    v1 = seed_to_vector(G, seed_list[i])
    v2 = seed_to_vector(G, seed_list[i + 1])

    diff = v2 - v1
    step = diff / steps
    current = v1.copy()

    for j in range(steps):
        current = current + step
        img = generate_image(device, G, current)
        img.save(os.path.join(temp, f'frame-{idx}.png'))
        idx += 1
        
    print(f'Completed walk between seeds {seed_list[i]} and {seed_list[i+1]}')

print('Image generation complete...')
print('Begin video processing...')


padding = input('Pad video? ("y" or "n"): ')
ffmpeg_cmd = ''
if padding == 'y':
    ffmpeg_cmd = 'ffmpeg -r 30 -i '+os.path.join(temp,'frame-%d.png')+' -vf "scale=512:512:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1" -vcodec mpeg4 -y '+os.path.join(save_path, "movie.mp4")+''
else:
    ffmpeg_cmd = 'ffmpeg -r 30 -i '+os.path.join(temp,'frame-%d.png')+' -vcodec mpeg4 -y '+os.path.join(save_path, "movie.mp4")+''
print(ffmpeg_cmd)
os.system(ffmpeg_cmd)

print("Removing all temporary files...")
del_temp_images = 'rmdir /Q /S '+temp
os.system(del_temp_images)
print('Done.')