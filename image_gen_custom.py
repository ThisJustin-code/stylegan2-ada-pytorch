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
            print ('warn: --class=lbl ignored when running on an unconditional network')
    return label

def generate_image(device, G, z, truncation_psi=1.0, noise_mode='const', class_idx=None):
    z = torch.from_numpy(z).to(device)
    label = grab_label(G, device, class_idx)
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

print("\nCUSTOM IMAGE GENERATOR made by JUSTIN GALLAGHER")
print("DO NOT INCLUDE QUOTATIONS IN ANY FILE OR DIRECTORY PATH NAMES!\n")
pkl_file = input('What is the .pkl file path: ')
save_path = input('Save images to which directory: ')

print(f'Loading networks from "{pkl_file}"...')
device = torch.device('cuda')
with dnnlib.util.open_url(pkl_file) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

print('Generate from a list of seeds or range of seeds?')
user = input('"r" for range, "l" for list: ')

if user == 'r':
    seed_from = input('Seed from: ')
    seed_to = input('Seed to: ')

    for i in range(int(seed_from), int(seed_to)):
      z = seed_to_vector(G, i)
      img = generate_image(device, G, z)
      img.save(os.path.join(save_path, f'{i}.png'))
      print(f"Generated Seed {i}")

elif user == 'l':
    seed_list = []
    seed_num = int(input("How many seeds would you like to input: "))
    print('Begin inputting seeds now...')
    for i in range(0, seed_num):
        seed = int(input(f'Seed #{i + 1}: '))
        seed_list.append(seed)

    for i in range(len(seed_list)):
      z = seed_to_vector(G, seed_list[i])
      img = generate_image(device, G, z)
      img.save(os.path.join(save_path, f'{seed_list[i]}.png'))
      print(f"Generated Seed {seed_list[i]}")

print('Done.')