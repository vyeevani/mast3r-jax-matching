import equinox
from dust3r.utils.image import load_images

images = load_images(['dust3r/croco/assets/Chateau1.png', 'dust3r/croco/assets/Chateau2.png'], size=512)

import torch
from mast3r.model import AsymmetricMASt3R

device = 'mps'  # or 'cuda' or 'cpu'
model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
model.eval()

image_0 = images[0]['img'].to(device)
true_shape_0 = torch.from_numpy(images[0]['true_shape']).to(device)
image_1 = images[1]['img'].to(device)
true_shape_1 = torch.from_numpy(images[1]['true_shape']).to(device)

with torch.no_grad():
    feats_1, positions_1, _ = model._encode_image(image_0, true_shape_0)
    feats_2, positions_2, _ = model._encode_image(image_1, true_shape_1)
    
@torch.inference_mode
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


res11, res21 = decoder(model, feats_1, feats_2, positions_1, positions_2, true_shape_0, true_shape_1)

reference_frame_points = res11['pts3d'][0]
reference_frame_conf = res11['conf'][0]
reference_frame_desc = res11['desc'][0]
reference_frame_desc_conf = res11['desc_conf'][0]

target_frame_points = res21['pts3d'][0]
target_frame_conf = res21['conf'][0]
target_frame_desc = res21['desc'][0]
target_frame_desc_conf = res21['desc_conf'][0]

import jax
import jax.numpy as jnp
import numpy as np

# Convert torch tensors to numpy arrays, then to jax arrays
def to_jax(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return jnp.array(x)

reference_frame_points_jax = to_jax(reference_frame_points)
reference_frame_conf_jax = to_jax(reference_frame_conf)
reference_frame_desc_jax = to_jax(reference_frame_desc)
reference_frame_desc_conf_jax = to_jax(reference_frame_desc_conf)

target_frame_points_jax = to_jax(target_frame_points)
target_frame_conf_jax = to_jax(target_frame_conf)
target_frame_desc_jax = to_jax(target_frame_desc)
target_frame_desc_conf_jax = to_jax(target_frame_desc_conf)

reference_frame_rays = reference_frame_points_jax / jnp.linalg.norm(reference_frame_points_jax, axis=-1, ord=2, keepdims=True)
target_frame_rays = target_frame_points_jax / jnp.linalg.norm(target_frame_points_jax, axis=-1, ord=2, keepdims=True)

from match import match
import time
jitted_match = jax.jit(match)
n_runs = 5
matches = costs = None
times = []
for i in range(n_runs):
    start = time.time()
    matches = jitted_match(target_frame_rays, reference_frame_rays)
    end = time.time()
    times.append(end - start)
print(f"Average match() time over {n_runs} runs: {sum(times)/len(times):.4f} seconds")
print(matches.shape)


import numpy as np
import torch
from matplotlib import pyplot as pl

n_viz = 40
flat_matches = np.array(np.asarray(matches).reshape(-1, 2))  # (num_pixels, 2), (x, y) order
h, w = target_frame_points_jax.shape[:2]
y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
target_pixels = np.stack([x_coords.flatten(), y_coords.flatten()], axis=-1)  # (num_pixels, 2), (x, y) order

# Select 40 grid points over the target image
ys = np.linspace(0, h - 1, int(np.sqrt(n_viz)), dtype=int)
xs = np.linspace(0, w - 1, int(np.ceil(n_viz / len(ys))), dtype=int)
selected_indices = []
for y in ys:
    for x in xs:
        idx = y * w + x
        if idx < target_pixels.shape[0]:
            selected_indices.append(idx)
        if len(selected_indices) == n_viz:
            break
    if len(selected_indices) == n_viz:
        break
selected_indices = np.array(selected_indices)

matched_pixels = flat_matches  # (num_pixels, 2), (x, y) order
viz_matches_im1 = target_pixels[selected_indices]  # (N, 2), (x, y)
viz_matches_im0 = matched_pixels[selected_indices]  # (N, 2), (x, y)

image_mean = torch.as_tensor([0.5, 0.5, 0.5], device=device).reshape(1, 3, 1, 1)
image_std = torch.as_tensor([0.5, 0.5, 0.5], device=device).reshape(1, 3, 1, 1)

viz_imgs = []
for view in [image_0, image_1]:
    rgb_tensor = view * image_std + image_mean
    viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
img = np.concatenate((img0, img1), axis=1)
pl.figure()
pl.imshow(img)
cmap = pl.get_cmap('jet')
for i in range(len(selected_indices)):
    x0, y0 = viz_matches_im0[i]
    x1, y1 = viz_matches_im1[i]
    pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / max(1, (n_viz - 1))), scalex=False, scaley=False)
pl.show(block=True)
