import argparse
import os
import cv2 #
import time #
import torch
import requests
from PIL import Image
import numpy as np
from rembg import remove #
import rembg #
from segment_anything import sam_model_registry, SamPredictor
from mvdiffusion.data.single_image_dataset import SingleImageDataset
from einops import rearrange #

from torchvision.utils import make_grid, save_image
from diffusers import DiffusionPipeline  # only tested on diffusers[torch]==0.19.3, may have conflicts with newer versions of diffusers
from io import BytesIO

from datetime import datetime #

_GPU_ID = 0

def save_image(input, fp=None):
                """
                    Converts a PyTorch tensor or NumPy array to an image, with an option to save it to disk.

                    Parameters:
                    - input: A PyTorch tensor or NumPy array. If it's a tensor, its shape should be [C, H, W], and the data range should be [0, 1].
                            If it's a NumPy array, its shape should be [H, W, C] and its data type should be uint8.
                    - fp: The file path where the image will be saved. If None, the image will not be saved to disk.

                    Returns:
                    - ndarr: The converted NumPy array, with shape [H, W, C] and data type uint8.
                """
                if isinstance(input, torch.Tensor):
                    ndarr = input.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                elif isinstance(input, np.ndarray):
                    ndarr = input
                else:
                    raise TypeError("The input type is not supported; it should be a PyTorch tensor or a NumPy array.")

                if fp:
                    im = Image.fromarray(ndarr)
                    im.save(fp)
                return ndarr

def sam_init():
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "sam_pt", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=f"cuda:{_GPU_ID}")
    predictor = SamPredictor(sam)
    return predictor

def sam_segment(predictor, input_image, *bbox_coords):
    bbox = np.array(bbox_coords)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(box=bbox, multimask_output=True)

    print(f"SAM Time: {time.time() - start_time:.3f}s")
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode='RGBA')

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def preprocess(predictor, input_image, chk_group=None, segment=True, rescale=False):
    RES = 1024
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    # if chk_group is not None:
    #     segment = "Background Removal" in chk_group
    #     rescale = "Rescale" in chk_group
    if segment:
        image_rem = input_image.convert('RGBA')
        image_nobg = remove(image_rem, alpha_matting=True)
        arr = np.asarray(image_nobg)[:, :, -1]
        x_nonzero = np.nonzero(arr.sum(axis=0))
        y_nonzero = np.nonzero(arr.sum(axis=1))
        x_min = int(x_nonzero[0].min())
        y_min = int(y_nonzero[0].min())
        x_max = int(x_nonzero[0].max())
        y_max = int(y_nonzero[0].max())
        input_image = sam_segment(predictor, input_image.convert('RGB'), x_min, y_min, x_max, y_max)
    # Rescale and recenter
    if rescale:
        image_arr = np.array(input_image)
        in_w, in_h = image_arr.shape[:2]
        out_res = min(RES, max(in_w, in_h))
        ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        max_size = max(w, h)
        ratio = 0.75
        side_len = int(max_size / ratio)
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len // 2
        padded_image[center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w] = image_arr[y : y + h, x : x + w]
        rgba = Image.fromarray(padded_image).resize((out_res, out_res), Image.LANCZOS)

        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
        input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        input_image = expand2square(input_image, (127, 127, 127, 0))
    return input_image, input_image.resize((320, 320), Image.Resampling.LANCZOS)

def load_wonder3d_pipeline():

    pipeline = DiffusionPipeline.from_pretrained(
    'flamehaze1115/wonder3d-v1.0', # or use local checkpoint './ckpts'
    custom_pipeline='flamehaze1115/wonder3d-pipeline',
    torch_dtype=torch.float16
    )

    # enable xformers
    pipeline.unet.enable_xformers_memory_efficient_attention()

    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    return pipeline

def prepare_data(single_image, crop_size):
    dataset = SingleImageDataset(root_dir='', num_views=6, img_wh=[256, 256], bg_color='white', crop_size=crop_size, single_image=single_image)
    return dataset[0]

def create_mask_from_removed_bg(image_array):
    
    alpha_channel = image_array[:, :, 3]
    mask = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)
    return mask

def run_pipeline(pipeline, single_image, guidance_scale=3, steps=50, seed=42, crop_size=220, chk_group='Write Results',name = 'mesh', output_path = "./outputs"): # crop_size 192
    import pdb
    global scene
    # pdb.set_trace()

    if chk_group is not None:
        write_image = "Write Results" in chk_group

    batch = prepare_data(single_image, crop_size)

    pipeline.set_progress_bar_config(disable=True)
    seed = int(seed)
    generator = torch.Generator(device=pipeline.unet.device).manual_seed(seed)

    # repeat  (2B, Nv, 3, H, W)
    imgs_in = torch.cat([batch['imgs_in']] * 2, dim=0).to(torch.float16)

    # (2B, Nv, Nce)
    camera_embeddings = torch.cat([batch['camera_embeddings']] * 2, dim=0).to(torch.float16)

    task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0).to(torch.float16)

    camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1).to(torch.float16)

    # (B*Nv, 3, H, W)
    imgs_in = rearrange(imgs_in, "Nv C H W -> (Nv) C H W")
    # (B*Nv, Nce)
    # camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

    out = pipeline(
        imgs_in,
        # camera_embeddings,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        output_type='pt',
        num_images_per_prompt=1,
    ).images

    bsz = out.shape[0] // 2
    normals_pred = out[:bsz]
    images_pred = out[bsz:]
    num_views = 6
    if write_image:
        VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        cur_dir = os.path.join(output_path, f"cropsize-{int(crop_size)}-cfg{guidance_scale:.1f}")

        if(name != 'mesh'):
            scene = str(name)
        else:
            scene = 'scene'+datetime.now().strftime('@%Y%m%d-%H%M%S')
        scene_dir = os.path.join(cur_dir, scene)
        normal_dir = os.path.join(scene_dir, "normals")
        masked_dir = os.path.join(scene_dir, "masked")
        masked_colors_dir = os.path.join(scene_dir, "masked_colors")
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(masked_colors_dir, exist_ok=True)
        os.makedirs(masked_dir, exist_ok=True)
        for j in range(num_views):
            view = VIEWS[j]
            normal = normals_pred[j]
            color = images_pred[j]

            normal_filename = f"normal_{view}.png"
            rgb_filename = f"{view}.png"
            masked_filename=f"masked_{view}.png"

            normal_numpy = save_image(normal, os.path.join(normal_dir, normal_filename))
            color_numpy = save_image(color, os.path.join(scene_dir, rgb_filename))

            # rm_normal = remove(normal_numpy)
            rm_color = remove(color_numpy)
            
            #masked black-white image
            masked = create_mask_from_removed_bg(rm_color)

            # save_image(rm_normal, os.path.join(scene_dir, normal_filename))
            save_image(rm_color, os.path.join(masked_colors_dir, rgb_filename))
            save_image(masked, os.path.join(masked_dir, masked_filename))

    normals_pred = [save_image(normals_pred[i]) for i in range(bsz)]
    images_pred = [save_image(images_pred[i]) for i in range(bsz)]

    out = images_pred + normals_pred
    return out

def load_image(image_path_or_url):
    
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

pipeline = load_wonder3d_pipeline()
pipeline.to(f'cuda:{_GPU_ID}')
predictor = sam_init()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--img-path",
    default="",
    type=str,
)
parser.add_argument(
    "--output-path",
    default="./outputs",
    type=str,
)
parser.add_argument(
    "--name",
    default="mesh",
    type=str,
    help="name of mesh",
)
args = parser.parse_args()
input_image = load_image(args.img_path)

processed_image_highres, processed_image = preprocess(predictor=predictor, input_image = input_image)

print("run pipeline")
run_pipeline(pipeline=pipeline,single_image=processed_image_highres,name=args.name,output_path = args.output_path)