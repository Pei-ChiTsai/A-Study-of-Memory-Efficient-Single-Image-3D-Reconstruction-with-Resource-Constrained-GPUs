import argparse
import pickle
import logging
import os
import time

import numpy as np
import rembg
import torch
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from memory_profiler import profile

@profile
class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")
        return t


timer = Timer()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
parser = argparse.ArgumentParser()
parser.add_argument("image", type=str, nargs="+", help="Path to input image(s).")
parser.add_argument(
    "--device",
    default="cuda:0",
    type=str,
    help="Device to use. If no CUDA-compatible device is found, will fallback to 'cpu'. Default: 'cuda:0'",
)
parser.add_argument(
    "--pretrained-model-name-or-path",
    default="stabilityai/TripoSR",
    type=str,
    help="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/TripoSR'",
)
parser.add_argument(
    "--chunk-size",
    default=8192,
    type=int,
    help="Evaluation chunk size for surface extraction and rendering. Smaller chunk size reduces VRAM usage but increases computation time. 0 for no chunking. Default: 8192",
)
parser.add_argument(
    "--mc-resolution",
    default=256,
    type=int,
    help="Marching cubes grid resolution. Default: 256"
)
parser.add_argument(
    "--no-remove-bg",
    action="store_true",
    help="If specified, the background will NOT be automatically removed from the input image, and the input image should be an RGB image with gray background and properly-sized foreground. Default: false",
)
parser.add_argument(
    "--foreground-ratio",
    default=0.85,
    type=float,
    help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
)
parser.add_argument(
    "--output-dir",
    default="output/",
    type=str,
    help="Output directory to save the results. Default: 'output/'",
)
parser.add_argument(
    "--model-save-format",
    default="obj",
    type=str,
    choices=["obj", "glb"],
    help="Format to save the extracted mesh. Default: 'obj'",
)
parser.add_argument(
    "--render",
    action="store_true",
    help="If specified, save a NeRF-rendered video. Default: false",
)
parser.add_argument(
    "--name",
    default="mesh",
    type=str,
    nargs="+",
    help="name of mesh",
)
parser.add_argument(
    "--stack-backbone",
    default=1,
    type = int,
    help="stack backbone",
)
parser.add_argument(
    "--block-mc",
    action="store_true",
    help="If specified, save a NeRF-rendered video. Default: false",
)

args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

device = args.device
if not torch.cuda.is_available():
    device = "cpu"
timer.start("all")
timer.start("Initializing model")
model = TSR.from_pretrained(
    args.pretrained_model_name_or_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(args.chunk_size)
model.to(device)
timer.end("Initializing model")

timer.start("Processing images")
images = []

if args.no_remove_bg:
    rembg_session = None
else:
    rembg_session = rembg.new_session()

if len(args.image)==1 and args.image[0].endswith(".txt"):
        img = []
        name_list = []
        with open(args.image[0],'r') as f:
            content = f.readlines()
            for line in content:
                case = line.strip()
                img.append(f"./dataset/{case}/thumbnails/0.jpg")
                name_list.append(case)
        args.image = img          
        args.name = name_list
else:
    args.image = args.image
    args.name = args.name

print("image : ",args.image)
print("file name : ", args.name)
for i, image_path in enumerate(args.image):
    if args.no_remove_bg:
        image = np.array(Image.open(image_path).convert("RGB"))
    else:
        image = remove_background(Image.open(image_path), rembg_session)
        image = resize_foreground(image, args.foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        if not os.path.exists(os.path.join(output_dir, args.name[i]+f"/{args.mc_resolution}")):
            os.makedirs(os.path.join(output_dir, args.name[i]+f"/{args.mc_resolution}"))
        image.save(os.path.join(output_dir, args.name[i], f"{args.mc_resolution}/input.png"))
    images.append(image)
timer.end("Processing images")

run_time = []
render_time = []
mesh_time = []

for i, image in enumerate(images):
    logging.info(f"Running image {i + 1}/{len(images)} ...")

    timer.start("Running model")
    with torch.no_grad():
        scene_codes = model([image], device=device,args = args)
    run_time.append(timer.end("Running model"))

    if args.render:
        timer.start("Rendering")
        render_images = model.render(scene_codes, n_views=15, return_type="pil")
        if not os.path.exists(os.path.join(output_dir,args.name[i], f'{args.mc_resolution}/render')):
            os.makedirs(os.path.join(output_dir,args.name[i], f'{args.mc_resolution}/render'))
        
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(output_dir, args.name[i], f"{args.mc_resolution}/render/{ri:03d}.png"))
        # save_video(
        #     render_images[0], os.path.join(output_dir, args.name[i], f"render/render.mp4"), fps=10
        # )
        render_time.append(timer.end("Rendering"))

    timer.start("Exporting mesh")
    meshes = model.extract_mesh(scene_codes, resolution=args.mc_resolution,block = args.block_mc)
    meshes[0].export(os.path.join(output_dir, args.name[i], f"{args.mc_resolution}/{args.name[i]}_{args.mc_resolution}.{args.model_save_format}"))
    mesh_time.append(timer.end("Exporting mesh"))
all_time = timer.end("all")

print("avg time :",all_time/len(args.image))
print("use case is ",args.name)
print("run time : ",run_time)
print("avg run : ",sum(run_time)/len(run_time))
if(args.render):
    print("render time : ",render_time)
    print("avg render : ",sum(render_time)/len(render_time))
print("mesh time : ",mesh_time)
print("avg mesh : ",sum(mesh_time)/len(mesh_time))


    
