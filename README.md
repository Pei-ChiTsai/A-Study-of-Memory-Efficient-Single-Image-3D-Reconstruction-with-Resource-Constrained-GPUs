# A Study of Memory-Efficient Single-Image-3D Reconstruction with Resource-Constrained GPUs

### Getting Started
### Installation

- Python >= 3.8
- Install CUDA if available (CUDA 11.8)
- Install PyTorch according to your platform: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) **[Please make sure that the locally-installed CUDA major version matches the PyTorch-shipped CUDA major version. For example if you have CUDA 11.x installed, make sure to install PyTorch compiled with CUDA 11.x.]**
- Update setuptools by `pip install --upgrade setuptools`
- Install other dependencies by `pip install -r requirements.txt`

### Create Environment 
```sh
conda create --name MEtripoSR python=3.10
pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip3 install -r requirements.txt
```

### Dataset
We used the [Google Scanned Object(GSO) dataset](https://arxiv.org/abs/2204.11918) mentioned in previous literature, selecting 30 objects as the evaluation standard. The dataset folder contains two files: GSO_list.txt, which records the names of each object, and zip.txt, which records the names of the compressed files downloaded from the [website](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research).
```sh
A-Study-of-Memory-Efficient-Single-Image-3D-Reconstruction-with-Resource-Constrained-GPUs
|-- dataset
    |-- gso_list.txt
    |-- zip.txt
```

### The Impact of Multi-view Generation(RQ1)
<p align="center">
    <img width="450" src="figures/"/>
</p>

執行下列指令，透過[Wonder3D](https://github.com/xxlong0/Wonder3D)生成 multi-view images

#### Installation:
Download the [SAM](https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth) model. Put it to the sam_pt folder.
```sh
A-Study-of-Memory-Efficient-Single-Image-3D-Reconstruction-with-Resource-Constrained-GPUs
|-- sam_pt
    |-- sam_vit_h_4b8939.pth
```
#### Generate six images from different viewpoints using [Wonder3D](https://github.com/xxlong0/Wonder3D):
```sh
# python3 MVGen.py --img-path [image path] --name [object class] --output-path [output path]
python3 MVGen.py --img-path ./dataset/bag/thumbnails/0.jpg --name bag
python3 MVGen.py --img-path ./dataset/owl/thumbnails/0.jpg --name owl --output-path ./outputs
```

#### Generate the corresponding mesh using the six images:
```sh
# python3 run.py [output path1] [output path2] [output path3] [output path4] [output path5] [output path6] \
# --output-dir [output path] --name [view1] [view2] view3] [view4] [view5] [view6]

python3 run.py ./outputs/cropsize-220-cfg3.0/bag/front.png \
./outputs/cropsize-220-cfg3.0/bag/back.png \
./outputs/cropsize-220-cfg3.0/bag/front_left.png \
./outputs/cropsize-220-cfg3.0/bag/front_right.png \
./outputs/cropsize-220-cfg3.0/bag/left.png \
./outputs/cropsize-220-cfg3.0/bag/right.png \
 --output-dir output/six_bag \
--name six_bag_front six_bag_back six_bag_front_left six_bag_front_right six_bag_left six_bag_right
```
### Single-View Generation model(TripoSR)

We use [TripoSR](https://github.com/VAST-AI-Research/TripoSR) as the base model. The `--stack-backbone` parameter determines the number of backbone network layers used, with a default of one. The `--block-mc` parameter determines the use of block-wise marching cubes for mesh generation, with the default being the original TripoSR method. The `--render` parameter allows for rendering the generated images, creating images of the mesh from various viewpoints.

#### single image
```sh
# python3 run.py [image path] --output-dir [output path] --name [object name] --stack-backbone [backbone layer] --device [cuda:0] [--block-mc] --render

python3 run.py ./example_images/owl.png --output-dir output --name owl --render
python3 run.py ./dataset/chicken_racer/thumbnails/0.jpg --output-dir ./output/block-mc --name chicken_racer 
python3 run.py ./example_images/duola.png --output-dir output/tripo --name duola --stack-backbone 3 --render
python3 run.py ./dataset/yoshi/thumbnails/0.jpg --output-dir output/ --name yoshi --stack-backbone 1 --render
python3 run.py ./dataset/chicken_racer/thumbnails/0.jpg --output-dir ./output/block-mc --name chicken_racer --block-mc --device cpu
```

#### GSO list
```sh
# python3 run.py [list path] --output-dir [output folder] [--block-mc]
# run model using block mc method 
python3 run.py ./dataset/gso_list.txt --output-dir ./output/block-mc --block-mc
# run model using original TripoSR method 
python3 run.py ./dataset/gso_list.txt --output-dir ./output/triposr 
```
### Evaluation 

#### install [kaolin](https://github.com/NVIDIAGameWorks/kaolin) tool 
```sh
git clone https://github.com/NVIDIAGameWorks/kaolin.git
cd kaolin
pip3 install -e .
#python3 setup.py develop
#pip3 install kaolin # 0.16.0
```
#### indivisual test
```sh
# python3 single_eval.py --gt-path [mesh path of ground truth] --output-path [mesh path of output]
python3 single_eval.py --gt-path ./dataset/bag/meshes/model.obj --output-path ./output/block-mc/bag/256/bag_256.obj
```
#### group test
```sh
# python3 eval.py --caselist [list path] --gt-path [ground truth dictionary] --output-path [output dictionary] --resolution [ex:256,1024]
python3 eval.py --caselist ./dataset/gso_list.txt --gt-path ./dataset --output-path ./output/triposr --resolution 256
python3 eval.py --caselist ./dataset/gso_list.txt --gt-path ./dataset --output-path ./output/block-mc --resolution 256
```

### Evaluate memory capacity
在 system.py 裡，將@profile的註解移除，並執行下列指令，會在folder生成.dat 檔案，在利用plot指令繪製圖形

In `system.py`, remove the comment from `@profile`, and run the following instruction. A .dat file will be generated in the folder. Then, use the plot command to generate a graph.
```sh
mprof run --include-children run.py ./dataset/chicken_racer/thumbnails/0.jpg --output-dir test/triposr --mc-resolution 256 
mprof plot ./mprofile_20240819145521.dat -o ./test/comparison/256_gpu.png
```

## Troubleshooting
> AttributeError: module 'torchmcubes_module' has no attribute 'mcubes_cuda'

or

> torchmcubes was not compiled with CUDA support, use CPU version instead.

This is because `torchmcubes` is compiled without CUDA support. Please make sure that 

- The locally-installed CUDA major version matches the PyTorch-shipped CUDA major version. For example if you have CUDA 11.x installed, make sure to install PyTorch compiled with CUDA 11.x.
- `setuptools>=49.6.0`. If not, upgrade by `pip install --upgrade setuptools`.

Then re-install `torchmcubes` by:

```sh
pip uninstall torchmcubes
pip install git+https://github.com/tatsy/torchmcubes.git
```

## Citation
```BibTeX
@article{TripoSR2024,
  title={TripoSR: Fast 3D Object Reconstruction from a Single Image},
  author={Tochilkin, Dmitry and Pankratz, David and Liu, Zexiang and Huang, Zixuan and and Letts, Adam and Li, Yangguang and Liang, Ding and Laforte, Christian and Jampani, Varun and Cao, Yan-Pei},
  journal={arXiv preprint arXiv:2403.02151},
  year={2024}
}
```

conda create --name meTripoSR python=3.10
pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip3 install -r requirements.txt


evaluate 
kaolin 0.15.0
git clone https://github.com/NVIDIAGameWorks/kaolin.git 
cd kaolin
python3 setup.py develop

----------
numpy 1.26.4
memory_profiler
segment_anything
diffusers[torch]==0.19.3
xformers==0.0.16
open3d

----------
