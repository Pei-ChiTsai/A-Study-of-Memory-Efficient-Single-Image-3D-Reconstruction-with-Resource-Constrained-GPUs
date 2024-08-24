# A Study of Memory-Efficient Single-Image-3D Reconstruction with Resource-Constrained GPUs

### Getting Started
### Installation
- Python >= 3.8
- Install CUDA if available
- Install PyTorch according to your platform: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) **[Please make sure that the locally-installed CUDA major version matches the PyTorch-shipped CUDA major version. For example if you have CUDA 11.x installed, make sure to install PyTorch compiled with CUDA 11.x.]**
- Update setuptools by `pip install --upgrade setuptools`
- Install other dependencies by `pip install -r requirements.txt`

### Dataset
使用 [GSO dataset](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research)

### The Impact of Multi-view Generation(RQ1)
<p align="center">
    <img width="450" src="figures/"/>
</p>

執行下列指令，透過[Wonder3D](https://github.com/xxlong0/Wonder3D)生成 multi-view images

#### example :  
```sh
# python3 MVGen.py --img-path [image path] --name [object class] --output-path [output path]
python3 MVGen.py --img-path ./dataset/bag/thumbnails/0.jpg --name bag
python3 MVGen.py --img-path ./dataset/owl/thumbnails/0.jpg --name owl --output-path ./outputs
```

### Generation mesh application views
生成六個mesh 利用對應的多視圖圖像
#### example : 
```sh
# python3 run.py [output path1] [output path2] [output path3] [output path4] [output path5] [output path6] \
# --output-dir [output path] --name [view1] [view2] view3] [view4] [view5] [view6]

python3 run.py ./outputs/cropsize-220-cfg3.0/bag/front_left.png \
./outputs/cropsize-220-cfg3.0/bag/front_right.png \
./outputs/cropsize-220-cfg3.0/bag/left.png \
./outputs/cropsize-220-cfg3.0/bag/right.png \
 --output-dir output/six_bag \
--name six_bag_front six_bag_back six_bag_front_left six_bag_front_right six_bag_left six_bag_right
```
### run single view generation model TripoSR
執行下列指令，透過[Wonder3D](https://github.com/xxlong0/Wonder3D)生成 multi-view images
#### example :  
```sh
# python3 run.py [image path] --output-dir [output path] --name [object name] --stack-backbone [backbone layer] --device [cuda:0] --render

python3 run.py ./example_images/owl.png --output-dir output --name owl --render
python3 run.py ./dataset/chicken_racer/thumbnails/0.jpg --output-dir ./output/block-mc --name chicken_racer 
python3 run.py ./example_images/duola.png --output-dir output/tripo --name duola --stack-backbone 3 --render
python3 run.py ./dataset/yoshi/thumbnails/0.jpg --output-dir output/ --name yoshi --stack-backbone 1 --render
python3 run.py ./dataset/chicken_racer/thumbnails/0.jpg --output-dir ./output/block-mc --name chicken_racer --block-mc --device cpu
```
### Evaluation 

#### installation
安裝 [kaolin](https://github.com/NVIDIAGameWorks/kaolin)套件
```sh
git clone https://github.com/NVIDIAGameWorks/kaolin.git
cd kaolin
python3 setup.py develop
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

### 評估記憶體容量
將 system.py 裡的@profile 加上，並執行下列指令生成.dat 檔案，在繪製圖形
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
