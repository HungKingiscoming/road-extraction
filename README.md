# DBRRNet for Road Extraction

## Install
```shell
pip install -r requirements.txt
```

## Dataset
Download the datasets: [Massachusetts Roads](https://huggingface.co/datasets/k4nnguyen/massachusetts-roads), [DeepGlobe Roads](https://huggingface.co/datasets/k4nnguyen/deepglobe-roads).

Create the `data` folder and put each dataset in it, keeping its `training` and `eval` folders (`eval` is the test set):
```
road-extraction
├── data
│   ├── massachusetts
│   │   ├── training
│   │   │   ├── images
│   │   │   ├── masks
│   │   ├── eval
│   │   │   ├── images
│   │   │   ├── masks
│   ├── deepglobe
│   │   ├── training
│   │   │   ├── images
│   │   │   ├── masks
│   │   ├── eval
│   │   │   ├── images
│   │   │   ├── masks
├── train.py
├── test_native.py
```

The folders `train` and `test` work as well. Inside them, images and labels can be in one folder or in sub-folders such as `images/` and `masks/`. They are paired by file name (without extension); a file is a label if its name ends with `_mask`, `_gt` or `_label`, or it is inside a folder named `labels`, `masks` or `gt`. Files without a partner are ignored.

To read a dataset from another place, pass its folder with `--data_root` (`--data-root` for `test_native.py`):
```shell
python train.py --dataset massachusetts --data_root /path/to/massachusetts
```

## Training
```shell
python train.py --dataset massachusetts --epochs 200 --save_dir ./checkpoints/mass
python train.py --dataset deepglobe --epochs 120 --save_dir ./checkpoints/dg
```

Multiple gpus for train:
```shell
torchrun --nproc_per_node=2 train.py --dataset massachusetts --save_dir ./checkpoints/mass
```

Resume:
```shell
python train.py --dataset massachusetts --resume ./checkpoints/mass/last.pt
```

## Testing
```shell
python test_native.py --ckpt ./checkpoints/mass/last.pt --dataset massachusetts
python test_native.py --ckpt ./checkpoints/dg/last.pt --dataset deepglobe
```
