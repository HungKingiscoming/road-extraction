# DBRRNet for Road Extraction

## Install
```shell
pip install -r requirements.txt
```

## Dataset
Put the images and labels of each dataset into its `train` and `test` folder:
```
road-extraction
├── data
│   ├── massachusetts
│   │   ├── train
│   │   ├── test
│   ├── deepglobe
│   │   ├── train
│   │   ├── test
├── train.py
├── test_native.py
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
