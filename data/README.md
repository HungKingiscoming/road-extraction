# Datasets

Put each downloaded dataset into its `train/` and `test/` folder:

```text
data/
├── massachusetts/
│   ├── train/     images and labels used for training
│   └── test/      images and labels used for testing
└── deepglobe/
    ├── train/
    └── test/
```

`train.py` and `test_native.py` read the folder chosen with
`--dataset massachusetts` or `--dataset deepglobe`. Inside `train/` and
`test/` the layout does not matter: the images and labels can be mixed in one
folder or kept in sub-folders such as `images/` and `labels/`. The scripts
tell them apart and pair each image with its label by file name (without
extension); files without a partner are ignored.

A file is a label when its name ends with `_mask`, `_gt` or `_label` (and
plural forms), or when it sits in a folder called `labels`, `masks`, `gt`, ...
Everything else is an image.

Dataset files in these folders are git-ignored.
