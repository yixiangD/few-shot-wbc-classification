# Deep learning model for WBC image classification
## Quick start
```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install tensorflow==2.4.0
```
Run default code, under the following condition
1. no data augmentation
2. raw dataset
```
python src/main.py
```
## Model
Transfer learning with Mobile-Net
## Methods for Data Imbalance
- Class weighting
- Oversampling
- Mixup ([Zhang et al. 2018](https://arxiv.org/pdf/1710.09412.pdf))
- Minority Mixup: perform oversampling using mixup

# Dataset:
- Dataset [2020](https://data.mendeley.com/datasets/snkd93bnjr/1)
- Raw dataset (data/raw2/): WBC images with RBCs. We used *dataset-master* and *dataset2-master* in the [kaggle dataset](https://www.kaggle.com/paultimothymooney/blood-cells).
- Masked dataset (data/masked/): WBC images with RBCs pixels removed. In addition to the raw images, we used the corresponding annotations from [BCCD_Dataset](https://www.kaggle.com/surajiiitm/bccd-dataset).

# Reference:
- [A comprehensive guide to the DataLoader class and abstractions in Pytorch](https://blog.paperspace.com/dataloaders-abstractions-pytorch/)
