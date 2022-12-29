# Deep learning model for WBC image classification
## Preprocess data
### Dataset
1. Dataset used in the paper is available [Github](https://github.com/akshaylamba/all_CELL_data).
1. A dataset for microscopic peripheral blood cell images for development of automatic recognition systems [a recently published large dataset](https://data.mendeley.com/datasets/snkd93bnjr/1) (optional for
  testing)
  
```
upzip all_Cell_data-master.zip
mv all_Cell_data-master ./data
```
### Prepare clearly labeled dataset and generate cropped images
```
python few_shot_wbc/preprocess.py
```

## Methodology

### CNN Model
Transfer learning with Mobile-Net

### Methods for Data Imbalance
- Class weighting
- Oversampling
- Mixup ([Zhang et al. 2018](https://arxiv.org/pdf/1710.09412.pdf))
- Minority Mixup: perform oversampling using mixup


## Reference:
- [A comprehensive guide to the DataLoader class and abstractions in Pytorch](https://blog.paperspace.com/dataloaders-abstractions-pytorch/)
