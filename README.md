# Deep learning model for WBC image classification
## Quick start
```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install tensorflow==2.4.0
```
## Dataset:
- Raw dataset (data/raw2/): WBC images with RBCs. We used *dataset-master* and *dataset2-master* in the [kaggle dataset](https://www.kaggle.com/paultimothymooney/blood-cells).
- Masked dataset (data/masked/): WBC images with RBCs pixels removed. In addition to the raw images, we used the corresponding annotations from [BCCD_Dataset](https://www.kaggle.com/surajiiitm/bccd-dataset).
## Model
Transfer learning with Mobile-Net
## Methods for Data Imbalance
- Class weighting
- Oversampling
- Mixup ([Zhang et al. 2018](https://arxiv.org/pdf/1710.09412.pdf))
- Minority Mixup: perform oversampling using mixup
