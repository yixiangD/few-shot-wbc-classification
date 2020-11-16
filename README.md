# Deep learning model for WBC classification
## Dataset:
*dataset-master* and *dataset2-master* are recorded in the [kaggle dataset](https://www.kaggle.com/paultimothymooney/blood-cells).
*BCCD_Dataset* is recorded in the [kaggle dataset](https://www.kaggle.com/surajiiitm/bccd-dataset).
- Raw dataset: WBC images with RBCs
- Masked dataset: WBC images with RBCs pixels removed
## Model
Transfer learning with Mobile-Net
## Methods for Data Imbalance
- Class weighting
- Oversampling
- Mixup ([Zhang et al. 2018](https://arxiv.org/pdf/1710.09412.pdf))
- Minority Mixup: perform oversampling using mixup
