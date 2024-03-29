# Deep learning model for WBC image classification
Repository for paper
```
@article{deng2023deep,
  title={Deep learning for few-shot white blood cell image classification and feature learning},
  author={Deng, Yixiang and Li, He},
  journal={Computer Methods in Biomechanics and Biomedical Engineering: Imaging \& Visualization},
  pages={1--11},
  year={2023},
  publisher={Taylor \& Francis}
}
```
If you find this repository helpful, we kindly encourage you to cite our paper listed above.

Please feel free to create issues if you need help running the code. We will try to help you as much as we can.
## Preprocess data
### Dataset
1. Dataset used in the paper is publicly available on [Github](https://github.com/akshaylamba/all_CELL_data). We also used the labels of WBC class for the same dataset on [kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells?resource=download), located at ```dataset-master/dataset-master/labels.csv```.
1. A dataset for microscopic peripheral blood cell images for development of automatic recognition systems [a recently published large dataset](https://data.mendeley.com/datasets/snkd93bnjr/1) (optional for
  testing)
  
```
mkdir ./data # at the root
# upzip all_Cell_data-master.zip
mv all_Cell_data-master/ ./data
mv labels.csv ./data
```
### Prepare clearly labeled dataset and generate cropped images
Run the following from the root directory to generate 

- a table recording the image indices with known WBC labels and the cropping coordinates in ```data/df_coord.csv```, note that images with more than two WBCs are excluded,
- corresponding cropped images in ```data/all_Cell_data-master/crop*.jpg```.

```
python few_shot_wbc/preprocess.py
```
The expected cell counts for each class,
<!---
|Cell type| Cell count|
| --- | ----------- |
|NEUTROPHIL| 205|
|EOSINOPHIL|87|
|LYMPHOCYTE| 33|
|MONOCYTE| 20| 
|BASOPHIL| 3|
-->

|Cell type| NEUTROPHIL|EOSINOPHIL|LYMPHOCYTE|MONOCYTE|BASOPHIL|
| --- | ----------- |----------- |----------- |----------- |----------- |
|Cell count| 203|83|33|20| 3|

### Spilt images into train and test folder
Here, we create train and test folders for our model. Available arguments:

- ```--shuffle```, enable image index shuffle,
- ```--nfold```, int, available choice [5, 10], for 5-fold splitting or 10-fold, default is 5,
- ```--num_class```, int, available choice [2, 4], for binary classification (lymphocyte or not) or four-class classification (four WBCs excluding basophil, since there is too few of them), default is 2,
- ```--crop```, if added, use cropped WBC-only images otherwise use the orignal images with RBCs.

```
python few_shot_wbc/split_images.py
```

## Methodology

### CNN Model
Transfer learning with 

- AlexNet
- VGG19
- ResNet152
- DenseNet
- MobileNetV2
- ResNext101_32x8d
- EfficientNet b3


### Methods for Data Imbalance
- Class weighting: [pytorch](https://discuss.pytorch.org/t/dealing-with-imbalanced-datasets-in-pytorch/22596)
- Traindata resampling: [dataset sampler](https://github.com/ufoym/imbalanced-dataset-sampler)
- Minority Mixup: perform oversampling using mixup ([paper](https://arxiv.org/pdf/1710.09412.pdf), [code](https://github.com/facebookresearch/mixup-cifar10)).


## Reference:
- [A comprehensive guide to the DataLoader class and abstractions in Pytorch](https://blog.paperspace.com/dataloaders-abstractions-pytorch/)
