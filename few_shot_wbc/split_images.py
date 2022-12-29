import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    path = os.path.join("./data", "all_CELL_data-master")
    # arguments: fold, image_type
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", action="store_true", help="enable shuffle indice")
    parser.add_argument(
        "--nfold",
        type=int,
        default=5,
        choices=[5, 10],
        help="number of fold split in train and test dataset",
    )
    parser.add_argument(
        "--num_class",
        type=int,
        choices=[2, 4],
        default=2,
        help="number of classes, 2 for binary and 4 for all classes excluding basophil",
    )
    parser.add_argument(
        "--crop", action="store_true", help="enable to use cropped images with only WBC"
    )
    args = parser.parse_args()
    # print(args)

    for root, dirs, files in os.walk(path):
        root, dirs, files = root, dirs, files
        break

    # reading table of labels and image names
    clean_path = "./data/labels_clean.csv"
    df_clean = pd.read_csv(clean_path, usecols=[1, 2])
    df_clean["Lympho"] = np.where(
        df_clean["Category"] == "LYMPHOCYTE", "LYMPHOCYTE", "nonLYMPHOCYTE"
    )
    # print(df_clean)

    # create a folder to hold the images under "data"
    fold_name = f"{args.num_class}class_{args.nfold}fold"
    if args.shuffle:
        fold_name += "_shuffle"
        df = df_clean.sample(frac=1).reset_index(drop=True)
    else:
        df = df_clean.copy()
    if args.crop:
        fold_name += "_crop"

    # train_path = os.path.join("./data", fold_name, "train")
    # test_path = os.path.join("./data", fold_name, "test")

    sampler = StratifiedShuffleSplit(
        n_splits=1, test_size=1 / float(args.nfold), random_state=0
    )
    if args.num_class == 2:
        lab = "Lympho"
    else:
        lab = "Category"
        df = df[df["Category"] != "BASOPHIL"]
        print(Counter(df["Category"]))
    for i, (train_id, test_id) in enumerate(
        sampler.split(df["Image"].values, df[lab].values)
    ):
        print(f"fold {i}")
        print(f"Train: {train_id}")
        print(f"Test: {test_id}")


if __name__ == "__main__":
    main()
