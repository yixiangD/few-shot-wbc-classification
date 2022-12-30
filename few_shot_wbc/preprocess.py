import os
import xml.etree.ElementTree as ET
from collections import Counter

import pandas as pd
from PIL import Image


def main():
    clean_path = "./data/labels_clean.csv"
    label_path = "./data/labels.csv"
    if os.path.exists(clean_path):
        df_label = pd.read_csv(clean_path, usecols=[1, 2])
    else:
        df_label = pd.read_csv(label_path, usecols=[1, 2])
        # select only image with non-NA labels
        df_label = df_label[~df_label.isnull().any(axis=1)]
        # drop image without annotations
        missing_img = [96, 116, 280, 329]
        df_label = df_label[~df_label["Image"].isin(missing_img)]
        # select only image with one WBC
        df_label = df_label[~df_label["Category"].str.contains(",")]
        df_label.to_csv(clean_path)
    print(Counter(df_label["Category"]))
    # creating cropped image
    # read xml file to get coordinates of wbc
    crop_coord_path = "./data/wbc_coord.csv"
    if os.path.exists(crop_coord_path):
        print(f"Loading from file {crop_coord_path}")
        df_coord = pd.read_csv(crop_coord_path)
    else:
        df_coord = get_crop_coord(df_label, crop_coord_path)
    print(len(df_coord))
    print(df_coord["Image"])
    # load image and crop
    crop0 = os.path.join("./data", "all_CELL_data-master", "crop_00000.jpg")
    if os.path.exists(crop0):
        print(f"Found cropped image in {crop0}, exiting.")
        exit()

    for i in range(df_coord.shape[0]):
        fname = "_".join(["BloodImage", str(df_coord.loc[i]["Image"]).zfill(5)])
        img_file = fname + ".jpg"
        img_path = os.path.join("./data", "all_CELL_data-master", img_file)
        img = Image.open(img_path)
        # print(df_coord.loc[i])
        coord = (
            df_coord.loc[i]["xmin"],
            df_coord.loc[i]["ymin"],
            df_coord.loc[i]["xmax"],
            df_coord.loc[i]["ymax"],
        )
        img2 = img.crop(coord)
        fname = "_".join(["crop", str(df_coord.loc[i]["Image"]).zfill(5)])
        img2_file = fname + ".jpg"
        img2_path = os.path.join("./data", "all_CELL_data-master", img2_file)
        img2.save(img2_path)


def get_crop_coord(df_label, out_path):
    list_coord = []
    for i in df_label["Image"]:
        fname = "_".join(["BloodImage", str(i).zfill(5)])
        annot_file = fname + ".xml"
        # print(df_label.iloc[i, :])

        annot_path = os.path.join("./data", "all_CELL_data-master", annot_file)
        tree = ET.parse(annot_path)
        root = tree.getroot()
        all_cell = []
        for group in root.findall("object"):
            cell_type = group.find("name")
            all_cell.append(cell_type.text)
            if cell_type.text == "WBC":
                coord = group.find("bndbox")
                xmin = coord.find("xmin").text
                xmax = coord.find("xmax").text
                ymin = coord.find("ymin").text
                ymax = coord.find("ymax").text
                list_coord.append([i, xmin, xmax, ymin, ymax])
        if "WBC" not in all_cell:
            print(f"Did not found WBC in image {i}")
    df_coord = pd.DataFrame(
        list_coord, columns=["Image", "xmin", "xmax", "ymin", "ymax"]
    )
    df_coord.to_csv(out_path, index=False)
    print(f"Saving to file {out_path}")
    return df_coord


if __name__ == "__main__":
    main()
