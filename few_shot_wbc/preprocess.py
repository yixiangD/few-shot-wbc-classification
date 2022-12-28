import pandas as pd


def main():
    label_path = "./data/labels.csv"
    df_label = pd.read_csv(label_path, usecols=[1, 2])
    # select only image with non-NA labels
    df_label = df_label[~df_label.isnull().any(axis=1)]
    # select only image with one WBC
    df_label = df_label[~df_label["Category"].str.contains(",")]
    print(df_label)


if __name__ == "__main__":
    main()
