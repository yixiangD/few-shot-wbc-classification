import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics

import myplotstyle


def plot_sample_distribution():
    lym = np.array([24, 10])
    # after augmentation
   # lym = np.array([144, 10])
    non_lym = np.array([225, 96])
    x1 = 1
    width = 0.2
    x = [x1, 2*x1]
    plt.figure()
    plt.bar(x, lym, width,
            label='Lymphocyte', color='k', edgecolor='k', alpha=0.5)
    plt.bar(x, non_lym, width,
            bottom=lym, label='Non Lymphocyte', color='r', edgecolor='k', alpha=0.5)
    #plt.text(x1, train[0] + 1.0, str(train), ha='center')
    #plt.text(2*x1, train[1] + 1.0, str(n_nonlymph), ha='center')
    plt.ylabel('Sample Count')
    plt.ylim([0, 380])
    plt.xlim([x1 - 2*width, 2*x1 + 2*width])
    plt.xticks(x, ['Train', 'Test'])
    plt.legend(loc='upper right')
    plt.savefig('../figs/sample_distribution.png')
    plt.show()

def show_stat(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    confusion = metrics.confusion_matrix(y_true, y_pred)
    print(confusion)
    df = pd.DataFrame(confusion, index=[i for i in ['LYMPHCYTE', 'NON LYMPHCYTE']],
                      columns=[i for i in ['LYMPHCYTE', 'NON LYMPHCYTE']])
    print('Prediction accuracy: {:.4f}'.format(acc))
    plt.rcParams['font.size'] = 12
    plt.figure()
    sn.heatmap(df, cmap='coolwarm', annot=True)
    pos, textvals = plt.yticks()
    plt.yticks(pos, textvals, va='center')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('../figs/init_confusion.png')
    plt.show()

def main():
    y_true = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,\
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    y_true = np.array(y_true)
    y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    y_pred = np.array(y_pred)
    plot_sample_distribution()
    #show_stat(y_true, y_pred)

if __name__ == "__main__":
    main()
