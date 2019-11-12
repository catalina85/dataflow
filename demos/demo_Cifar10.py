import os, sys
import dataflow as df
from scipy.misc import imrotate

import utils.image_utils as image_utils


def rot90(sample):
    image = imrotate(sample[0], 90)
    return image, sample[1]


def load_Cifar10():
    data_path = "../dataset/Cifar10"
    Cifar10 = df.dataset.Cifar10("train", dir=data_path)
    Cifar10.reset_state()

    class_names = Cifar10.get_label_names()
    Cifar10_m = df.MapData(Cifar10, lambda dp: rot90(dp))
    batches = df.BatchData(Cifar10_m, 9).get_data()
    images, labels = next(batches)
    image_utils.plot_classification_data(images, labels, class_names=class_names, save_name="cifar10.png")


def Run():
    if (len(sys.argv) != 1):
        print("args error ...")
        sys.exit(0)

    load_Cifar10()


if __name__ == "__main__":
    Run()
