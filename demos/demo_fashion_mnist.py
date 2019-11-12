import os, sys
import src.dataflow as df
from scipy.misc import imrotate
import utils.image_utils as image_utils


def rot90(sample):
    image = imrotate(sample[0], 0)
    return image, sample[1]


def load_fashion_mnist():
    data_path = "../dataset/fashion_mnist_data"
    fashion_mnist = df.dataset.FashionMnist("train", dir=data_path)
    fashion_mnist.reset_state()

    fashion_mnist_batch = df.MapData(fashion_mnist, lambda dp: rot90(dp))

    class_names = fashion_mnist.get_label_names()
    batches = df.BatchData(fashion_mnist_batch, 9, remainder=True).get_data()
    images, labels = next(batches)
    image_utils.plot_classification_data(images, labels, class_names=class_names,save_name="fashion_mnist.png")


def Run():
    if (len(sys.argv) != 1):
        print("args error ...")
        sys.exit(0)

    load_fashion_mnist()


if __name__ == "__main__":
    Run()
