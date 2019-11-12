import os, sys
import src.dataflow as df
from scipy.misc import imrotate
import utils.image_utils as image_utils


def rot90(sample):
    image = imrotate(sample[0], 180)
    return image, sample[1]


def load_minst():
    data_path = "../dataset/mnist/mnist0"
    mnist = df.dataset.Mnist("train", shuffle=True, dir=data_path)

    mnist.reset_state()
    mnist_map = df.MapData(mnist, lambda dp: rot90(dp))
    batches = df.BatchData(mnist_map, 9, remainder=True).get_data()
    images, labels = next(batches)
    image_utils.plot_classification_data(images, labels,save_name="mnist.png")


def Run():
    if (len(sys.argv) != 1):
        print("args error ...")
        sys.exit(0)

    load_minst()


if __name__ == "__main__":
    Run()
