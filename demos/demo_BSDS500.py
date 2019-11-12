import os, sys
import dataflow as df
from scipy.misc import imrotate

import utils.image_utils as image_utils

def rot90(sample):
    print(type(sample[0]))
    print(sample[0].shape)
    image = imrotate(sample[0], 90)
    return image, sample[1]


def load_BSDS500():
    data_path = "../dataset/BSDS500"
    BSDS500 = df.dataset.BSDS500("train", data_dir=data_path, shuffle=True)
    BSDS500.reset_state()

    batches = df.BatchData(BSDS500, 8).get_data()
    images, labels = next(batches)
    image_utils.plot_semantics_data(images, labels,save_name="BSDS500_1.png")


def Run():
    if (len(sys.argv) != 1):
        print("args error ...")
        sys.exit(0)

    load_BSDS500()


if __name__ == "__main__":
    Run()
