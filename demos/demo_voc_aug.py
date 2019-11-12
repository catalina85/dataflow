from src.dataflow.common import BatchData
import utils.image_utils as image_utils
from pyUtils.utils.utils import *
from scipy.misc import imrotate

import src.dataflow as df


def rot90(sample):
    image = imrotate(sample[0], 90)
    return image, sample[1]


if __name__ == "__main__":
    data_dir = "/Users/tianm/materials/deepLearning/others/voc/VOC2007"
    voc = df.dataset.VOC('train_part', data_dir)
    voc.reset_state()

    class_name_dict = voc.class_id_dict
    traverse_dict(class_name_dict)

    aug = df.imgaug.Rotation(25.0)
    aug = df.imgaug.AugmentorList([aug])
    voc = df.AugmentImageCoordinates(voc, augmentors=aug)

    batches = BatchData(voc, 5, use_list=True).get_data()
    images, labels = next(batches)
    print(images)
    print(labels)
    print(type(labels))
    print(len(images))

    for i in range(len(images)):
        img = images[i]
        classes = labels[i][:, 0]
        bboxes = labels[i][:, 1:]
        score = [0 for i in range(len(classes))]
        print(classes)
        print(bboxes)
        image_utils.plt_bboxes1(img, classes, score, bboxes, class_name_dict)
