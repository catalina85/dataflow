from src.dataflow.common import BatchData
import utils.image_utils as image_utils
from pyUtils.utils.utils import *
from scipy.misc import imrotate

import src.dataflow as df
import time

from data_aug.data_aug import *
from data_aug.bbox_util import *

if __name__ == "__main__":
    data_dir = "/Users/tianm/materials/deepLearning/others/voc/VOC2007"
    voc = df.dataset.VOC('train_part', data_dir)
    voc.reset_state()

    class_name_dict = voc.class_id_dict
    traverse_dict(class_name_dict)

    batches = BatchData(voc, 2, use_list=True).get_data()
    images, labels = next(batches)

    out_dir = "./decetion/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(len(images)):
        now = time.time()
        img = images[i]
        classes = labels[i][:, 0]
        bboxes = labels[i][:, 1:]
        score = [0 for i in range(len(classes))]
        # image_utils.plot_detection_data(img, classes, score, bboxes, class_name_dict, save_name=out_dir +str(now) + "_ori.png")

        # img_, bboxes_ = RandomRotate(90)(img.copy(), bboxes.copy())
        # image_utils.plot_detection_data(img_, classes, score, bboxes_, class_name_dict, save_name=out_dir +str(now) + "_aug1.png")

        seq = Sequence(
            [RandomHSV(40, 40, 30), RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomRotate(20),
             RandomShear()])
        img_, bboxes_ = seq(img.copy(), bboxes.copy())
        image_utils.plot_detection_data(img_, classes, score, bboxes_, class_name_dict, save_name=out_dir +str(now) + "_aug2.png")


