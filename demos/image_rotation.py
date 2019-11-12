from pyUtils.utils.utils import *
import src.dataflow as df
import utils.image_utils as image_utils

if __name__ == "__main__":
    data_dir = "/Users/tianm/materials/deepLearning/others/voc/VOC2007"
    voc = df.dataset.VOC('train_part', data_dir)

    voc.reset_state()
    class_name_dict = voc.class_id_dict
    traverse_dict(class_name_dict)

    aug = df.imgaug.Rotation(25.0)
    aug = df.imgaug.AugmentorList([aug])
    voc = df.AugmentImageCoordinates(voc, augmentors=aug)

    batches = df.BatchData(voc, 5, use_list=True).get_data()
    images, labels = next(batches)
    print(images)
    print(labels)
    print(type(labels))

    for i in range(len(images)):
        img = images[i]
        classes = labels[i][:, 0]
        bboxes = labels[i][:, 1:]
        score = [0 for i in range(len(classes))]

        image_utils.plot_detection_data(img, classes, score, bboxes, class_name_dict)
