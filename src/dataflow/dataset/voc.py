import glob
import imageio

from ..base import *
from src.utils.dataflow import *

from pyUtils.utils.utils import *


class VOC(RNGDataFlow):
    def __init__(self, name, data_dir=None, shuffle=True):
        # print("dataflow voc")
        # self.data_dir = data_dir
        self.shuffle = shuffle
        self.im_dir = os.path.join(data_dir, name, 'JPEGImages')
        self.xml_dir = os.path.join(data_dir, name, 'Annotations')
        class_name_dict, class_id_dict = get_class_dict_from_xml(self.xml_dir)
        self.class_dict = class_name_dict
        # traverse_dict(class_id_dict)
        self.class_id_dict = class_id_dict
        category_index = {}
        for class_id in class_id_dict:
            category_index[class_id] = {'id': class_id, 'name': class_id_dict[class_id]}

        # traverse_dict(category_index)

        image_glob = os.path.join(self.im_dir, '*.jpg')
        self.image_files = glob.glob(image_glob)
        self._load()

        # traverse_list(self.data)
        # traverse_list(self.label)

    def _load(self):
        self.data = [-1 for i in range(len(self.image_files))]
        self.label = [-1 for i in range(len(self.image_files))]

        for idx, f in enumerate(self.image_files):
            img = imageio.imread(f)
            # print(img.shape)

            imgid = os.path.basename(f).split('.')[0]
            xml_path = os.path.join(self.xml_dir, imgid + ".xml")
            gt = self._read_xml(xml_path)
            # print(gt)
            self.data[idx] = img
            self.label[idx] = gt

        self.size = len(self.data)

    def _read_xml(self, xml_path):
        """
            Returns:
                [(class_id, [xmin, ymin, xmax, ymax])]
        """
        # [class_name, xmin, ymin, xmax, ymax]
        re = parse_bbox_xml(xml_path, self.class_dict)
        # print(re)
        # re = [1,1,1,1,1]
        # n_bbox = min(len(re), self._max_bbox)
        # self.true_boxes[self._sample_in_batch][:n_bbox] = re[:n_bbox, 1:]
        return re

    def __len__(self):
        return self.size

    def __iter__(self):
        idxs = np.arange(self.size)
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield [self.data[k], self.label[k]]
