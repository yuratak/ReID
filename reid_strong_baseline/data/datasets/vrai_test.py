import glob
import pickle
import re

import os.path as osp

from .bases import BaseImageDataset


class VRAI_TEST(BaseImageDataset):
    """
       VRAI
       Reference:
       Liu, Xinchen, et al. "Large-scale vehicle re-identification in urban surveillance videos." ICME 2016.

       URL:https://vehiclereid.github.io/VeRi/

       Dataset statistics:
       # identities: 6302
       # images: 66113 (train) + 1678 (query VeRi) + 11579 (gallery VeRi)
       # cameras: 2
       """

    dataset_dir = 'VRAI'

    def __init__(self, root='../', verbose=True, **kwargs):
        super(VRAI_TEST, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'images_train')
        self.query_dir = osp.join(self.dataset_dir, 'images_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'images_gallery')

        self._check_before_run()

        with open('/home/yura/data/VRAI/test_annotation.pkl', 'rb') as handle:
            order_pickle = pickle.load(handle)

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir_test(self.query_dir, order_pickle["query_order"], relabel=False)
        gallery = self._process_dir_test(self.gallery_dir, order_pickle["gallery_order"], relabel=False)

        if verbose:
            print("=> VRAI loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([0-9]+)_([0-9]+)_([0-9]+).jpg')

        pid_container = set()
        for img_path in img_paths:
            pid, _, _ = map(str, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid, _ = map(str, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    def _process_dir_test(self, dir_path, img_paths_ordered, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        #pattern = re.compile(r'([0-9]+)_([0-9]+)_([0-9]+).jpg')

        #pid_container = set()
        '''for img_path in img_paths:
            pid, _, _ = map(str, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}'''

        dataset = []
        pid = 0
        for img_path in img_paths_ordered:
            #pid, camid, _ = map(str, pattern.search(img_path).groups())
            #if pid == -1: continue  # junk images are just ignored
            #if relabel: pid = pid2label[pid]
            camid = int(img_path.split(".")[0].split("_")[-1][-1])
            img_path_full = osp.join(dir_path, img_path)
            dataset.append((img_path_full, pid, camid))
            pid += 1

        return dataset