import random

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from . import stereo_trans as ST


class StereoBatchAggDataset(Dataset):
    def __init__(self, data_cfg, scope='train'):
        super().__init__()
        self.data_cfg = data_cfg
        self.is_train = scope == 'train'
        self.scope = scope.lower()
        self.dataset = None
        self.transform = None
        self.image_reader_type = data_cfg.get('image_reader', 'PIL')
        self.disp_reader_type = data_cfg.get('disp_reader', 'PIL')
        self.return_right_disp = data_cfg.get('return_right_disp', False)
        self.return_occ_mask = data_cfg.get('return_occ_mask', False)
        # for batch uniform
        self.batch_uniform = data_cfg.get('batch_uniform', False)
        self.random_type = data_cfg.get('random_type', None)
        self.w_range = data_cfg.get('w_range', None)
        self.h_range = data_cfg.get('h_range', None)
        self.random_crop_index = None  # for batch uniform random crop, record the index of the crop transform operator
        self.build_dataset()
        self.aug_reg_type = self.data_cfg['aug_reg_type']

    def build_dataset(self):
        if self.data_cfg['name'] in ['KITTI2012', 'KITTI2015']:
            if "test" in self.scope:
                from data.reader.kitti_reader import KittiTestReader
                self.disp_reader_type = 'PIL'
                self.dataset = KittiTestReader(
                    self.data_cfg['root'],
                    self.data_cfg['test_list'],
                    self.image_reader_type,
                    self.disp_reader_type,
                    right_disp=False,
                    use_noc=False,
                )
            else:
                from data.reader.kitti_reader import KittiReader
                self.disp_reader_type = 'PIL'
                # Instantiate the KittiReader
                self.dataset = KittiReader(
                    self.data_cfg['root'],
                    self.data_cfg[f'{self.scope}_list'],
                    self.image_reader_type,
                    self.disp_reader_type,
                    right_disp=self.return_right_disp,
                    use_noc=self.data_cfg['use_noc'] if 'use_noc' in self.data_cfg else False,  # NOC disp or OCC disp
                )
        elif self.data_cfg['name'] in ['KITTI2012&2015']:
            if "test" in self.scope:
                from data.reader.kitti_reader import KittiTestReader
                self.disp_reader_type = 'PIL'
                dataset_2012 = KittiTestReader(
                    self.data_cfg['root2012'],
                    self.data_cfg['test_list_2012'],
                    self.image_reader_type,
                    self.disp_reader_type,
                    right_disp=False,
                    use_noc=False,
                )
                dataset_2015 = KittiTestReader(
                    self.data_cfg['root2015'],
                    self.data_cfg['test_list_2015'],
                    self.image_reader_type,
                    self.disp_reader_type,
                    right_disp=False,
                    use_noc=False,
                )
                test_on = self.data_cfg.get('test_on', '2015')
                if test_on == 2015:
                    self.dataset = dataset_2015
                elif test_on == 2012:
                    self.dataset = dataset_2012
                elif test_on == 'all':
                    self.dataset = torch.utils.data.ConcatDataset([dataset_2012, dataset_2015])
                else:
                    raise NotImplementedError(f'test on: {test_on} is not supported yet.')
            else:
                from data.reader.kitti_reader import KittiReader
                self.disp_reader_type = 'PIL'
                dataset_2012 = KittiReader(
                    self.data_cfg['root2012'],
                    self.data_cfg[f'{self.scope}_list_2012'],
                    self.image_reader_type,
                    self.disp_reader_type,
                    right_disp=self.return_right_disp,
                    use_noc=self.data_cfg['use_noc'] if 'use_noc' in self.data_cfg else False,  # NOC disp or OCC disp
                )
                dataset_2015 = KittiReader(
                    self.data_cfg['root2015'],
                    self.data_cfg[f'{self.scope}_list_2015'],
                    self.image_reader_type,
                    self.disp_reader_type,
                    right_disp=self.return_right_disp,
                    use_noc=self.data_cfg['use_noc'] if 'use_noc' in self.data_cfg else False,  # NOC disp or OCC disp
                )
                if self.scope == 'train':
                    self.dataset = torch.utils.data.ConcatDataset([dataset_2012, dataset_2015])
                else:
                    val_on = self.data_cfg.get('val_on', 2015)
                    if val_on == 2015:
                        self.dataset = dataset_2015
                    elif val_on == 2012:
                        self.dataset = dataset_2012
                    elif val_on == 'all':
                        self.dataset = torch.utils.data.ConcatDataset([dataset_2012, dataset_2015])
                    else:
                        raise NotImplementedError(f'val on: {val_on} is not supported yet.')
        elif self.data_cfg['name'] == 'FlyingThings3DSubset':
            from data.reader.sceneflow_reader import FlyingThings3DSubsetReader
            self.disp_reader_type = 'PFM'
            self.return_right_disp = True
            self.return_occ_mask = True
            self.dataset = FlyingThings3DSubsetReader(
                self.data_cfg['root'],
                self.data_cfg[f'{self.scope}_list'],
                self.image_reader_type,
                self.disp_reader_type,
                right_disp=self.return_right_disp,
                occ_mask=self.return_occ_mask
            )
        elif self.data_cfg['name'] == 'SceneFlow':
            from data.reader.sceneflow_reader import SceneFlowReader
            self.disp_reader_type = 'PFM'
            self.dataset = SceneFlowReader(
                self.data_cfg['root'],
                self.data_cfg[f'{self.scope}_list'],
                self.image_reader_type,
                self.disp_reader_type,
                right_disp=self.return_right_disp,
            )
        elif self.data_cfg['name'] == 'DrivingStereo':
            from data.reader.driving_reader import DrivingReader
            self.return_right_disp = False
            self.return_occ_mask = False
            self.disp_reader_type = 'PIL'
            self.dataset = DrivingReader(
                self.data_cfg['root'],
                self.data_cfg[f'{self.scope}_list'],
                self.image_reader_type,
                self.disp_reader_type
            )
        elif self.data_cfg['name'] == 'Middlebury':
            from data.reader.middlebury_reader import MiddleburyReader
            self.disp_reader_type = 'PFM'
            self.dataset = MiddleburyReader(
                self.data_cfg['root'],
                self.data_cfg[f'{self.scope}_list'],
                self.image_reader_type,
                self.disp_reader_type
            )
        elif self.data_cfg['name'] == 'ETH3D':
            from data.reader.eth3d_reader import ETH3DReader
            self.disp_reader_type = 'PFM'
            self.dataset = ETH3DReader(
                self.data_cfg['root'],
                self.data_cfg[f'{self.scope}_list'],
                self.image_reader_type,
                self.disp_reader_type
            )
        else:
            name = self.data_cfg['name']
            raise NotImplementedError(f'{name} dataset is not supported yet.')
        self.build_transform()

    def build_transform(self):
        transform_config = self.data_cfg['transform']
        # Create a dictionary to map scope to the corresponding configuration
        scope_to_config = {
            'train': transform_config['train'],
            'val': transform_config.get('val', transform_config.get('test')),
            'test': transform_config.get('test'),
        }
        # Get the configuration based on the scope
        config = scope_to_config.get(self.scope)
        if config is None:
            # If the scope is not supported, raise a ValueError
            raise NotImplementedError(f'{self.scope} is not supported yet.')
        self.transform = self.build_transform_by_cfg(config)

    def __getitem__(self, indexs):
        # set the image_size for this batch
        if self.batch_uniform and self.scope == 'train':
            base_size = self.transform.transforms[self.random_crop_index].size
            size = self.get_crop_size(base_size)
            self.transform.transforms[self.random_crop_index].size = size

        batch_result = {}

        if isinstance(indexs, int):
            indexs = [indexs]

        for index in indexs:
            sample = self.dataset[index]
            result = self.transform(sample)
            # results key has dict_keys(['left', 'right', 'disp', 'disp_right'])
            left_image = result['left'].clone()
            left_image = left_image.permute(1,2,0).numpy().astype(np.uint8)
            if self.aug_reg_type == 'sift':
                sift_aux, sift_aux_gauss = extract_Sift_feature(left_image)
                # print("sift_aux ", np.max(sift_aux), np.min(sift_aux), np.shape(sift_aux))
                # sift_aux  255.0 0.0 (320, 736)
                tmp_aug = np.expand_dims(sift_aux_gauss, 0)
            if 'aug_reg' not in batch_result:
                batch_result['aug_reg'] = tmp_aug
            else:
                batch_result['aug_reg'] = np.concatenate([batch_result['aug_reg'], tmp_aug], 0)
            
            for each_item in result:
                if isinstance(result[each_item], np.ndarray):
                    tmp = np.expand_dims(result[each_item], 0)
                    if each_item not in batch_result:
                        batch_result[each_item] = tmp
                    else:
                        batch_result[each_item] = np.concatenate([batch_result[each_item], tmp], 0)
                else:
                    tmp = torch.unsqueeze(result[each_item], 0) if torch.is_tensor(result[each_item]) else result[
                        each_item]
                    if each_item not in batch_result:
                        batch_result[each_item] = tmp
                    else:
                        batch_result[each_item] = torch.cat([batch_result[each_item], tmp], 0)
        batch_result['index'] = indexs
        return batch_result

    def build_transform_by_cfg(self, transform_config):
        transform_compose = []
        for trans in transform_config:
            if trans['type'] == 'CenterCrop':
                transform_compose.append(ST.CenterCrop(trans['size']))
            elif trans['type'] == 'TestCrop':
                transform_compose.append(ST.TestCrop(trans['size']))
            elif trans['type'] == 'CropOrPad':
                transform_compose.append(ST.CropOrPad(trans['size']))
            elif trans['type'] == 'StereoPad':
                transform_compose.append(ST.StereoPad(trans['size']))
            elif trans['type'] == 'DivisiblePad':
                transform_compose.append(ST.DivisiblePad(trans['by'], trans.get('mode', 'single')))
            elif trans['type'] == 'RandomCrop':
                transform_compose.append(ST.RandomCrop(trans['size']))
                self.random_crop_index = len(transform_compose) - 1
            elif trans['type'] == 'RandomHorizontalFlip':
                assert self.return_right_disp, 'RandomHorizontalFlip is used, but return_right_disp is False.'
                transform_compose.append(ST.RandomHorizontalFlip(p=trans['prob']))
            elif trans['type'] == 'GetValidDispNOcc':
                transform_compose.append(ST.GetValidDispNOcc())
            elif trans['type'] == 'GetValidDisp':
                transform_compose.append(ST.GetValidDisp(trans['max_disp']))
            elif trans['type'] == 'TransposeImage':
                transform_compose.append(ST.TransposeImage())
            elif trans['type'] == 'ToTensor':
                transform_compose.append(ST.ToTensor())
            elif trans['type'] == 'NormalizeImage':
                transform_compose.append(ST.NormalizeImage(trans['mean'], trans['std']))
            elif trans['type'] == 'FlowAugmentor':
                transform_compose.append(ST.FlowAugmentor(trans['size'], trans['min_scale'], trans['max_scale'], trans['do_flip']))
            elif trans['type'] == 'SparseFlowAugmentor':
                transform_compose.append(ST.SparseFlowAugmentor(trans['size'], trans['min_scale'], trans['max_scale'], trans['do_flip']))
            elif trans['type'] == 'ColorTransform':
                transform_compose.append(ST.ColorTransform(trans['range']))
            elif trans['type'] == 'EraserTransform':
                transform_compose.append(ST.EraserTransform(trans['prob']))
            elif trans['type'] == 'SpatialTransform':
                transform_compose.append(ST.SpatialTransform(trans['size'], trans['min_scale'], trans['max_scale'], trans['do_flip']))
            elif trans['type'] == 'RandomFlip':
                transform_compose.append(ST.RandomFlip(trans['do_flip_type'], trans['h_flip_prob'], trans['v_flip_prob']))

        return ST.Compose(transform_compose)

    def get_crop_size(self, base_size):
        if self.random_type == 'range_for_sttr':
            w = random.randint(640, 960)
            h = random.randint(360, 640)
        elif self.random_type == 'range':
            w = random.randint(self.w_range[0] * base_size[1], self.w_range[1] * base_size[1])
            h = random.randint(self.h_range[0] * base_size[0], self.h_range[1] * base_size[0])
        elif self.random_type == 'choice':
            w = random.choice(self.w_range) if isinstance(self.w_range, list) else self.w_range
            h = random.choice(self.h_range) if isinstance(self.h_range, list) else self.h_range
        else:
            raise NotImplementedError
        return int(h), int(w)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collect_fn(batch):
        return batch[0]


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center_x, center_y, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x = center_x
    y = center_y

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def extract_Sift_feature(img, gaussian_radius= 5, show_case=False):
    '''

    :param image_path: input image
    :param show_case: plt.imshow(img) or not
    :return: sift keypoint binary mask and gaussian heatmap
    '''

    # img = cv2.imread(image_path)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img, None)
    # generate binary mask (0 for background, 1 for sift keypoint)
    binary_mask = np.zeros(shape=(img.shape[0], img.shape[1]))
    gaussian_mask = np.zeros(shape=(img.shape[0], img.shape[1]))
    for k_p in kp1:
        binary_mask[int(k_p.pt[1]), int(k_p.pt[0])] = 1
        draw_gaussian(gaussian_mask, int(k_p.pt[0]), int(k_p.pt[1]), radius=gaussian_radius)

    '''
    print(np.count_nonzero(binary_mask))
    print(np.count_nonzero(gaussian_mask), np.unique(gaussian_mask))
    629
    59674 [0.00000000e+00 5.88451217e-04 2.24472210e-03 6.35920926e-03
     8.56277826e-03 1.33792593e-02 2.09049649e-02 2.42580135e-02
     5.10368881e-02 6.87219964e-02 7.97446503e-02 9.25352812e-02
     1.44585493e-01 2.25913453e-01 2.62148806e-01 3.04196123e-01
     4.75303537e-01 5.51539774e-01 7.42657239e-01 8.61775631e-01
     1.00000000e+00]
    '''

    if show_case:
        import matplotlib.pyplot as plt
        kp_image1 = cv2.drawKeypoints(img, kp1, None)
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(kp_image1)
        plt.subplot(1, 3, 2)
        plt.imshow(binary_mask)
        plt.subplot(1, 3, 3)
        plt.imshow(gaussian_mask)
        plt.show()

    return binary_mask* 255. , gaussian_mask* 255.

