"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

"""
Composite data-loaders derived from class specific data loaders
"""
import torch
from torch.utils import data
from torch.autograd import Variable
import numpy as np

import sys

sys.path.insert(0, '../')
from common.pose_utils import calc_vos_simple, calc_vos_safe
from dataset_loaders.utils import *


class MF(data.Dataset):
    """
    Returns multiple consecutive frames, and optionally VOs
    """
    
    def __init__(self, dataset, include_vos=False, no_duplicates=False,
                 *args, **kwargs):
        """
        :param steps: Number of frames to return on every call
        :param skip: Number of frames to skip
        :param variable_skip: If True, skip = [1, ..., skip]
        :param include_vos: True if the VOs have to be appended to poses. If real
        and include_vos are both on, it gives absolute poses from GT and VOs from
        the SLAM / DSO
        :param no_duplicates: if True, does not duplicate frames when len(self) is
        not a multiple of skip*steps
        """
        self.steps = kwargs.pop('steps', 2)
        self.skip = kwargs.pop('skip', 1)
        self.variable_skip = kwargs.pop('variable_skip', False)
        self.real = kwargs.pop('real', False)
        self.include_vos = include_vos
        self.train = kwargs['train']
        self.vo_func = kwargs.pop('vo_func', calc_vos_simple)
        self.no_duplicates = no_duplicates
        
        if dataset == '7Scenes':
            from dataset_loaders.seven_scenes import SevenScenes
            self.dset = SevenScenes(*args, real=self.real, **kwargs)
            if self.include_vos and self.real:
                self.gt_dset = SevenScenes(*args, skip_images=True, real=False,
                                           **kwargs)
        elif dataset == 'RobotCar':
            from dataset_loaders.robotcar import RobotCar
            self.dset = RobotCar(*args, real=self.real, **kwargs)
            if self.include_vos and self.real:
                self.gt_dset = RobotCar(*args, skip_images=True, real=False,
                                        **kwargs)
        elif dataset == 'RIO10':
            from dataset_loaders.rio10 import RIO10
            self.dset = RIO10(*args, **kwargs)
        else:
            raise NotImplementedError
        
        self.L = self.steps * self.skip
    
    def get_indices(self, index):
        if self.variable_skip:
            skips = np.random.randint(1, high=self.skip + 1, size=self.steps - 1)
        else:
            # 尺寸为[2]
            skips = self.skip * np.ones(self.steps - 1)
        # np.insert在给定索引之前沿给定轴插入值
        # np.cumsum返回给定轴上元素的累计和
        # 尺寸为[3]
        offsets = np.insert(skips, 0, 0).cumsum()
        offsets -= offsets[len(offsets) // 2]
        if self.no_duplicates:
            offsets += self.steps // 2 * self.skip
        offsets = offsets.astype(np.int)
        # 尺寸为[3]
        idx = index + offsets
        idx = np.minimum(np.maximum(idx, 0), len(self.dset) - 1)
        assert np.all(idx >= 0), '{:d}'.format(index)
        assert np.all(idx < len(self.dset))
        return idx
    
    def __getitem__(self, index):
        """
        :param index:
        :return: imgs: STEPS x 3 x H x W
                 poses: STEPS x 7
                 vos: (STEPS-1) x 7 (only if include_vos = True)
        """
        # 尺寸为[3]
        idx = self.get_indices(index)
        # 列表，包含3个元组
        clip = [self.dset[i] for i in idx]
        
        # 尺寸为[STEPS,3,H,W]
        imgs = torch.stack([c[0] for c in clip], dim=0)
        # 尺寸为[STEPS,6]
        poses = torch.stack([c[2] for c in clip], dim=0)
        if self.include_vos:
            # vos = calc_vos_simple(poses.unsqueeze(0))[0] if self.train else \
            #   calc_vos_safe(poses.unsqueeze(0))[0]
            vos = self.vo_func(poses.unsqueeze(0))[0]
            if self.real:  # absolute poses need to come from the GT dataset
                clip = [self.gt_dset[self.dset.gt_idx[i]] for i in idx]
                poses = torch.stack([c[2] for c in clip], dim=0)
            poses = torch.cat((poses, vos), dim=0)
        metas = [c[1] for c in clip]
        return imgs, metas, poses
    
    def __len__(self):
        L = len(self.dset)
        if self.no_duplicates:
            L -= (self.steps - 1) * self.skip
        return L


class MFOnline(data.Dataset):
    """
    Returns a minibatch of train images with absolute poses and test images
    with real VOs
    """
    
    def __init__(self, gps_mode=False, *args, **kwargs):
        self.gps_mode = gps_mode
        self.train_set = MF(train=True, *args, **kwargs)
        self.val_set = MF(train=False, include_vos=(not gps_mode), real=True,
                          vo_func=calc_vos_safe, no_duplicates=True, *args,
                          **kwargs)
    
    def __getitem__(self, idx):
        train_idx = idx % len(self.train_set)
        train_ims, train_poses = self.train_set[train_idx]
        val_idx = idx % len(self.val_set)
        val_ims, val_vos = self.val_set[val_idx]  # val_vos contains abs poses if gps_mode
        if not self.gps_mode:
            val_vos = val_vos[len(val_ims):]
        ims = torch.cat((train_ims, val_ims))
        poses = torch.cat((train_poses, val_vos))
        return ims, poses
    
    def __len__(self):
        return len(self.val_set)


class OnlyPoses(data.Dataset):
    """
    Returns real poses aligned with GT poses
    """
    
    def __init__(self, dataset, *args, **kwargs):
        kwargs = dict(kwargs, skip_images=True)
        if dataset == '7Scenes':
            from seven_scenes import SevenScenes
            self.real_dset = SevenScenes(*args, real=True, **kwargs)
            self.gt_dset = SevenScenes(*args, real=False, **kwargs)
        elif dataset == 'RobotCar':
            from robotcar import RobotCar
            self.real_dset = RobotCar(*args, real=True, **kwargs)
            self.gt_dset = RobotCar(*args, real=False, **kwargs)
        else:
            raise NotImplementedError
    
    def __getitem__(self, index):
        """
        :param index:
        :return: poses: 2 x 7
        """
        _, real_pose = self.real_dset[index]
        _, gt_pose = self.gt_dset[self.real_dset.gt_idx[index]]
        
        return real_pose, gt_pose
    
    def __len__(self):
        return len(self.real_dset)


def flatten(t):
    res=[]
    for subt in t:
        if isinstance(subt,(list,tuple)):
            res.extend(flatten(subt))
        else:
            res.append(subt)
    return res


if __name__ == '__main__':
    from common.vis_utils import show_batch, show_stereo_batch
    from torchvision.utils import make_grid
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

    kwargs = dict(data_path='../data/deepslam_data/RIO10', train=True,
                  transform=transform, target_transform=target_transform,
                  mode=3, scene='1',
                  steps=4,
                  )
    dset = MF('RIO10', **kwargs)
    # print(dset[0])
    data_loader = data.DataLoader(dset, batch_size=3, shuffle=False,
                                  num_workers=0, collate_fn=new_collate)
    
    for batch in data_loader:
        rgb, meta, target = batch
        if dset.dset.mode == 0:  # not load meta
            assert meta is None
            print(rgb.shape, target.shape)
        elif dset.dset.mode == 3:  # load meta
            assert meta is not None
            for meta_sample_in_batch in meta:
                for meta_step_in_sample in meta_sample_in_batch:
                    print(
                        meta_step_in_sample['bbox'].shape,
                        meta_step_in_sample['fea'].shape,
                        meta_step_in_sample['msk'].shape,
                    )
        break
