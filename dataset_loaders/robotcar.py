"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import sys
import os.path as osp
from torch.utils import data
import numpy as np
sys.path.insert(0, './')
sys.path.insert(0, '../')
from dataset_loaders.utils import *
from dataset_loaders.robotcar_sdk.interpolate_poses import interpolate_vo_poses, \
    interpolate_ins_poses
from dataset_loaders.robotcar_sdk.camera_model import CameraModel
from dataset_loaders.robotcar_sdk.image import load_image
import dataset_loaders.utils
from dataset_loaders.utils import new_collate, flatten, plt_imshow, resize_bbox, permutation_generator
from functools import partial
from common.pose_utils import process_poses
import pickle, torch
from torchvision.datasets.folder import default_loader

filename_dict = {'loop1': '2014-06-23-15-41-25',
                 'loop2': '2014-06-23-15-36-04',
                 'full1': '2014-12-09-13-21-02',
                 'full2': '2014-12-12-10-45-15',
                 'full3': '2014-11-25-09-18-32',
                 'full4': '2014-12-16-09-14-09',
                 'full5': '2015-10-30-13-52-14',
                 'full01': '2014-11-28-12-07-13',
                 'full02': '2014-12-02-15-30-08'}

class RobotCar(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None, target_transform=None,
                 real=False, skip_images=False, seed=7, log=0, num_obj_cut=0,
                 undistort=False, vo_lib='stereo', mode=0):
        """
        :param scene: e.g. 'full' or 'loop'. collection of sequences.
        :param data_path: Root RobotCar data directory.
        Usually '../data/deepslam_data/RobotCar'
        :param train: flag for training / validation
        :param transform: Transform to be applied to images
        :param target_transform: Transform to be applied to poses
        :param real: if True, load poses from SLAM / integration of VO
        :param skip_images: return None images, only poses
        :param seed: random seed
        :param undistort: whether to undistort images (slow)
        :param vo_lib: Library to use for VO ('stereo' or 'gps')
        (`gps` is a misnomer in this code - it just loads the position information
        from GPS)
        """
        np.random.seed(seed)
        self.mode = mode
        assert mode in [0, 3], 'only accept mode=0 and 3'
        self.train = train
        self.transform = transform
        if target_transform is None:
            from torchvision import transforms as tv_transforms
            target_transform = tv_transforms.Lambda(lambda x: torch.from_numpy(x).float())
        self.target_transform = target_transform
        self.skip_images = skip_images
        self.log = log
        self.num_obj_cut = num_obj_cut
        self.undistort = undistort
        data_path = osp.abspath(data_path)
        base_scene = re.sub(r'\d', '', scene)
        base_dir = osp.abspath(osp.expanduser(osp.join(data_path, base_scene)))
        data_dir = base_dir
        
        # decide which sequences to use
        if train:
            split_filename = osp.join(base_dir, 'train_split.txt')
        else:
            split_filename = osp.join(base_dir, 'test_split.txt')
        if not train and scene in filename_dict:
            seqs = []
            seqs.append(filename_dict[scene])
        else:
            with open(split_filename, 'r') as f:
                seqs = [l.rstrip() for l in f if not l.startswith('#')]
        
        ps = {}
        ts = {}
        vo_stats = {}
        self.imgs = []
        for seq in seqs:
            seq_dir = osp.join(base_dir, seq)
            seq_data_dir = osp.join(data_dir, seq)
            
            # read the image timestamps
            ts_filename = osp.join(seq_dir, 'stereo.timestamps')
            with open(ts_filename, 'r') as f:
                ts[seq] = [int(l.rstrip().split(' ')[0]) for l in f]
            
            if real:  # poses from integration of VOs
                if vo_lib == 'stereo':
                    vo_filename = osp.join(seq_dir, 'vo', 'vo.csv')
                    p = np.asarray(interpolate_vo_poses(vo_filename, ts[seq], ts[seq][0]))
                elif vo_lib == 'gps':
                    vo_filename = osp.join(seq_dir, 'gps', 'gps_ins.csv')
                    p = np.asarray(interpolate_ins_poses(vo_filename, ts[seq], ts[seq][0]))
                else:
                    raise NotImplementedError
                vo_stats_filename = osp.join(seq_data_dir, '{:s}_vo_stats.pkl'.
                                             format(vo_lib))
                with open(vo_stats_filename, 'rb') as f:
                    vo_stats[seq] = pickle.load(f, encoding='iso-8859-1')
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
            else:  # GT poses
                pose_filename = osp.join(seq_dir, 'gps', 'ins.csv')
                p = np.asarray(interpolate_ins_poses(pose_filename, ts[seq], ts[seq][0]))
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
            if undistort:
                prefix = osp.join(seq_dir, 'stereo', 'centre')
            else:
                prefix = osp.join(seq_dir, 'stereo', 'centre_processed')
            assert osp.exists(prefix), prefix
            self.imgs.extend([osp.join(prefix, '{:d}.png'.format(t)) for t in ts[seq]])
        
        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        assert not real, 'we do not use vo'
        if train and not osp.exists(pose_stats_filename):
            # 尺寸为[3]
            mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)
            # 尺寸为[3]
            std_t = np.std(poses[:, [3, 7, 11]], axis=0)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)
        
        # convert the pose to translation + log quaternion, align, normalize
        self.poses = np.empty((0, 6))
        for seq in seqs:
            # 尺寸为[n,6]
            pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))
        self.gt_idx = np.asarray(range(len(self.poses)))
        
        # camera model and image loader
        camera_model = CameraModel(osp.abspath(data_path + '/../robotcar_camera_models/'),
                                   osp.join('stereo', 'centre'))
        # partial返回一个新的partial对象，该对象在被调用时的行为将类似于使用位置参数args和关键字参数arguments调用的func
        self.im_loader = partial(load_image, model=camera_model)
    
    @staticmethod
    def imgfn2metafn(imgfn, log=0):
        lst = imgfn.split('/')
        sec = lst[-2] + '.npz'
        if log > 0:
            res = '/'.join(lst[:-2]) + '/' + str(log) + '/' + sec + '/' + lst[-1].replace('.png', '.npz')
        else:
            res = '/'.join(lst[:-2]) + '/' + sec + '/' + lst[-1].replace('.png', '.npz')
        imgpn = osp.dirname(res)
        assert osp.exists(imgpn), imgpn
        assert osp.exists(res), res
        return res
    
    def load_img(self, index):
        import dataset_loaders.utils
        img = None
        while img is None:
            if self.undistort:
                # img = dataset_loaders.utils.load_image(self.imgs[index], loader=self.im_loader)
                img = self.im_loader(self.imgs[index])
            else:
                img = dataset_loaders.utils.load_image(self.imgs[index])
            index += 1
        index -= 1
        return img, index
    
    def __getitem__(self, index):
        '''
        :param index: index of img
        :return: img, meta, pose
        mode == 0, rgb, None, pose
        mode == 3, rgb, meta, pose
        '''
        img = pose = meta = None
        if self.skip_images:
            img = None
            pose = self.poses[index]
        else:
            img, index = self.load_img(index)
            # if index >=297:
            #     print(self.imgs[index])
            pose = self.poses[index]
        if self.mode == 3:
            metafn = self.imgfn2metafn(self.imgs[index], self.log)
            npz = np.load(metafn)
            # fea = torch.from_numpy(npz['fea'].astype('float32'))
            bbox = npz['bbox'].astype('float32')
            # the loaded bbox is mat format, center point + size
            # row col height width
            # in a 256x256 img
            # we need to resize it to 256x341 (height = 256, width = 341)
            # 256x341 is resize from 960x1280
            bbox = resize_bbox(bbox, (256, 256), (256, 341))
            if self.train:
                perm = permutation_generator(bbox, self.num_obj_cut)
            else:
                perm = torch.arange(len(bbox))
            meta = {'bbox': torch.from_numpy(bbox)[perm],
                    # 'msk': torch.from_numpy(npz['msk'].astype('uint8')),
                    # 'fea': fea[perm],
                    'idx': torch.from_numpy(npz['idx'])[perm].float(),
                    'label_nm': npz['label_nm'][perm],
                    # 'fn': self.imgs[index],
                    # 'metafn': metafn,
                    }
        
        if self.target_transform is not None:
            pose = self.target_transform(pose)
        
        if self.skip_images:
            return img, meta, pose
        
        if self.transform is not None:
            img = self.transform(img)
        return img, meta, pose
    
    def __len__(self):
        return len(self.poses)


if __name__ == '__main__':
    from common.vis_utils import show_batch, show_stereo_batch
    from dataset_loaders.utils import trans_bbox_cv2mat_fmt, trans_bbox_mat2cv2_fmt, trans_bbox_center2pnts, draw_bbox
    from torchvision.utils import make_grid
    import torchvision.transforms as transforms
    import cv2, matplotlib.pylab as plt
    
    mode = 3  # 0
    log = 2
    num_workers = 3
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()])

    for scene in ['full', ]:  # 'full', 'loop'
        for train in [True, False]:
            data_path = osp.join('..', 'data', 'deepslam_data', 'RobotCar')
            dset = RobotCar(scene, data_path, train=train, real=False, skip_images=True, transform=transform, log=log, mode=mode)
            print('Loaded RobotCar scene {:s}, length = {:d}'.format(scene, len(dset)))
            ## plot the poses
            # plt.figure()
            # plt.plot(dset.poses[:, 0], dset.poses[:, 1])
            # plt.show()
            data_loader = data.DataLoader(dset, batch_size=1, shuffle=False,
                                          num_workers=num_workers, collate_fn=new_collate)
            for batch in data_loader:
                if mode == 0:
                    show_batch(make_grid(batch[0], nrow=5, padding=25, normalize=True))
                elif mode == 3:
                    rgb, meta, target = batch
                    for rgb0, meta0 in zip(rgb, meta):
                        # dets = meta0['bbox']
                        # dets_test = trans_bbox_center2pnts(trans_bbox_mat2cv2_fmt(dets))
                        # img_test = to_img(rgb0.cpu().numpy())
                        # if dets_test.shape[0] != 0:
                        #     img_test = draw_bbox(img_test, dets_test, text=None)
                        # dstfn = meta0['fn'].replace('centre_processed', 'centre_bbox')
                        # dstpfn = osp.dirname(dstfn)
                        # mkdir_p(dstpfn)
                        # cv2.imwrite(dstfn, img_test[..., ::-1])
                        print(
                        meta0['bbox'].shape,
                        # meta0['fea'].shape,
                        # meta0['msk'].shape,
                        meta0['idx'].shape,
                        # meta0['fn'],
                        )
                        print(
                            meta0['bbox'],
                            meta0['idx'],
                        )
                        # plt_imshow(img_test)
                        # plt.show()
                break
