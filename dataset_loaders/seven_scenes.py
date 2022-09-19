"""
pytorch data loader for the 7-scenes dataset
"""
import os
import sys
import os.path as osp
import numpy as np
import torch
from torch.utils import data
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, './')
sys.path.insert(0, '../')
from dataset_loaders.utils import load_image, new_collate, flatten, plt_imshow, plt_imshow_tensor, to_img, resize_bbox, permutation_generator
import pickle
from common.pose_utils import process_poses


class SevenScenes(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None, target_transform=None,
                 seed=7, real=False, log=0, num_obj_cut=0,
                 skip_images=False, vo_lib='orbslam', mode=0):
        """
        :param scene: scene name ['chess', 'pumpkin', ...]
        :param data_path: root 7scenes data directory.
        Usually '../data/deepslam_data/7Scenes'
        :param train: if True, return the training images. If False, returns the
        testing images
        :param transform: transform to apply to the images
        :param target_transform: transform to apply to the poses
        :param mode: 0: just color image, 1: just depth image, 2: [c_img, d_img]
        :param real: If True, load poses from SLAM/integration of VO
        :param skip_images: If True, skip loading images and return None instead
        :param vo_lib: Library to use for VO (currently only 'dso')
        """
        self.mode = mode
        self.train = train
        self.transform = transform
        if target_transform is None:
            from torchvision import transforms as tv_transforms
            target_transform = tv_transforms.Lambda(lambda x: torch.from_numpy(x).float())
        self.target_transform = target_transform
        self.log = log
        self.num_obj_cut = num_obj_cut
        self.skip_images = skip_images
        np.random.seed(seed)
        
        # directories
        base_dir = osp.join(osp.expanduser(data_path), scene)
        data_dir = base_dir
        
        # decide which sequences to use
        if train:
            split_file = osp.join(base_dir, 'TrainSplit.txt')
        else:
            split_file = osp.join(base_dir, 'TestSplit.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]
        
        # read poses and collect image names
        self.c_imgs = []
        self.d_imgs = []
        self.gt_idx = np.empty((0,), dtype=np.int)
        ps = {}
        vo_stats = {}
        gt_offset = int(0)
        for seq in seqs:
            seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))
            seq_data_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))
            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if
                           n.find('pose') >= 0]
            assert not real, 'we do not use vo'
            frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
            pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.
                                       format(i))).flatten()[:12] for i in frame_idx]
            ps[seq] = np.asarray(pss)
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
            
            self.gt_idx = np.hstack((self.gt_idx, gt_offset + frame_idx))
            gt_offset += len(p_filenames)
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i))
                      for i in frame_idx]
            d_imgs = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i))
                      for i in frame_idx]
            self.c_imgs.extend(c_imgs)
            self.d_imgs.extend(d_imgs)
        
        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')

        if train and not real:
            mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
            std_t = np.ones(3)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)
        
        # convert pose to translation + log quaternion
        self.poses = np.empty((0, 6))
        for seq in seqs:
            pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))
        
    @staticmethod
    def imgfn2metafn(imgfn, log=0):
        lst = imgfn.split('/')
        sec = lst[-2] + '.npz'
        if log > 0:
            res = '/'.join(lst[:-2]) + '/' + str(log) + '/' + sec + '/' + lst[-1].replace('.png', '.npz')
        else:
            res = '/'.join(lst[:-2]) + '/' + sec + '/' + lst[-1].replace('.png', '.npz')
        imgpn = osp.dirname(res)
        assert osp.exists(imgpn), res
        return res
    
    @staticmethod
    def fea_reduce_spatial(fea):
        import torch.nn.functional as F
        n_inst = fea.shape[0]
        n_chnl = fea.shape[1]
        tsz = (fea.shape[0], fea.shape[1])
        if (n_inst == 0):
            return torch.zeros((0, n_chnl * 2))
        else:
            return torch.cat(
                (
                    F.adaptive_avg_pool2d(fea, (1, 1)).view(tsz),
                    F.adaptive_max_pool2d(fea, (1, 1)).view(tsz),
                ), axis=-1
            )
    
    def __getitem__(self, index):
        '''
        :param index: index of img
        :return: img, meta, pose
        mode == 0, rgb, None, pose
        mode == 1, depth, None, pose
        mode == 2, (rgb, depth), None, pose
        mode == 3, rgb, meta, pose
        '''
        img = pose = meta = None
        if self.skip_images:
            pose = self.poses[index]
        else:
            if self.mode == 0:  # rgb, None, pose
                while img is None:
                    img = load_image(self.c_imgs[index])
                    pose = self.poses[index]
                    index += 1
                index -= 1
            elif self.mode == 1:  # depth, None, pose
                while img is None:
                    img = load_image(self.d_imgs[index])
                    pose = self.poses[index]
                    index += 1
                index -= 1
            elif self.mode == 2:  # (rgb, depth), None, pose
                c_img = None
                d_img = None
                while (c_img is None) or (d_img is None):
                    c_img = load_image(self.c_imgs[index])
                    d_img = load_image(self.d_imgs[index])
                    pose = self.poses[index]
                    index += 1
                img = [c_img, d_img]
                index -= 1
            elif self.mode == 3:  # rgb, meta, pose
                while img is None:
                    img = load_image(self.c_imgs[index])
                    pose = self.poses[index]
                    index += 1
                index -= 1
                metafn = self.imgfn2metafn(self.c_imgs[index], log=self.log)
                npz = np.load(metafn)
                fea = torch.from_numpy(npz['fea'].astype('float32'))
                fea = self.fea_reduce_spatial(fea)
                bbox = npz['bbox'].astype('float32')
                # the loaded bbox is mat format, center point + size
                # row col height width
                # in a 256x256 img
                # we need to resize it to 256x341 (height = 256, width = 341)
                bbox = resize_bbox(bbox, (256,256),(256,341))
                if self.train:
                    perm = permutation_generator(bbox, self.num_obj_cut)
                else:
                    perm = torch.arange(len(bbox))
                meta = {'bbox': torch.from_numpy(bbox)[perm],
                        'msk': torch.from_numpy(npz['msk'].astype('uint8')),
                        'fea': fea[perm],
                        'idx': torch.from_numpy(npz['idx'])[perm].float(),
                        'label_nm': npz['label_nm'][perm],
                        }
            else:
                raise Exception('Wrong mode {:d}'.format(self.mode))
        
        if self.target_transform is not None:
            pose = self.target_transform(pose)
        
        if self.transform is not None and img is not None:
            if self.mode == 2:
                img = [self.transform(i) for i in img]
            else:
                img = self.transform(img)
        
        return img, meta, pose
    
    def __len__(self):
        return self.poses.shape[0]


if __name__ == '__main__':
    """
       visualizes the dataset
       # mode == 0, rgb, None, pose
       # mode == 1, depth, None, pose
       # mode == 2, (rgb, depth), None, pose
       # mode == 3, rgb, meta, pose
    """
    from common.vis_utils import show_batch, show_stereo_batch
    from torchvision.utils import make_grid
    from dataset_loaders.utils import trans_bbox_cv2mat_fmt, trans_bbox_mat2cv2_fmt, trans_bbox_center2pnts, draw_bbox
    import torchvision.transforms as transforms
    import cv2
    
    for seq in ['stairs',
               # 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs', 'chess',
                ]:
        mode = 3
        log = 0
        num_workers = 8
        transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        ## modify this path for compatible
        dset = SevenScenes(seq, '../data/deepslam_data/7Scenes', False, transform, log=log, num_obj_cut=0, mode=mode)
        # dset = SevenScenes(seq, '../data/deepslam_data/7Scenes', True, transform, log=log, mode=mode)
        print('Loaded 7Scenes sequence {:s}, length = {:d}'.format(seq,
                                                                   len(dset)))

        data_loader = data.DataLoader(dset, batch_size=3, shuffle=False,
                                      num_workers=num_workers, collate_fn=new_collate)
        sample_idx = 0
        for batch_count, batch in enumerate(data_loader):
            print('Minibatch {:d}'.format(batch_count), len(data_loader))
            if mode < 2:
                show_batch(make_grid(batch[0], nrow=1, padding=25, normalize=True))
            elif mode == 2:
                lb = make_grid(batch[0][0], nrow=1, padding=25, normalize=True)
                rb = make_grid(batch[0][1], nrow=1, padding=25, normalize=True)
                show_stereo_batch(lb, rb)
            elif mode == 3:
                rgb, meta, target = batch
                img_tests = []
                for rgb0, meta0 in zip(rgb, meta):
                    print(
                        meta0['bbox'].shape,
                        meta0['fea'].shape,
                        meta0['msk'].shape,
                        meta0['idx'].shape
                    )
                    print(
                        meta0['bbox'],
                        meta0['idx'],
                    )
            break
