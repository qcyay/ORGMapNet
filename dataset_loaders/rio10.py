"""
pytorch data loader for the 7-scenes dataset
"""
import os
import sys
import os.path as osp
import numpy as np
import pickle
import torch
from torch.utils import data
import matplotlib
import matplotlib.pyplot as plt
from torch.nn import functional as F

sys.path.insert(0, './')
sys.path.insert(0, '../')
from dataset_loaders.utils import *
from common.pose_utils import process_poses

fx, fy, cx, cy = 756.026, 756.832, 270.418, 492.889

semantic_cls = ['backpack', 'bag', 'balcony door', 'basket', 'bath cabinet', 'bathtub',
                'beanbag', 'bed', 'bench', 'blackboard', 'blanket', 'blinds', 'book', 'bottle',
                'bowl', 'box', 'cabinet', 'candles', 'chair', 'chandelier', 'clothes',
                'clothes dryer', 'clutter', 'commode', 'console', 'couch', 'couch table',
                'counter', 'cupboard', 'curtain', 'cushion', 'decoration', 'desk',
                'dining table', 'dishdrainer', 'dishes', 'dishwasher', 'door', 'doorframe',
                'drawer', 'fireplace', 'food', 'frame', 'fruits', 'garbage', 'garbage bin',
                'hand brush', 'hanger', 'heater', 'humidifier', 'ironing board', 'item',
                'kettle', 'keyboard', 'kitchen cabinet', 'ladder', 'lamp', 'light', 'microwave',
                'milk', 'mirror', 'monitor', 'nightstand', 'object', 'ottoman', 'picture',
                'pillow', 'pipe', 'plant', 'printer', 'rack', 'radiator', 'recycle bin',
                'refrigerator', 'scale', 'shades', 'shampoo', 'shelf', 'shoes', 'sidecouch',
                'sink', 'sofa', 'stair', 'stairs', 'stand', 'statue', 'stove', 'table', 'toilet',
                'toilet paper', 'towel', 'trash can', 'tube', 'tv', 'tv stand',
                'vacuum cleaner', 'vase', 'ventilator', 'wall frame', 'wardrobe',
                'washing basket', 'washing machine', 'washing powder', 'window',
                'window board', 'window frame', 'windowsill', 'wood',
                'wall', 'ceiling', 'floor', 'pipe' # these 4 cls is null_cls
                ]

cls2idx = {cls: idx for idx, cls in enumerate(semantic_cls)}


class RIO10(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None, target_transform=None,
                 seed=7, log=0, num_obj_cut=0, mode=0, with_null=False,  # default: with null_class
                 **kwargs):
        """
        :param scene: scene name [1, ... 10]
        :param data_path: root 7scenes data directory.  Usually '../data/deepslam_data/7Scenes'
        :param train: if True, return the training images. If False, returns the testing images
        :param transform: transform to apply to the images
        :param target_transform: transform to apply to the poses
        :param mode: 0: just color image, 3: color img and meta
        """
        scene = int(scene)
        self.mode = mode
        self.train = train
        self.transform = transform
        if target_transform is None:
            from torchvision import transforms as tv_transforms
            target_transform = tv_transforms.Lambda(lambda x: torch.from_numpy(x).float())
        self.target_transform = target_transform
        self.log = log
        self.num_obj_cut = num_obj_cut
        np.random.seed(seed)
        
        # directories
        base_dir = osp.join(osp.expanduser(data_path))
        img_dir = base_dir
        
        # decide which sequences to use
        if train:
            seqs = [1]
        else:
            seqs = [2]
        
        # read poses and collect image names
        self.c_imgs, self.d_imgs = [], []
        self.gt_idx = np.empty((0,), dtype=np.int)
        ps = {}
        vo_stats = {}
        gt_offset = int(0)
        for seq in seqs:
            scene_folder_nm = str(int(scene)) + '/seq{:02d}_{:02d}'.format(scene, seq)
            seq_dir = osp.join(img_dir, scene_folder_nm)
            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if
                           n.find('pose') >= 0]

            frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
            pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.
                                       format(i))).flatten()[:12] for i in frame_idx]
            ps[seq] = np.asarray(pss)
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
            
            self.gt_idx = np.hstack((self.gt_idx, gt_offset + frame_idx))
            gt_offset += len(p_filenames)
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.jpg'.format(i))
                      for i in frame_idx]
            self.c_imgs.extend(c_imgs)
            d_imgs = [img.replace('color.jpg', 'rendered.depth.png') for img in c_imgs]
            self.d_imgs.extend(d_imgs)
        
        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(base_dir, str(int(scene)), 'pose_stats.txt')
        if train and not osp.exists(pose_stats_filename):
            ## follow 7Scenes
            # mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
            # std_t = np.ones(3)
            ## follow robotcar
            mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)
            std_t = np.std(poses[:, [3, 7, 11]], axis=0)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)
        
        # convert pose to translation + log quaternion
        self.poses = np.empty((0, 6))
        self.with_null = with_null
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
            res = '/'.join(lst[:-2]) + '/' + str(log) + '/' + sec + '/' + lst[-1].replace('.png', '.npz').replace('.jpg', '.npz')
        else:
            res = '/'.join(lst[:-2]) + '/' + sec + '/' + lst[-1].replace('.png', '.npz').replace('.jpg', '.npz')
        imgpn = osp.dirname(res)
        assert osp.exists(imgpn)
        return res
    
    def __getitem__(self, index):
        '''
        :param index: index of img
        :return: img, meta, pose
        mode == 0, rgb, None, pose
        mode == 1, depth, None, pose
        mode == 2, (rgb, depth), None, pose
        mode == 3, rgb, meta, pose
        '''
        img = pose = meta = pnt = None
        mtx = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(-1, 3)
        
        while img is None:
            img = load_image(self.c_imgs[index])
            pose = self.poses[index]
            index += 1
        index -= 1
        if self.mode == 0:  # rgb, None, pose
            pass
        elif self.mode == 3:  # rgb, meta, pose
            metafn = self.imgfn2metafn(self.c_imgs[index], log=self.log)
            npz = np.load(metafn)
            bbox = npz['bbox'].astype('float32')
            # the loaded bbox is mat format, center point + size
            # row col height width
            # in a 256x256 img
            # we need to resize it to 455x256 (height = 455, width = 256)
            # 455x256 is the resized size of original 960x540
            bbox = resize_bbox(bbox, (256, 256), (455, 256))
            if self.train:
                perm = permutation_generator(bbox, self.num_obj_cut)
            else:
                perm = torch.arange(len(bbox))
            label_nm = npz['label_nm']
            idx = [cls2idx[l] for l in label_nm]
            if len(perm) == 1:
                label_nm = label_nm[perm, np.newaxis]
            else:
                label_nm = label_nm[perm]
            meta = {'bbox': torch.from_numpy(bbox)[perm],
                    'idx': torch.from_numpy(np.asarray(idx, dtype='float32'))[perm],
                    'label_nm': label_nm,
                    }
            null_label = ['wall', 'ceiling', 'floor', 'pipe']
            nonvalid_idx = np.array([lnm in null_label for lnm in meta['label_nm']])
            valid_idx = (~nonvalid_idx)
            ## to filter out null
            if not self.with_null:
                for k in meta.keys():
                    if k in ['msk', 'fn', 'metafn']: continue
                    meta[k] = meta[k][valid_idx]
        elif self.mode == 4:  # dict with keys: rgb, pnt, target
            depth_map = cv2.imread(self.d_imgs[index], -1)
            orow, ocol = depth_map.shape
            rat = 9
            nrow, ncol = int(orow / rat), int(ocol / rat)
            row_rat, col_rat = orow / nrow, ocol / ncol
            cond = (depth_map != 65535) & (depth_map != 0)
            ndepth = cv2.inpaint(depth_map, np.uint8(~cond), 3, cv2.INPAINT_TELEA)
            ndepth = cv2.resize(ndepth, dsize=(ncol, nrow),
                                interpolation=cv2.INTER_LINEAR)
            cond_all = ndepth != -1
            nu, nv = np.where(cond_all)
            nu = (nu + 0.5) * row_rat
            nv = (nv + 0.5) * col_rat
            
            pnt = np.stack((nu, nv, ndepth[cond_all]), axis=-1)
            pnt = project_img_to_rect(pnt, fx, fy, cx, cy)
        else:
            raise Exception('Wrong mode {:d}'.format(self.mode))

        pose = self.target_transform(pose)
        
        if self.transform is not None and img is not None:
            if self.mode == 2:
                img = [self.transform(i) for i in img]
            else:
                img = self.transform(img)
        if self.mode != 4:
            return img, meta, pose
        else:
            return {
                'img': img,
                'meta': meta,
                'pose': pose,
                'pnt': pnt,
            }
    
    def __len__(self):
        return self.poses.shape[0]


if __name__ == '__main__':
    """
       visualizes the dataset
       # mode == 0, rgb, None, pose
       # mode == 3, rgb, meta, pose
    """
    from common.vis_utils import show_batch, show_stereo_batch
    from dataset_loaders.utils import trans_bbox_cv2mat_fmt, trans_bbox_mat2cv2_fmt, trans_bbox_center2pnts, draw_bbox
    from torchvision.utils import make_grid
    import torchvision.transforms as transforms
    import cv2
    
    all_lbl = []
    
    for seq in range(1, 11):
        for train in [True, False]:  # False,
            name = f'{seq}_{train}'
            mode = 3
            num_workers = 0
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            dset = RIO10(seq, '../data/deepslam_data/RIO10', train, transform, num_obj_cut=2, mode=mode,
                         with_null=False)
            print('Loaded, length = {:d}'.format(len(dset)))
            
            data_loader = data.DataLoader(dset, batch_size=3, shuffle=False,
                                          num_workers=num_workers, collate_fn=new_collate)
            for batch_count, batch in enumerate(data_loader):
                print('Minibatch {:d}'.format(batch_count), len(data_loader))
                if mode == 0:
                    show_batch(make_grid(batch[0], nrow=1, padding=25, normalize=True))
                elif mode == 3:
                    rgb, meta, target = batch
                    img_tests = []
                    for rgb0, meta0 in zip(rgb, meta):
                        print(
                            meta0['bbox'].shape,
                            meta0['idx'].shape
                        )
                elif mode == 4:
                    rgb, target, pnt = batch['img'], batch['pose'], batch['pnt']
                break
