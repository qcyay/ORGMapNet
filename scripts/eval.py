"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
import set_paths
from common import Logger
from common.criterion import PoseNetCriterion, MapNetCriterion
from models.posenet import PoseNet, MapNet
from models.orgposenet import ORGPoseNet
from common.train import load_state_dict, step_feedfwd
from common.pose_utils import optimize_poses, quaternion_angular_error, qexp,\
  calc_vos_safe_fc, calc_vos_safe
from dataset_loaders.composite import MF
import argparse
import os
import os.path as osp
import sys
import re
import numpy as np
import matplotlib
# DISPLAY = 'DISPLAY' in os.environ
# if not DISPLAY:
#   matplotlib.use('Agg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import configparser
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pickle

from dataset_loaders.seven_scenes import new_collate

# config
parser = argparse.ArgumentParser(description='Evaluation script for PoseNet and'
                                             'MapNet variants')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'RobotCar', 'RIO10'),
                    help='Dataset')
parser.add_argument('--scene', type=str, help='Scene name')
parser.add_argument('--weights', type=str, help='trained weights to load')
parser.add_argument('--model', choices=('posenet', 'mapnet', 'mapnet++', 'orgposenet', 'orgmapnet'),
  help='Model to use (mapnet includes both MapNet and MapNet++ since their'
       'evluation process is the same and they only differ in the input weights'
       'file')
parser.add_argument('--device', type=str, default='0', help='GPU device(s)')
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--val', action='store_true', help='Plot graph for val')
parser.add_argument('--save_feat', action='store_true', help='Save intermediate features')
parser.add_argument('--output_dir', type=str, default=None,
  help='Output image directory')
parser.add_argument('--pose_graph', action='store_true',
  help='Turn on Pose Graph Optimization')
args = parser.parse_args()
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
  os.environ['CUDA_VISIBLE_DEVICES'] = args.device

settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
  settings.read_file(f)
seed = settings.getint('training', 'seed')
section = settings['dataloader']
npz_log = section.getint('npz_log')
num_obj_cut = section.getint('num_obj_cut')
section = settings['network']
act = section['act']
aggr = section['aggr']
section = settings['hyperparameters']
dropout = section.getfloat('dropout')
if (args.model.find('mapnet') >= 0) or args.pose_graph:
  steps = section.getint('steps')
  skip = section.getint('skip')
  real = section.getboolean('real')
  variable_skip = section.getboolean('variable_skip')
  fc_vos = args.dataset == 'RobotCar'
  if args.pose_graph:
    vo_lib = section.get('vo_lib')
    sax = section.getfloat('s_abs_trans', 1)
    saq = section.getfloat('s_abs_rot', 1)
    srx = section.getfloat('s_rel_trans', 20)
    srq = section.getfloat('s_rel_rot', 20)

if args.dataset == 'RobotCar':
  base_scene = re.sub(r'\d', '', args.scene)
else:
  base_scene = args.scene

pattern = '\d+'
epoch = re.findall(pattern, args.weights.split('/')[-1])[0]

# model
feature_extractor = models.resnet34(pretrained=False)
if args.model.find('org') >= 0:
  posenet = ORGPoseNet(feature_extractor, droprate=dropout, pretrained=False, act=act, aggr=aggr)
else:
  posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=False)
if (args.model.find('mapnet') >= 0) or args.pose_graph:
  model = MapNet(mapnet=posenet)
else:
  model = posenet
model.eval()

# loss functions
if args.model.find('posenet') >= 0:
  train_criterion = PoseNetCriterion(learn_beta=True)
elif args.model.find('mapnet') >= 0:
  kwargs = dict(learn_beta=True, learn_gamma=True)
  train_criterion = MapNetCriterion(**kwargs)
else:
  raise NotImplementedError

t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
q_criterion = quaternion_angular_error

# load weights
# os.path.expanduser把path中包含的"~"和"~user"转换成用户目录
weights_filename = osp.expanduser(args.weights)
if osp.isfile(weights_filename):
  loc_func = lambda storage, loc: storage
  checkpoint = torch.load(weights_filename, map_location=loc_func)
  load_state_dict(model, checkpoint['model_state_dict'])
  c_state = checkpoint['criterion_state_dict']
  train_criterion.load_state_dict(c_state)
  print('Loaded weights from {:s}'.format(weights_filename))
else:
  print('Could not load weights from {:s}'.format(weights_filename))
  sys.exit(-1)

data_dir = osp.join('..', 'data', args.dataset)
stats_filename = osp.join(data_dir, base_scene, 'stats.txt')
stats = np.loadtxt(stats_filename)
# transformer
data_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.ToTensor(),
  transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# read mean and stdev for un-normalizing predictions
pose_stats_file = osp.join(data_dir, base_scene, 'pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

# dataset
train = not args.val
if train:
  print('Running {:s} on TRAIN data'.format(args.model))
else:
  print('Running {:s} on VAL data'.format(args.model))
if args.model.find('org') >= 0:
  mode = 3
else:
  mode = 0
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, train=train, transform=data_transform,
              target_transform=target_transform, seed=seed, log=npz_log, num_obj_cut=num_obj_cut, mode=mode)
if (args.model.find('mapnet') >= 0) or args.pose_graph:
  if args.pose_graph:
    assert real
    kwargs = dict(kwargs, vo_lib=vo_lib)
  vo_func = calc_vos_safe_fc if fc_vos else calc_vos_safe
  data_set = MF(dataset=args.dataset, steps=steps, skip=skip, real=real,
                variable_skip=variable_skip, include_vos=args.pose_graph,
                vo_func=vo_func, no_duplicates=False, **kwargs)
  L = len(data_set.dset)
elif args.dataset == '7Scenes':
  from dataset_loaders.seven_scenes import SevenScenes
  data_set = SevenScenes(**kwargs)
  L = len(data_set)
elif args.dataset == 'RobotCar':
  from dataset_loaders.robotcar import RobotCar
  data_set = RobotCar(**kwargs)
  L = len(data_set)
elif args.dataset == 'RIO10':
  from dataset_loaders.rio10 import RIO10
  data_set = RIO10(**kwargs)
  L = len(data_set)
else:
  raise NotImplementedError

# loader (batch_size MUST be 1)
batch_size = 1
assert batch_size == 1
loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,
                    num_workers=5, pin_memory=True, collate_fn=new_collate)

# activate GPUs
CUDA = torch.cuda.is_available()
torch.manual_seed(seed)
if CUDA:
  torch.cuda.manual_seed(seed)
  model.cuda()

if args.model.find('posenet') >= 0:
  T = 1
elif (args.model.find('mapnet') >= 0):
  T = steps

if args.model.find('org') >= 0:
    C = 512
else:
    C = 2048

num_bboxs = np.zeros(L)

pred_poses = np.zeros((L, T, 7))  # store all predicted poses
targ_poses = np.zeros((L, T, 7))  # store all target poses

feats = np.zeros((L, C))

# inference loop
for batch_idx, (data, meta, target) in enumerate(loader):
  # if batch_idx < 297:
  #   continue
  # if batch_idx == 300:
  #   break
  if batch_idx % 200 == 0:
    print('Image {:d} / {:d}'.format(batch_idx, len(loader)))

  # indices into the global arrays storing poses
  if (args.model.find('vid') >= 0) or args.pose_graph:
    idx = data_set.get_indices(batch_idx)
  else:
    idx = [batch_idx]
  idx = idx[len(idx) // 2]

  # output : 1 x 6 or 1 x STEPS x 6
  _, _, output = step_feedfwd(data, meta, model, CUDA, train=False)
  if isinstance(output, tuple):
    #尺寸为[1,T,C]
    feat = output[1]
    s = feat.size()
    #尺寸为[T,C]
    feat = feat.cpu().data.numpy().reshape((-1, s[-1]))
    #尺寸为[C]
    feat = feat[T//2]
    #尺寸为[1,T,6]
    output = output[0]
  s = output.size()
  #尺寸为[T,6]
  output = output.cpu().data.numpy().reshape((-1, s[-1]))
  #尺寸为[T,6]
  target = target.numpy().reshape((-1, s[-1]))
  
  # normalize the predicted quaternions
  #列表，包含T个数组，尺寸为[4]
  q = [qexp(p[3:]) for p in output]
  #尺寸为[T,7]
  output = np.hstack((output[:, :3], np.asarray(q)))
  #列表，包含T个数组，尺寸为[4]
  q = [qexp(p[3:]) for p in target]
  #尺寸为[T,7]
  target = np.hstack((target[:, :3], np.asarray(q)))

  # print(output)
  # print(target)

  if args.pose_graph:  # do pose graph optimization
    kwargs = {'sax': sax, 'saq': saq, 'srx': srx, 'srq': srq}
    # target includes both absolute poses and vos
    vos = target[len(output):]
    target = target[:len(output)]
    output = optimize_poses(pred_poses=output, vos=vos, fc_vos=fc_vos, **kwargs)

  # un-normalize the predicted and target translations
  output[:, :3] = output[:, :3] * pose_s
  target[:, :3] = target[:, :3] * pose_s

  pred_poses[idx, :] = output
  targ_poses[idx, :] = target

  if args.save_feat:
    feats[idx, :] = feat

  if args.model == 'orgposenet':
    num_bboxs[idx] = len(meta[0]['bbox'])
  elif args.model == 'orgmapnet':
    num_bboxs[idx] = len(meta[0][len(output) // 2]['bbox'])

# pred_t_poses = pred_poses[:, :3]
# targ_t_poses = targ_poses[:, :3]
# print(pred_t_poses.mean(0))
# print(pred_t_poses.std(0))
# print(targ_t_poses.mean(0))
# print(targ_t_poses.std(0))

# calculate losses
t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, T//2, :3],
                                                       targ_poses[:, T//2, :3])])
q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, T//2, 3:],
                                                       targ_poses[:, T//2, 3:])])
#eval_func = np.mean if args.dataset == 'RobotCar' else np.median
#eval_str  = 'Mean' if args.dataset == 'RobotCar' else 'Median'
#t_loss = eval_func(t_loss)
#q_loss = eval_func(q_loss)
#print '{:s} error in translation = {:3.2f} m\n' \
#      '{:s} error in rotation    = {:3.2f} degrees'.format(eval_str, t_loss,

result = ''.join(f'{k}: {v[0]:.2f} ' for k,v in train_criterion.named_parameters())
print(''.join(f'{k}: {v[0]:.2f} ' for k,v in train_criterion.named_parameters()))

keys = [k for k,_ in train_criterion.named_parameters()]
values = [np.around(v[0].cpu().data.numpy(),2) for _,v in train_criterion.named_parameters()]
params = dict(zip(keys,values))

print('Error in translation: median {:3.2f} m,  mean {:3.2f} m\n' \
    'Error in rotation: median {:3.2f} degrees, mean {:3.2f} degree'.format(np.median(t_loss), np.mean(t_loss),
                    np.median(q_loss), np.mean(q_loss)))

# create figure object
fig = plt.figure()
if args.dataset == 'RobotCar':
  ax = fig.add_subplot(111)
else:
  ax = fig.add_subplot(111, projection='3d')
#subplots_adjust调整子图布局参数
# plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
if args.dataset != 'RobotCar':
  ax.set_zlabel('z (m)')
  ax.set_title(f'{result}\n'
               f'Error in translation: median {np.median(t_loss):.2f} m, mean {np.mean(t_loss):.2f} m\n'
               f'Error in rotation: median {np.median(q_loss):.2f} degrees, mean {np.mean(q_loss):.2f} degree', fontsize=8, y=0.95)

# plot on the figure object
ss = max(1, int(len(data_set) // 1000))  # 100 for stairs
# scatter the points and draw connecting line
#尺寸为[2,n]
x = np.vstack((pred_poses[::ss, T//2, 0].T, targ_poses[::ss, T//2, 0].T))
#尺寸为[2,n]
y = np.vstack((pred_poses[::ss, T//2, 1].T, targ_poses[::ss, T//2, 1].T))
if args.dataset == 'RobotCar':  # 2D drawing
  ax.plot(x[0, :], y[0, :], c='r', marker='o', markersize=1, linewidth=1)
  ax.plot(x[1, :], y[1, :], c='k', linewidth=1)
  # ax.plot(x, y, c='b')
  # ax.scatter(x[0, :], y[0, :], c='r')
  # ax.scatter(x[1, :], y[1, :], c='g')
else:
  #尺寸为[2,n]
  z = np.vstack((pred_poses[::ss, T//2, 2].T, targ_poses[::ss, T//2, 2].T))
  for xx, yy, zz in zip(x.T, y.T, z.T):
    ax.plot(xx, yy, zs=zz, c='b')
  ax.scatter(x[0, :], y[0, :], zs=z[0, :], c='r', depthshade=0)
  ax.scatter(x[1, :], y[1, :], zs=z[1, :], c='g', depthshade=0)
  #view_init设置轴的仰角和方位角
  ax.view_init(azim=119, elev=13)

# if DISPLAY:
#   plt.show(block=True)

if args.output_dir is not None:
  model_name = args.model
  if args.weights.find('++') >= 0:
    model_name += '++'
  if args.pose_graph:
    model_name += '_pgo_{:s}'.format(vo_lib)
  if train:
    experiment_name = '{:s}_{:s}_{:s}_{:s}_train'.format(args.dataset, args.scene, model_name, epoch)
  else:
    experiment_name = '{:s}_{:s}_{:s}_{:s}_val'.format(args.dataset, args.scene, model_name, epoch)
  image_filename = osp.join(osp.expanduser(args.output_dir),
    '{:s}.png'.format(experiment_name))
  fig.savefig(image_filename)
  print('{:s} saved'.format(image_filename))
  result_filename = osp.join(osp.expanduser(args.output_dir), '{:s}.pkl'.
    format(experiment_name))
  result = {'targ_poses': targ_poses,
            'pred_poses': pred_poses,
            'epoch': int(epoch),
            't_loss': t_loss,
            'q_loss': q_loss,
            'mean_t_loss': np.around(np.mean(t_loss),2),
            'mean_q_loss': np.around(np.mean(q_loss),2),
            **params}
  if args.model.find('org') >= 0:
    result['num_bboxs'] = num_bboxs
  if args.save_feat:
    result['feats'] = feats
  with open(result_filename, 'wb') as f:
    pickle.dump(result, f)
  print('{:s} written'.format(result_filename))
