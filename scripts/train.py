import set_paths
from common.train import Trainer
from common.optimizer import Optimizer
from common.criterion import PoseNetCriterion, MapNetCriterion,\
  MapNetOnlineCriterion
from models.posenet import PoseNet, MapNet
from models.orgposenet import ORGPoseNet
from dataset_loaders.composite import MF, MFOnline
import os.path as osp
import numpy as np
import argparse
import configparser
import json
import torch
from torch import nn
from torchvision import transforms, models

parser = argparse.ArgumentParser(description='Training script for ORGPoseNet and'
                                             'ORGMapNet variants')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'RobotCar', 'RIO10'),
                    help='Dataset')
parser.add_argument('--scene', type=str, help='Scene name')
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--model', choices=('posenet', 'mapnet', 'mapnet++', 'orgposenet', 'orgmapnet'),
  help='Model to train')
parser.add_argument('--device', type=str, default='0',
  help='value to be set to $CUDA_VISIBLE_DEVICES')
parser.add_argument('--checkpoint', type=str, help='Checkpoint to resume from',
  default=None)
parser.add_argument('--learn_beta', action='store_true',
  help='Learn the weight of translation loss')
parser.add_argument('--learn_gamma', action='store_true',
  help='Learn the weight of rotation loss')
parser.add_argument('--resume_optim', action='store_true',
  help='Resume optimization (only effective if a checkpoint is given')
parser.add_argument('--log', type=str, default='0')
parser.add_argument('--suffix', type=str, default='',
                    help='Experiment name suffix (as is)')
args = parser.parse_args()

settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
  settings.read_file(f)
section = settings['dataloader']
npz_log = section.getint('npz_log')
num_obj_cut = section.getint('num_obj_cut')
section = settings['optimization']
#json.loads用于解码JSON数据
optim_config = {k: json.loads(v) for k,v in section.items() if k != 'opt' and k != 'dif'}
opt_method = section['opt']
#pop方法删除字典给定键key及对应的值，返回值为被删除的值
lr = optim_config.pop('lr')
weight_decay = optim_config.pop('weight_decay')
dif = section.getboolean('dif')

section = settings['network']
act = section['act']
aggr = section['aggr']

section = settings['hyperparameters']
#getfloat(section,option)将在section中指定的option转换为浮点数的方法
dropout = section.getfloat('dropout')
color_jitter = section.getfloat('color_jitter', 0)
brightness = section.getfloat('brightness', 0)
contrast = section.getfloat('contrast', 0)
saturation = section.getfloat('saturation', 0)
hue = section.getfloat('hue', 0)
sax = section.getfloat('beta_t')
saq = section.getfloat('beta_q')
q_loss_weight = section.getfloat('q_loss_weight')
#find方法检测字符串中是否包含子字符串str
if args.model.find('mapnet') >= 0:
  #getint(section,option)将在section中指定的option转换为整数的方法
  skip = section.getint('skip')
  #getboolean(section,option)将在指定的section中的option转换为布尔值的方法
  real = section.getboolean('real')
  variable_skip = section.getboolean('variable_skip')
  srx = section.getfloat('gamma_t')
  srq = section.getfloat('gamma_q')
  vo_q_loss_weight = section.getfloat('vo_q_loss_weight')
  steps = section.getint('steps')
if args.model.find('++') >= 0:
  vo_lib = section.get('vo_lib', 'orbslam')
  print('Using {:s} VO'.format(vo_lib))

section = settings['training']
seed = section.getint('seed')

# model
feature_extractor = models.resnet34(pretrained=True)
if args.model.find('org') >= 0:
  posenet = ORGPoseNet(feature_extractor, droprate=dropout, pretrained=True, act=act,
                       aggr=aggr, filter_nans=(args.model=='mapnet++'))
else:
  posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=True,
                    filter_nans=(args.model=='mapnet++'))
if args.model.find('posenet') >= 0:
  model = posenet
elif args.model.find('mapnet') >= 0:
  model = MapNet(mapnet=posenet)
else:
  raise NotImplementedError

# loss function
if args.model.find('posenet') >= 0:
  train_criterion = PoseNetCriterion(sax=sax, saq=saq, q_loss_weight=q_loss_weight, learn_beta=args.learn_beta)
  val_criterion = PoseNetCriterion()
elif args.model.find('mapnet') >= 0:
  kwargs = dict(sax=sax, saq=saq, srx=srx, srq=srq, q_loss_weight=q_loss_weight,
                vo_q_loss_weight=vo_q_loss_weight, learn_beta=args.learn_beta,
                learn_gamma=args.learn_gamma)
  if args.model.find('++') >= 0:
    kwargs = dict(kwargs, gps_mode=(vo_lib=='gps') )
    train_criterion = MapNetOnlineCriterion(**kwargs)
    val_criterion = MapNetOnlineCriterion()
  else:
    train_criterion = MapNetCriterion(**kwargs)
    val_criterion = MapNetCriterion()
else:
  raise NotImplementedError

# optimizer
if dif:
  if args.model.find('posenet') >= 0:
    #id函数返回对象的唯一标识符，标识符是一个整数
    #map会根据提供的函数对指定序列做映射
    fe_params = list(map(id, model.feature_extractor.parameters()))
    #filter函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表
    rest_params = filter(lambda p: id(p) not in fe_params, model.parameters())
    param_list = [{'params': model.feature_extractor.parameters(), 'lr': 1e-4},
                  {'params': rest_params}]
  elif args.model.find('mapnet') >= 0:
    fe_params = list(map(id, model.mapnet.feature_extractor.parameters()))
    rest_params = filter(lambda p: id(p) not in fe_params, model.parameters())
    param_list = [{'params': model.mapnet.feature_extractor.parameters(), 'lr': 1e-4},
                  {'params': rest_params}]
else:
  param_list = [{'params': model.parameters()}]
#hasattr用于判断对象是否包含对应的属性
if args.learn_beta and hasattr(train_criterion, 'sax') and \
    hasattr(train_criterion, 'saq'):
  param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
if args.learn_gamma and hasattr(train_criterion, 'srx') and \
    hasattr(train_criterion, 'srq'):
  param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
optimizer = Optimizer(params=param_list, method=opt_method, base_lr=lr,
  weight_decay=weight_decay, **optim_config)

data_dir = osp.join('..', 'data', args.dataset)
stats_file = osp.join(data_dir, args.scene, 'stats.txt')
stats = np.loadtxt(stats_file)
crop_size_file = osp.join(data_dir, 'crop_size.txt')
crop_size = tuple(np.loadtxt(crop_size_file).astype(np.int))
# transformers
tforms = [transforms.Resize(256)]
if color_jitter > 0 or brightness > 0 or contrast > 0 or saturation > 0 or hue > 0:
  assert color_jitter <= 1.0
  print('Using ColorJitter data augmentation')
  t_brightness, t_contrast, t_saturation, t_hue = color_jitter, color_jitter, color_jitter, 0
  if brightness > 0:
    t_brightness = brightness
  if contrast > 0:
    t_contrast = contrast
  if saturation > 0:
    t_saturation = saturation
  if hue > 0:
    t_hue = hue
  #ColorJitter随机改变图像的亮度、对比度和饱和度
  tforms.append(transforms.ColorJitter(brightness=t_brightness,
    contrast=t_contrast, saturation=t_saturation, hue=t_hue))
tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
data_transform = transforms.Compose(tforms)
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# datasets
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
if args.model.find('org') >= 0:
  mode = 3
else:
  mode = 0
kwargs = dict(scene=args.scene, data_path=data_dir, transform=data_transform,
  target_transform=target_transform, seed=seed, log=npz_log, num_obj_cut=num_obj_cut, mode=mode)
if args.model.find('posenet') >= 0:
  if args.dataset == '7Scenes':
    from dataset_loaders.seven_scenes import SevenScenes
    train_set = SevenScenes(train=True, **kwargs)
    val_set = SevenScenes(train=False, **kwargs)
  elif args.dataset == 'RobotCar':
    from dataset_loaders.robotcar import RobotCar
    train_set = RobotCar(train=True, **kwargs)
    val_set = RobotCar(train=False, **kwargs)
  elif args.dataset == 'RIO10':
    from dataset_loaders.rio10 import RIO10
    train_set = RIO10(train=True, **kwargs)
    val_set = RIO10(train=False, **kwargs)
  else:
    raise NotImplementedError
elif args.model.find('mapnet') >= 0:
  kwargs = dict(kwargs, dataset=args.dataset, skip=skip, steps=steps,
    variable_skip=variable_skip)
  if args.model.find('++') >= 0:
    train_set = MFOnline(vo_lib=vo_lib, gps_mode=(vo_lib=='gps'), **kwargs)
    val_set = None
  else:
    train_set = MF(train=True, real=real, **kwargs)
    val_set = MF(train=False, real=real, **kwargs)
else:
  raise NotImplementedError

# trainer
config_name = args.config_file.split('/')[-1]
config_name = config_name.split('.')[0]
if args.model.find('posenet') >= 0:
  model_name = 'posenet'
elif args.model.find('mapnet') >= 0:
  model_name = 'mapnet'
experiment_name = '{:s}/{:s}/{:s}/{:s}_{:s}_{:s}_{:s}'.format(model_name, args.dataset, args.log, args.dataset, args.scene,
  args.model, config_name)
if args.learn_beta:
  experiment_name = '{:s}_learn_beta'.format(experiment_name)
if args.learn_gamma:
  experiment_name = '{:s}_learn_gamma'.format(experiment_name)
experiment_name += args.suffix
trainer = Trainer(model, optimizer, train_criterion, args.config_file,
                  experiment_name, train_set, val_set, device=args.device,
                  checkpoint_file=args.checkpoint,
                  resume_optim=args.resume_optim, val_criterion=val_criterion)
lstm = args.model == 'vidloc'
trainer.train_val(lstm=lstm)
