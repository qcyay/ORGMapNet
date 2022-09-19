"""
implementation of ORGPoseNet networks 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torchvision import models
import numpy as np

import os
os.environ['TORCH_HOME'] = os.path.join('..', 'data', 'models')

import sys
sys.path.insert(0, '../')

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    dis, idx = pairwise_distance.topk(k=k, dim=-1)   # (batch_size, num_points, k)
    return dis, idx

def get_graph_feature(x, k=20, idx=None):
  batch_size = x.size(0)
  num_points = x.size(2)
  k = torch.min(torch.tensor(num_points), torch.tensor(k))
  x = x.view(batch_size, -1, num_points)
  if idx is None:
    dis, idx = knn(x, k=k)  # (batch_size, num_points, k)
  device = torch.device('cuda')
  idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
  idx = idx + idx_base
  idx = idx.view(-1)

  _, num_dims, _ = x.size()
  x = x.transpose(2,
                  1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
  feature = x.view(batch_size * num_points, -1)[idx, :]
  feature = feature.view(batch_size, num_points, k, num_dims)
  x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
  feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

  return feature, dis

def get_feature(x, n_idx, k=5):

  xs = []
  dis = []
  for i in range(len(n_idx)-1):
    x0 = x[:,:,n_idx[i]:n_idx[i+1]]
    x0, dis0 = get_graph_feature(x0, k=k)
    x0 = x0.reshape(x0.size(0), x0.size(1), -1)
    dis0 = dis0.reshape(dis0.size(0), -1)
    xs.append(x0)
    dis.append(dis0)

  xs = torch.cat(xs, 2)
  dis = torch.cat(dis, 1)

  return xs, dis

def batch_aggr(x, N_idx, n_idx, aggr='max', dis=None):

  thresh = 1
  xs = []
  for i in range(len(N_idx)-1):
    n = n_idx[i+1] - n_idx[i]
    x0 = x[:,:,N_idx[i]:N_idx[i+1]]
    dis0 = dis[:,N_idx[i]:N_idx[i+1]]
    x0 = x0.reshape(x0.size(0), x0.size(1), n, -1)
    dis0 = dis0.reshape(dis0.size(0), n, -1)
    if aggr.find('weighted') >= 0:
      dis0 = dis0 + thresh
      dis0 = torch.clamp(dis0, min=-thresh*2, max=thresh)
      sim = F.softmax(dis0,-1)
      sim = sim.unsqueeze(0)
      x0 = x0 * sim
    if aggr.find('max') >= 0:
      x0 = x0.max(-1)[0]
    elif aggr.find('mean') >= 0:
      x0 = x0.mean(-1)
    elif aggr.find('sum') >= 0:
      x0 = x0.sum(-1)
    xs.append(x0)

  xs = torch.cat(xs, 2)

  return xs

def batch_pool(x, n_idx):

  xs = []
  for i in range(len(n_idx)-1):
    x0 = x[:,:,n_idx[i]:n_idx[i+1]]
    x1 = F.adaptive_max_pool1d(x0, 1).reshape(x0.size(0), -1)
    x2 = F.adaptive_avg_pool1d(x0, 1).reshape(x0.size(0), -1)
    x0 = torch.cat((x1, x2), 1)
    xs.append(x0)

  xs = torch.cat(xs, 0)

  return xs

def filter_hook(m, g_in, g_out):
  g_filtered = []
  for g in g_in:
    g = g.clone()
    g[g != g] = 0
    g_filtered.append(g)
  return tuple(g_filtered)

class GraphModel(nn.Module):
  def __init__(self, k=5, act='leakyrelu', aggr='max', device='cuda'):
    super(GraphModel, self).__init__()
    self.k = k
    self.device = device

    self.bn1 = nn.BatchNorm1d(64)
    self.bn2 = nn.BatchNorm1d(64)
    self.bn3 = nn.BatchNorm1d(128)
    self.bn4 = nn.BatchNorm1d(256)
    self.bn5 = nn.BatchNorm1d(1024)

    if act == 'relu':
      self.relu = nn.ReLU(inplace=True)
    elif act == 'leakyrelu':
      self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    self.aggr = aggr

    self.conv1 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                               self.bn1,
                               self.relu)
    self.conv2 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                               self.bn2,
                               self.relu)
    self.conv3 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),
                               self.bn3,
                               self.relu)
    self.conv4 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                               self.bn4,
                               self.relu)
    self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                               self.bn5,
                               self.relu)
    self.fc1 = nn.Linear(2048, 1024)
    self.bn6 = nn.BatchNorm1d(1024)

  def forward(self, f_obj):

    n_batch = len(f_obj)
    n_obj = np.array([len(i) for i in f_obj])
    n_idx = np.where(n_obj > 0)
    n_obj = n_obj[n_idx]
    n_obj = np.insert(n_obj, 0, 0)
    n_obj = torch.from_numpy(n_obj).int().to(self.device)
    N_obj = n_obj * n_obj.min(torch.tensor(self.k).int().to(self.device))
    n_obj = n_obj.cumsum(0)
    N_obj = N_obj.cumsum(0)
    x = torch.cat(f_obj)
    x = x.transpose(0,1).unsqueeze(0)

    x0 = torch.zeros(n_batch, 1024).to(self.device)

    if self.training and len(n_idx[0]) <= 1:

      x = x0

    elif x.size(2) == 0:

      x = x0

    else:

      x, dis = get_feature(x, n_obj, k=self.k)
      x = self.conv1(x)
      x1 = batch_aggr(x, N_obj, n_obj, aggr=self.aggr, dis=dis)

      x, dis = get_feature(x1, n_obj, k=self.k)
      x = self.conv2(x)
      x2 = batch_aggr(x, N_obj, n_obj, aggr=self.aggr, dis=dis)

      x, dis = get_feature(x2, n_obj, k=self.k)
      x = self.conv3(x)
      x3 = batch_aggr(x, N_obj, n_obj, aggr=self.aggr, dis=dis)

      x, dis = get_feature(x3, n_obj, k=self.k)
      x = self.conv4(x)
      x4 = batch_aggr(x, N_obj, n_obj, aggr=self.aggr, dis=dis)
      x = torch.cat((x1, x2, x3, x4), dim=1)

      x = self.conv5(x)
      x = batch_pool(x, n_obj)

      x = self.relu(self.bn6(self.fc1(x)))

      x0[n_idx] = x

      x = x0

    return x

class ORGPoseNet(nn.Module):
  def __init__(self, feature_extractor, droprate=0.5, pretrained=True, feat_dim=2048,
               act='leakyrelu', aggr='max', device='cuda', filter_nans=False):
    super(ORGPoseNet, self).__init__()
    self.droprate = droprate
    self.device = device

    self.graphmodel = GraphModel(act=act, aggr=aggr)

    if act == 'relu':
      self.relu = nn.ReLU(inplace=True)
    elif act == 'leakyrelu':
      self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # replace the last FC layer in feature extractor
    fe_out_planes = feature_extractor.fc.in_features
    self.feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-2])
    self.conv1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1, bias=False),
                              nn.BatchNorm2d(64),
                              self.relu)
    self.conv2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                              nn.BatchNorm2d(512),
                              self.relu)
    self.fc1_1 = nn.Sequential(nn.Linear(5, 64),
                               self.relu)
    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.maxpool = nn.AdaptiveMaxPool2d(1)
    self.fc2_1 = nn.Linear(1024, 1024)
    self.bn1 = nn.BatchNorm1d(1024)
    self.fc = nn.Linear(2048, 512)
    self.bn2 = nn.BatchNorm1d(512)

    self.fc_xyz  = nn.Linear(512, 3)
    self.fc_wpqr = nn.Linear(512, 3)

    if filter_nans:
      self.fc_wpqr.register_backward_hook(hook=filter_hook)

    # initialize
    if pretrained:
      init_modules = [*list(self.graphmodel.modules()), *list(self.conv1.modules()), *list(self.conv2.modules()), *list(self.fc1_1.modules()), self.fc2_1, self.fc, self.fc_xyz, self.fc_wpqr]
    else:
      init_modules = self.modules()

    for m in init_modules:
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)

  def forward(self, x, meta):
    x = self.feature_extractor(x)

    x0 = self.conv1(x)
    f_obj = self.obj_feature(x0, meta)
    x1 = self.graphmodel(f_obj)

    x2 = self.conv2(x)
    x3 = self.avgpool(x2)
    x4 = self.maxpool(x2)
    x2 = torch.cat((x3, x4), dim=1)
    x2 = torch.flatten(x2, 1)
    x2 = self.relu(self.bn1(self.fc2_1(x2)))

    x = torch.cat((x1, x2), dim=1)
    x = self.bn2(self.fc(x))
    x = self.relu(x)
    if self.droprate > 0:
      x = F.dropout(x, p=self.droprate, training=self.training)

    xyz = self.fc_xyz(x)
    wpqr = self.fc_wpqr(x)
    return torch.cat((xyz, wpqr), 1), x

  def bbox_to_feature(self, x, bbox):

    n_channel, h, w = x.size()
    n_obj = len(bbox)

    if n_obj > 0:
      x_obj = bbox
      x_obj = self.fc1_1(x_obj)
      f_obj = x_obj
    elif n_obj == 0:
      f_obj = torch.zeros(0, n_channel).to(self.device)

    return f_obj


  def obj_feature(self, x, meta):

    n_batch = len(meta)

    fs = []
    for i in range(n_batch):

      x0 = x[i]
      bbox = meta[i]['bbox'][:,:4]
      bbox_idx = meta[i]['idx'].unsqueeze(1)
      bbox = torch.cat((bbox, bbox_idx), 1)
      f_obj2 = self.bbox_to_feature(x0, bbox)
      f_obj = f_obj2
      fs.append(f_obj)

    return fs

if __name__ == '__main__':
  feature_extractor = models.resnet34(pretrained=True)
  model = ORGPoseNet(feature_extractor)