"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
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

#log6:利用物体框的位置信息和尺寸信息以及物体框的类别得到物体的特征输入图网络

#def trace_hook(m, g_in, g_out):
#  for idx,g in enumerate(g_in):
#    g = g.cpu().data.numpy()
#    if np.isnan(g).any():
#      set_trace()
#  return None

#k近邻函数
#输入x，尺寸为[B,C,n]，k
#输出dis,每个点和距离最近的k个点之间的距离，尺寸为[B,n,k]，idx，每个点距离最近的k个点的序号，尺寸为[B,n,k]
def knn(x, k):
    #torch.matmul计算两个张量的矩阵乘积
    #尺寸为[B,n,n]
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    #尺寸为[B,1,n]
    xx = torch.sum(x**2, dim=1, keepdim=True)
    #pairwise_distance，两个点之间距离的负值，尺寸为[B,n,n]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    #torch.tensor.topk返回给定输入张量沿给定维度的k个最大元素
    #dis，每个点和距离最近的k个点之间的距离，尺寸为[B,n,k]，idx，每个点距离最近的k个点的序号，尺寸为[B,n,k]
    dis, idx = pairwise_distance.topk(k=k, dim=-1)   # (batch_size, num_points, k)
    return dis, idx

# 点特征和点的k近邻点特征获取函数
# 输入，x，点的特征，尺寸为[B,C,n]，k，最近邻点的数量，idx，每个点距离最近的k个点的序号，尺寸为[B,n,k]
# 输出feature，点的特征和点的k近邻点的特征，尺寸为[B,2C,n,k]，dis，每个点和距离最近的k个点之间的距离，尺寸为[B,n,k]
def get_graph_feature(x, k=20, idx=None):
  batch_size = x.size(0)
  num_points = x.size(2)
  k = torch.min(torch.tensor(num_points), torch.tensor(k))
  # 尺寸为[B,C,n]
  x = x.view(batch_size, -1, num_points)
  if idx is None:
    #dis，尺寸为[B,n,k]，idx，尺寸为[B,n,k]
    dis, idx = knn(x, k=k)  # (batch_size, num_points, k)
  device = torch.device('cuda')
  # 尺寸为[B,1,1]
  idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
  # 尺寸为[B,n,k]
  idx = idx + idx_base
  # 尺寸为[Bnk]
  idx = idx.view(-1)

  _, num_dims, _ = x.size()
  # 尺寸为[B,n,C]
  x = x.transpose(2,
                  1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
  # feature，每个点距离最近的k个点的特征，尺寸为[Bnk,C]
  feature = x.view(batch_size * num_points, -1)[idx, :]
  # 尺寸为[B,n,k,C]
  feature = feature.view(batch_size, num_points, k, num_dims)
  # 尺寸为[B,n,k,C]
  x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
  # 尺寸为[B,2C,n,k]
  feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

  return feature, dis

#特征获取函数
#输入x，图像中物体的特征，尺寸为[1,C,N]，n_idx，尺寸为[B+1]，k，最近邻点的数量
#输出xs，物体的特征和每个物体距离最近的k个物体的特征，尺寸为[1,2C,Nk]，dis，每个点和距离最近的k个点之间的距离，尺寸为[1,Nk]
def get_feature(x, n_idx, k=5):

  xs = []
  dis = []
  for i in range(len(n_idx)-1):
    #尺寸为[1,C,n]
    x0 = x[:,:,n_idx[i]:n_idx[i+1]]
    #x0，尺寸为[1,2C,n,k]，dis0，尺寸为[1,n,k]
    x0, dis0 = get_graph_feature(x0, k=k)
    #尺寸为[1,2C,nk]
    x0 = x0.reshape(x0.size(0), x0.size(1), -1)
    #尺寸为[1,nk]
    dis0 = dis0.reshape(dis0.size(0), -1)
    #列表，包含B个tensor，tensor尺寸为[1,2C,nk]
    xs.append(x0)
    #列表，包含B个tensor，tensor尺寸为[1,nk]
    dis.append(dis0)

  #尺寸为[1,2C,Nk]
  xs = torch.cat(xs, 2)
  #尺寸为[1,Nk]
  dis = torch.cat(dis, 1)

  return xs, dis

#输入x，物体的特征和每个物体距离最近的k个物体的特征，尺寸为[1,C,Nk]，N_idx，尺寸为[B+1]，n_idx，尺寸为[B+1]
#输出xs，图像中物体的特征，尺寸为[1,C,N]
def batch_aggr(x, N_idx, n_idx, aggr='max', dis=None):

  thresh = 1
  xs = []
  for i in range(len(N_idx)-1):
    n = n_idx[i+1] - n_idx[i]
    #尺寸为[1,C,nk]
    x0 = x[:,:,N_idx[i]:N_idx[i+1]]
    #尺寸为[1,nk]
    dis0 = dis[:,N_idx[i]:N_idx[i+1]]
    #尺寸为[1,C,n,k]
    x0 = x0.reshape(x0.size(0), x0.size(1), n, -1)
    #尺寸为[1,n,k]
    dis0 = dis0.reshape(dis0.size(0), n, -1)
    if aggr.find('weighted') >= 0:
      #尺寸为[1,n,k]
      dis0 = dis0 + thresh
      #尺寸为[1,n,k]
      dis0 = torch.clamp(dis0, min=-thresh*2, max=thresh)
      #尺寸为[1,n,k]
      sim = F.softmax(dis0,-1)
      #尺寸为[1,1,n,k]
      sim = sim.unsqueeze(0)
      #尺寸为[1,C,n,k]
      x0 = x0 * sim
    if aggr.find('max') >= 0:
      #尺寸为[1,C,n]
      x0 = x0.max(-1)[0]
    elif aggr.find('mean') >= 0:
      #尺寸为[1,C,n]
      x0 = x0.mean(-1)
    elif aggr.find('sum') >= 0:
      #尺寸为[1,C,n]
      x0 = x0.sum(-1)
    #列表，包含B个tensor，tensor尺寸为[1,C,n]
    xs.append(x0)

  #尺寸为[1,C,N]
  xs = torch.cat(xs, 2)

  return xs

#输入x，图像中物体的特征，尺寸为[1,C,N]，n_idx，尺寸为[B+1]
#输出xs，图像根据物体表示的特征，尺寸为[B,2C]
def batch_pool(x, n_idx):

  xs = []
  for i in range(len(n_idx)-1):
    #尺寸为[1,C,n]
    x0 = x[:,:,n_idx[i]:n_idx[i+1]]
    #尺寸为[1,C]
    x1 = F.adaptive_max_pool1d(x0, 1).reshape(x0.size(0), -1)
    #尺寸为[1,C]
    x2 = F.adaptive_avg_pool1d(x0, 1).reshape(x0.size(0), -1)
    #尺寸为[1,2C]
    x0 = torch.cat((x1, x2), 1)
    #列表，包含B个tensor，tensor尺寸为[1,2C]
    xs.append(x0)

  #尺寸为[B,2C]
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
    #尺寸为[B]
    n_obj = np.array([len(i) for i in f_obj])
    #尺寸为[b]
    n_idx = np.where(n_obj > 0)
    #尺寸为[b]
    n_obj = n_obj[n_idx]
    #尺寸为[b+1]
    n_obj = np.insert(n_obj, 0, 0)
    #尺寸为[b+1]
    n_obj = torch.from_numpy(n_obj).int().to(self.device)
    #尺寸为[b+1]
    N_obj = n_obj * n_obj.min(torch.tensor(self.k).int().to(self.device))
    #尺寸为[b+1]
    n_obj = n_obj.cumsum(0)
    #尺寸为[b+1]
    N_obj = N_obj.cumsum(0)
    #尺寸为[N,64]
    x = torch.cat(f_obj)
    #尺寸为[1,64,N]
    x = x.transpose(0,1).unsqueeze(0)

    #尺寸为[B,1024]
    x0 = torch.zeros(n_batch, 1024).to(self.device)

    if self.training and len(n_idx[0]) <= 1:

      #尺寸为[B,1024]
      x = x0

    elif x.size(2) == 0:

      #尺寸为[B,1024]
      x = x0

    else:

      #x，尺寸为[1,128,Nk]，dis，尺寸为[1,Nk]
      x, dis = get_feature(x, n_obj, k=self.k)
      #尺寸为[1,64,Nk]
      x = self.conv1(x)
      #尺寸为[1,64,N]
      x1 = batch_aggr(x, N_obj, n_obj, aggr=self.aggr, dis=dis)

      #x，尺寸为[1,128,Nk]，dis，尺寸为[1,Nk]
      x, dis = get_feature(x1, n_obj, k=self.k)
      #尺寸为[1,64,Nk]
      x = self.conv2(x)
      #尺寸为[1,64,N]
      x2 = batch_aggr(x, N_obj, n_obj, aggr=self.aggr, dis=dis)

      #x，尺寸为[1,128,Nk]，dis，尺寸为[1,Nk]
      x, dis = get_feature(x2, n_obj, k=self.k)
      #尺寸为[1,128,Nk]
      x = self.conv3(x)
      #尺寸为[1,128,N]
      x3 = batch_aggr(x, N_obj, n_obj, aggr=self.aggr, dis=dis)

      #尺寸为[1,256,Nk]，dis，尺寸为[1,Nk]
      x, dis = get_feature(x3, n_obj, k=self.k)
      #尺寸为[1,256,Nk]
      x = self.conv4(x)
      #尺寸为[1,256,N]
      x4 = batch_aggr(x, N_obj, n_obj, aggr=self.aggr, dis=dis)
      #尺寸为[1,512,N]
      x = torch.cat((x1, x2, x3, x4), dim=1)

      #尺寸为[1,1024,N]
      x = self.conv5(x)
      #尺寸为[b,2048]
      x = batch_pool(x, n_obj)

      #尺寸为[b,1024]
      x = self.relu(self.bn6(self.fc1(x)))

      #尺寸为[B,1024]
      x0[n_idx] = x

      #尺寸为[B,1024]
      x = x0

    return x

#输入x，图像，尺寸为[B,3,H,W]，meta，列表，包含B个字典
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
      #register_backward_hook在module上注册一个backward hook
      #hook下面应该有下面的signature
      #hook(module, grad_input, grad_output)
      #如果module有多个输入输出的话，那么grad_input和grad_output将会是个tuple
      #hook可以选择性返回关于输入的梯度，返回的梯度在后续的计算中会替代grad_input
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
    #尺寸为[B,512,h,w]
    x = self.feature_extractor(x)

    #尺寸为[B,64,h,w]
    x0 = self.conv1(x)
    #列表，包含B个tensor，tensor尺寸为[N,64]
    f_obj = self.obj_feature(x0, meta)
    #尺寸为[B,1024]
    x1 = self.graphmodel(f_obj)

    #尺寸为[B,512,h,w]
    x2 = self.conv2(x)
    #尺寸为[B,512,1,1]
    x3 = self.avgpool(x2)
    #尺寸为[B,512,1,1]
    x4 = self.maxpool(x2)
    #尺寸为[B,1024,1,1]
    x2 = torch.cat((x3, x4), dim=1)
    #尺寸为[B,1024]
    x2 = torch.flatten(x2, 1)
    #尺寸为[B,1024]
    x2 = self.relu(self.bn1(self.fc2_1(x2)))

    #尺寸为[B,2048]
    x = torch.cat((x1, x2), dim=1)
    #尺寸为[B,512]
    x = self.bn2(self.fc(x))
    x = self.relu(x)
    if self.droprate > 0:
      x = F.dropout(x, p=self.droprate, training=self.training)

    #尺寸为[B,3]
    xyz = self.fc_xyz(x)
    #尺寸为[B,3]
    wpqr = self.fc_wpqr(x)
    return torch.cat((xyz, wpqr), 1), x

  #输入x，特征图，尺寸为[C,h,w],bbox，尺寸为[n,5]
  #输出f_obj，图像中表示物体框的特征，尺寸为[n,64]
  def bbox_to_feature(self, x, bbox):

    n_channel, h, w = x.size()
    n_obj = len(bbox)

    if n_obj > 0:
      #尺寸为[n,5]
      x_obj = bbox
      #尺寸为[n,64]
      x_obj = self.fc1_1(x_obj)
      #尺寸为[n,64]
      f_obj = x_obj
    elif n_obj == 0:
      f_obj = torch.zeros(0, n_channel).to(self.device)

    return f_obj


  #物体特征获取函数
  #输入x，特征图，尺寸为[B,C,h,w]，meta，列表，包含B个字典
  #输出fs，图像中物体的特征，列表，包含B个tensor，tensor尺寸为[n,C]
  def obj_feature(self, x, meta):

    n_batch = len(meta)

    fs = []
    for i in range(n_batch):

      #尺寸为[C,h,w]
      x0 = x[i]
      #尺寸为[n,4]
      bbox = meta[i]['bbox'][:,:4]
      #尺寸为[n,1]
      bbox_idx = meta[i]['idx'].unsqueeze(1)
      #尺寸为[n,5]
      bbox = torch.cat((bbox, bbox_idx), 1)
      #尺寸为[n,C]
      f_obj2 = self.bbox_to_feature(x0, bbox)
      #尺寸为[n,C]
      f_obj = f_obj2
      #列表，包含B个tensor，tensor尺寸为[n,C]
      fs.append(f_obj)

    return fs

if __name__ == '__main__':
  feature_extractor = models.resnet34(pretrained=True)
  model = ORGPoseNet(feature_extractor)
  # print(model)
  # for param in model.named_parameters():
  #   print(param[0])
  # for m in model.modules():
  #   if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
  #     print(m.weight)
  # for m in model.graphmodel.modules():
  #   if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
  #     print(m)

  # feature_extractor = models.resnet34(pretrained=True)
  # model = ORGPoseNet(feature_extractor)
  # model = model.cuda()
  # x = torch.rand(64,8,8).cuda()
  # bbox = torch.rand(0,4).cuda()
  # # bbox = torch.rand(10,4).cuda()
  # output = model.bbox_to_feature(x, bbox)
  # print(output.size())

  # x = torch.rand(10,32,8,8)
  # meta = []
  # for i in range(10):
  #   meta0 = {'mask':torch.zeros(256,256,3)}
  #   # meta0 = {'mask':torch.randint(0,10,(256,256,3))}
  #   meta.append(meta0)
  # output = model.obj_feature(x, meta)
  # print(output[0].size())

  # x=torch.rand(1,10,10).cuda()
  # dis, idx = knn(x, k=2)
  # print(dis.size())
  # print(idx.size())

  # x=torch.rand(1,256,10).cuda()
  # n_obj = torch.tensor((0,2,5,10)).cuda()
  # x, dis = get_feature(x, n_obj, k=3)
  # print(x.size())
  # print(dis.size())

  # x=torch.rand(1,256,28)
  # N_obj = torch.tensor((0,4,13,28))
  # n_obj = torch.tensor((0,2,5,10))
  # aggr = 'weighted_sum'
  # dis = torch.rand(1,28)
  # dis -= 1
  # output = batch_aggr(x, N_obj, n_obj, aggr=aggr, dis=dis)
  # print(output.size())

  # x = torch.rand(1,256,10)
  # n_obj =torch.tensor((0,2,5,10))
  # output = batch_pool(x, n_obj)
  # print(output.size())

  # model = GraphModel(5, aggr='weighted_sum')
  # model = model.cuda()
  # fs = []
  # n_obj = [2,3,5]
  # # n_obj = [2,0,8]
  # # n_obj = [0,0,0]
  # for i in range(len(n_obj)):
  #   f_obj = torch.rand(n_obj[i],64).cuda()
  #   fs.append(f_obj)
  # output = model(fs)
  # print(output.size())

  # feature_extractor = models.resnet34(pretrained=True)
  # model = ORGPoseNet(feature_extractor)
  # model = model.cuda()
  # x = torch.rand(3,3,256,256).cuda()
  # meta = []
  # for i in range(3):
  #   # meta0 = {'bbox':torch.zeros(0,4).cuda(),
  #   #          'msk':torch.zeros(256,256,3).cuda(),
  #   #          'fea':torch.zeros(0,256).cuda(),
  #   #          'idx':torch.zeros(0).cuda()}
  #   meta0 = {'bbox':torch.rand(10,4).cuda(),
  #            'msk':torch.randint(0,10,(256,256,3)).cuda(),
  #            'fea':torch.rand(10,256).cuda(),
  #            'idx':torch.rand(10).cuda()}
  #   meta.append(meta0)
  # output = model(x, meta)
  # feats = output[1]
  # output = output[0]
  # print(output.size())
  # print(feats.size())
