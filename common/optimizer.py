import torch.optim as optim

class Optimizer:
  """
  Wrapper around torch.optim + learning rate
  """
  def __init__(self, params, method, base_lr, weight_decay, **kwargs):
    self.method = method
    self.base_lr = base_lr

    if self.method == 'sgd':
      self.lr_decay = kwargs.pop('lr_decay')
      self.lr_stepvalues = sorted(kwargs.pop('lr_stepvalues'))
      self.learner = optim.SGD(params, lr=self.base_lr,
        weight_decay=weight_decay, **kwargs)
    elif self.method == 'adam':
      self.lr_decay_epoch = kwargs.pop('lr_decay_epoch')
      self.learner = optim.Adam(params, lr=self.base_lr,
        weight_decay=weight_decay, **kwargs)
    elif self.method == 'rmsprop':
      self.learner = optim.RMSprop(params, lr=self.base_lr,
        weight_decay=weight_decay, **kwargs)

  def adjust_lr(self, epoch):

    lr = self.base_lr
    if epoch >= self.lr_decay_epoch:
      lr = self.base_lr / 10
    for param_group in self.learner.param_groups[1:]:
      param_group['lr'] = lr

    return lr

  def mult_lr(self, f):
    for param_group in self.learner.param_groups:
      param_group['lr'] *= f
