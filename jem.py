import random
import numpy as np
from six.moves import xrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor

eta = 20
alpha = 1.0
sigma = 0.01
rou = 0.05

def jacobian(f, x):
    """Computes the Jacobian of f w.r.t x.

    This is according to the reverse mode autodiff rule,

    sum_i v^b_i dy^b_i / dx^b_j = sum_i x^b_j R_ji v^b_i,

    where:
    - b is the batch index from 0 to B - 1
    - i, j are the vector indices from 0 to N-1
    - v^b_i is a "test vector", which is set to 1 column-wise to obtain the correct
        column vectors out ot the above expression.

    :param f: function R^N -> R
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, N]
    """

    B, N = x.shape
    y = f(x)
    v = torch.zeros_like(y)
    v[:, 0] = 1.
    dy_i_dx = torch.autograd.grad(y,
                   x,
                   grad_outputs=v,
                   retain_graph=True,
                   create_graph=True,
                   allow_unused=True)[0]  # shape [B, N]
    return dy_i_dx

class ReplayBuffer(object):

  def __init__(self, max_size):
    self.max_size = max_size
    self.cur_size = 0
    self.buffer = {}
    self.init_length = 0

  def __len__(self):
    return self.cur_size

  def seed_buffer(self, episodes):
    self.init_length = len(episodes)
    self.add(episodes, np.ones(self.init_length))

  def add(self, episodes, *args):
    """Add episodes to buffer."""
    idx = 0
    while self.cur_size < self.max_size and idx < len(episodes):
      self.buffer[self.cur_size] = episodes[idx]
      self.cur_size += 1
      idx += 1

    if idx < len(episodes):
      remove_idxs = self.remove_n(len(episodes) - idx)
      for remove_idx in remove_idxs:
        self.buffer[remove_idx] = episodes[idx]
        idx += 1

    assert len(self.buffer) == self.cur_size

  def remove_n(self, n):
    """Get n items for removal."""
    # random removal
    idxs = random.sample(xrange(self.init_length, self.cur_size), n)
    return idxs

  def get_batch(self, n):
    """Get batch of episodes to train on."""
    # random batch
    idxs = random.sample(xrange(self.cur_size), n)
    return [self.buffer[idx] for idx in idxs]

  def update_last_batch(self, delta):
    pass


def LogSumExp(x):
    x = torch.logsumexp(x, 1)
    x = x.view(len(x), 1)
    return x

def sampler(f, B, batch_size, dim, device):
    m_uniform = torch.distributions.uniform.Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
    m_normal = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    batch_size1 = int(batch_size*(1-rou))
    batch_size2 = batch_size - batch_size1
    x1 = torch.stack(B.get_batch(batch_size1))
    x2 = m_uniform.sample((batch_size2, dim)).squeeze()
    x = torch.cat([x1,x2],dim=0)
    x = x.to(device)
    x.requires_grad_(True)
    for i in range(eta):
        jac = jacobian(f,x)
        if torch.isnan(jac).any():
            print("jac nan")
            exit(1)
        x = x + alpha * jac + sigma * m_normal.sample(x.shape).squeeze().to(device)
    x = x.detach()
    return x

def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.

    """
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))

def create_supervised_trainer2(model, optimizer, loss_fn,
                              replay_buffer,
                              device=None, non_blocking=False,
                              prepare_batch=_prepare_batch):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()

        LogSumExpf = lambda x: LogSumExp(model(x))
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        x = x.detach()
        y_pred = model(x)
        loss_elf = loss_fn(y_pred, y)
        x_sample = sampler(LogSumExpf, replay_buffer, x.shape[0], x.shape[1], device)
        replay_buffer.add(x_sample.cpu())
        loss_gen =-(LogSumExpf(x) - LogSumExpf(x_sample)).mean()
        loss = loss_elf + loss_gen
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)

