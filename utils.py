import random
import numpy as np
from six.moves import xrange
import torch

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


import torch

rb = ReplayBuffer(3)
print(rb.buffer)

episodes = torch.tensor([[0,1,2],[3,4,5]])
rb.add(episodes)
episodes = torch.tensor([[3,4,5],[6,7,8]])
rb.add(episodes)
episodes = torch.tensor([[6,7,8],[9,10,11]])
rb.add(episodes)
print(rb.buffer)

print(rb.get_batch(2))
print(torch.stack(rb.get_batch(2)))
