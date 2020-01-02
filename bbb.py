import torch
import torch.nn as nn
import torch.nn.functional as F
import net
from torch.nn.parameter import Parameter

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

class Net(nn.Module):
    def __init__(self, n_units, n_out):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, n_units) # 入力層から隠れ層へ
        self.l2 = nn.Linear(n_units, n_units) # 入力層から隠れ層へ
        self.l3 = nn.Linear(n_units, 10) # 隠れ層から出力層へ

    def forward(self, x):
#        x = x.view(-1, 28 * 28) # テンソルのリサイズ: (N, 1, 28, 28) --> (N, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = x.exp().sum(axis=1).log()
        return x

def jacobian(f, x):
    """Computes the Jacobian of f w.r.t x.

    This is according to the reverse mode autodiff rule,

    sum_i v^b_i dy^b_i / dx^b_j = sum_i x^b_j R_ji v^b_i,

    where:
    - b is the batch index from 0 to B - 1
    - i, j are the vector indices from 0 to N-1
    - v^b_i is a "test vector", which is set to 1 column-wise to obtain the correct
        column vectors out ot the above expression.

    :param f: function R^N -> R^N
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, N, N]
    """

    B, N = x.shape
    y = f(x)
    y = y.view(100,1)
    v = torch.zeros_like(y)
    v[:, 0] = 1.
    dy_i_dx = torch.autograd.grad(y,
                   x,
                   grad_outputs=v,
                   retain_graph=True,
                   create_graph=True,
                   allow_unused=True)[0]  # shape [B, N]

    return dy_i_dx

mnist = fetch_mldata('MNIST original', data_home='./')
x = mnist.data / 255
y = mnist.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/7, random_state=0)
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 *28)

x_train = x_train[0:100]
y_train = y_train[0:100]
x_train = torch.Tensor(x_train)
y_train = torch.LongTensor(y_train)

x_train.requires_grad_()

net = Net(1000,10)
jac = jacobian(net, x_train)

print(jac)
print(jac.shape)
