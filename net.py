import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return x

class CNN(nn.Module):
 
    def __init__(self, num_classes):
        """
        Convolutional Neural Network
        
        ネットワーク構成：
            input - CONV - CONV - MaxPool - CONV - CONV - MaxPool - FC - output
            ※MaxPoolの直後にバッチ正規化を実施
 
        引数：
            num_classes: 分類するクラス数（＝出力層のユニット数）
        """
 
        super(CNN, self).__init__() # nn.Moduleを継承する
 
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1), # 出力サイズ: チャネル=16, 高さ=27, 幅=27
            nn.BatchNorm2d(16)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1), # 出力サイズ: チャネル=32, 高さ=26, 幅=26
            nn.BatchNorm2d(32)
        )
        self.full_connection = nn.Sequential(
            nn.Linear(in_features=32*26*26, out_features=512), # in_featuresは直前の出力ユニット数
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=num_classes)
        )
 
 
    # Forward計算の定義
    # 参考：Define by Runの特徴（入力に合わせてForward計算を変更可）
    def forward(self, x):
 
        x = self.block1(x)
        x = self.block2(x)
 
        # 直前のMaxPoolの出力が2次元（×チャネル数）なので，全結合の入力形式に変換
        # 参考：KerasのFlatten()と同じような処理
        x = x.view(x.size(0), 32 * 26 * 26)
 
        y = self.full_connection(x)
        
        return y
