import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parameter import Parameter

import net
import jem

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Accuracy, Loss

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_data_loaders(batch_size):
    # 1. MNISTのデータセットを取得
    mnist = fetch_mldata('MNIST original', data_home='./')

    # 2. データの設定（入力データは閉区間[0, 1]に正規化する）
    x = mnist.data / 255
    y = mnist.target

    # 3. DataLoaderの作成

    # 3-1. データを学習用とテスト用に分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/7, random_state=0)
 
    # 3-2. データのフォーマットを変換：PyTorchでの形式 = [画像数，チャネル数，高さ，幅]
    x_train = x_train.reshape(60000, 28* 28)
    x_test = x_test.reshape(10000, 28*28)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))

    scaler.fit(x)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print(x_train.max(), x_train.min())
    print(x_test.max(), x_test.min())

 
    # 3-3. PyTorchのテンソルに変換
    x_train = Parameter(torch.Tensor(x_train))
    x_test = torch.Tensor(x_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
 
    # 3-4. 入力（x）とラベル（y）を組み合わせて最終的なデータを作成
    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)
 
    # 3-5. DataLoaderを作成
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main(batch_size, epochs):
 
    # 1. GPUの設定（PyTorchでは明示的に指定する必要がある）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_loader, test_loader = get_data_loaders(batch_size)
 
    # 2. モデル作成
#    model = net.CNN(num_classes=num_classes).to(device)
    model = net.Net(1000,10).to(device)
    print(model) # ネットワークの詳細を確認用に表示
 
    # 3. 損失関数を定義
    criterion = nn.CrossEntropyLoss()
 
    # 4. 最適化手法を定義（ここでは例としてAdamを選択）
#    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters())

    buffer_size = 10000
    B = jem.ReplayBuffer(buffer_size)
    m_uniform = torch.distributions.uniform.Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
    B.add(m_uniform.sample((100, 784)).squeeze())

    trainer = jem.create_supervised_trainer2(model, optimizer, criterion, device=device, replay_buffer=B)

    train_evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),'loss':Loss(criterion)}, device=device)
    test_evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),'loss':Loss(criterion)}, device=device)

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )


    log_interval = 10

    # 5. ログ出力
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        i = (engine.state.iteration - 1) % len(train_loader) + 1
        if i % log_interval == 0:
            pbar.desc = desc.format(engine.state.output['loss'])
            pbar.update(log_interval)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        engine_metrics = engine.state.metrics
        avg_loss = engine_metrics['loss']
        avg_loss_elf = engine_metrics['loss_elf']
        avg_loss_gen = engine_metrics['loss_gen']
        tqdm.write(
            "Engine Results - Epoch: {}  Avg loss: {:.4f} Avg loss elf: {:.4f} Avg loss gen:{:.4f}"
            .format(engine.state.epoch, avg_loss, avg_loss_elf, avg_loss_gen)
        )

        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.4f}"
            .format(engine.state.epoch, avg_accuracy, avg_loss)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        test_evaluator.run(test_loader)
        metrics = test_evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.4f}"
            .format(engine.state.epoch, avg_accuracy, avg_loss))
        pbar.n = pbar.last_print_n = 0

    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    # 5. checkpoint setting
    best_handler = ModelCheckpoint(dirname='./checkpoints', filename_prefix='best',
            n_saved=3,score_name='loss',score_function=score_function,
            create_dir=True,require_empty=False)
    test_evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_handler, {'mymodel': model})

    early_handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset)
    test_evaluator.add_event_handler(Events.COMPLETED, early_handler)

    # 6. 実行
    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()
 
if __name__ == '__main__':
    batch_size = 100
#    num_classes = 10
    epochs = 20
    main(batch_size, epochs)
