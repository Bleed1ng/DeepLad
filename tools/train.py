import gc
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.log import LogDataset
from dataset.sample import sliding_window, session_window
from tools.utils import save_parameters


class Trainer:
    def __init__(self, model, options):
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.batch_size = options['batch_size']

        self.device = options['device']
        self.lr_step = options['lr_step']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.accumulation_step = options['accumulation_step']
        self.max_epoch = options['max_epoch']

        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.sample = options['sample']
        self.feature_num = options['feature_num']

        os.makedirs(self.save_dir, exist_ok=True)
        if self.sample == 'sliding_window':
            train_logs, train_labels = sliding_window(self.data_dir,
                                                      datatype='train',
                                                      window_size=self.window_size)
            val_logs, val_labels = sliding_window(self.data_dir,
                                                  datatype='val',
                                                  window_size=self.window_size,
                                                  sample_ratio=0.001)
        elif self.sample == 'session_window':
            train_logs, train_labels = session_window(self.data_dir, datatype='train')
            val_logs, val_labels = session_window(self.data_dir, datatype='val')
        else:
            raise NotImplementedError

        train_dataset = LogDataset(logs=train_logs,
                                   labels=train_labels,
                                   seq=self.sequentials,
                                   quan=self.quantitatives,
                                   sem=self.semantics)
        valid_dataset = LogDataset(logs=val_logs,
                                   labels=val_labels,
                                   seq=self.sequentials,
                                   quan=self.quantitatives,
                                   sem=self.semantics)

        del train_logs
        del val_logs
        gc.collect()

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=True)

        self.num_train_log = len(train_dataset)
        self.num_valid_log = len(valid_dataset)

        print('Find %d train logs, %d validation logs' %
              (self.num_train_log, self.num_valid_log))
        print('Train batch size %d, Validation batch size %d' %
              (options['batch_size'], options['batch_size']))

        self.model = model.to(self.device)

        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=options['lr'],
                                             momentum=0.9)
        elif options['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=options['lr'],
                                              betas=(0.9, 0.999))
        else:
            raise NotImplementedError

        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_score = -1
        save_parameters(options, self.save_dir + "parameters.txt")
        self.log = {
            "train": {key: [] for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: [] for key in ["epoch", "lr", "time", "loss"]}
        }
        if options['resume_path'] is not None:
            if os.path.isfile(options['resume_path']):
                self.resume(options['resume_path'], load_optimizer=True)
            else:
                print("Checkpoint not found")

    # 恢复模型
    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.log = checkpoint['log']
        self.best_f1_score = checkpoint['best_f1_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    # 保存模型
    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        save_path = self.save_dir + self.model_name + "_" + suffix + ".pth"
        torch.save(checkpoint, save_path)
        print("Save model checkpoint at {}".format(save_path))

    # 保存日志
    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv",
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    # 训练
    def train(self, epoch):
        self.log['train']['epoch'].append(epoch)
        start_time = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: train | ⏰: %s | Learning rate: %f" % (epoch, start_time, lr))
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start_time)
        self.model.train()  # 设置模型为训练模式，用于启用训练相关的功能，例如Dropout和BatchNorm
        self.optimizer.zero_grad()  # 将梯度初始化为0，清空之前的梯度信息
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        tbar = tqdm(self.train_loader, desc="\r")  # 进度条，可视化训练进度
        num_batch = len(self.train_loader)  # 训练集batch数量，用于计算平均loss
        total_losses = 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().detach().to(self.device))  # 将当前batch的特征值添加到列表中，并将其移至指定的设备（如GPU）上
            output = self.model(features=features, device=self.device)  # 使用模型前向传播计算输出
            loss = criterion(output, label.to(self.device))  # 计算当前batch的loss
            total_losses += float(loss)
            loss /= self.accumulation_step  # 对损失值进行累积梯度的平均处理。
            loss.backward()  # 反向传播，计算当前梯度
            if (i + 1) % self.accumulation_step == 0:
                # 当累积梯度达到指定步数时，执行一次优化器的参数更新操作，并清空梯度
                self.optimizer.step()
                self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))
        self.log['train']['loss'].append(total_losses / num_batch)

    # 验证
    def valid(self, epoch):
        self.model.eval()
        self.log['valid']['epoch'].append(epoch)
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log['valid']['lr'].append(lr)
        start_time = time.strftime("%H:%M:%S")
        print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch, start_time))
        self.log['valid']['time'].append(start_time)
        total_losses = 0
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)
        for i, (log, label) in enumerate(tbar):
            with torch.no_grad():
                features = []
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))
                output = self.model(features=features, device=self.device)
                loss = criterion(output, label.to(self.device))
                total_losses += float(loss)
        print("Validation loss:", total_losses / num_batch)
        self.log['valid']['loss'].append(total_losses / num_batch)

        # 保存最优模型
        if total_losses / num_batch < self.best_loss:
            self.best_loss = total_losses / num_batch
            self.save_checkpoint(epoch, save_optimizer=False, suffix="bestloss")

    def start_train(self):
        """
        训练
        随迭代调整学习率
        """
        for epoch in range(self.start_epoch, self.max_epoch):
            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            if epoch in self.lr_step:
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio

            self.train(epoch)

            if epoch >= self.max_epoch // 2 and epoch % 2 == 0:
                self.valid(epoch)
                self.save_checkpoint(epoch, save_optimizer=True, suffix="epoch" + str(epoch))
            self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
            self.save_log()

        # 绘制学习率变化曲线
        plt.plot(self.log['train']['epoch'], self.log['train']['lr'], 'r-')
        ax1 = plt.gca()
        ax1.set_title('Learning Rate')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Learning Rate')
        plt.savefig(self.save_dir + 'learning_rate.png')
        plt.show()
        # loss曲线
        plt.plot(self.log['train']['epoch'], self.log['train']['loss'], 'r-', )
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(self.save_dir + 'loss.png')
        plt.show()
