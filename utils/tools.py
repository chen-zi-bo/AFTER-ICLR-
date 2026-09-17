import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, extra_state=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, extra_state=extra_state)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, extra_state=extra_state)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, extra_state=None):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        checkpoint = {'model_state_dict': model.state_dict(), 'val_loss': val_loss}
        if extra_state is not None:
            checkpoint.update(extra_state)
        torch.save(checkpoint, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

    def state_dict(self):
        return {
            'patience': self.patience,
            'verbose': self.verbose,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'val_loss_min': self.val_loss_min,
            'delta': self.delta,
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.patience = state.get('patience', self.patience)
        self.verbose = state.get('verbose', self.verbose)
        self.counter = state.get('counter', self.counter)
        self.best_score = state.get('best_score', self.best_score)
        self.early_stop = state.get('early_stop', self.early_stop)
        self.val_loss_min = state.get('val_loss_min', self.val_loss_min)
        self.delta = state.get('delta', self.delta)


def load_checkpoint(path, map_location=None):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support the weights_only argument.
        return torch.load(path, map_location=map_location)


def extract_model_state(checkpoint):
    if checkpoint is None:
        return None
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

import re

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Discretizer:
    def __init__(self, low_limit=-1, high_limit=1, n_tokens=10002):
        self.scaler = MinMaxScaler()

        self.boundaries = np.linspace(low_limit, high_limit, n_tokens - 1)
        self.centers = (self.boundaries[1:] + self.boundaries[:-1]) / 2
        self.centers = np.concatenate((self.centers[:1], self.centers, self.centers[-1:]))

    def get_centers(self):
        return self.centers

    def discretize(self, context, fit_length=None):
        fit_length = len(context) if fit_length is None else fit_length
        self.scaler.fit(context[:fit_length].reshape(-1, 1))
        scaled_context = self.scaler.transform(context.reshape(-1, 1)).reshape(-1) - 0.5

        bin_ids = np.digitize(x=scaled_context, bins=self.boundaries, right=True)
        dispersed_context = self.centers[bin_ids]

        dispersed_context[np.isnan(context)] = np.nan

        return dispersed_context

    def inverse_discretize(self, scaled_context):
        context = self.scaler.inverse_transform(scaled_context.reshape(-1, 1) + 0.5).reshape(-1)

        return context


class Serializer:
    def __init__(self, prec=4, time_sep=" ", time_flag="###", nan_flag="Nan"):
        self.prec = prec
        self.time_sep = time_sep
        self.time_flag = time_flag
        self.nan_flag = nan_flag

    def serialize(self, context):
        serialized_context = np.array([f"{self.time_flag}{i:.{self.prec}f}{self.time_flag}" for i in context])
        serialized_context[np.isnan(context)] = f"{self.time_flag}{self.nan_flag}{self.time_flag}"
        serialized_context = self.time_sep.join(serialized_context)

        return serialized_context

    def inverse_serialize(self, serialized_context):
        pattern = rf"{self.time_flag}(.*?){self.time_flag}"
        matches = re.findall(pattern, serialized_context)

        context = []
        for num in matches:
            try:
                context.append(float(num))
            except ValueError as e:
                print(e)
                context.append(np.NaN)

        context = np.array(context)

        return context
