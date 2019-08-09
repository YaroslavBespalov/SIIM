import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from albumentations import *
import torch
from sklearn.metrics import roc_auc_score
import pandas as pd
from scipy.ndimage import label


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


class map3(nn.Module):
    def __init__(self, ):
        super(map3, self).__init__()

    def forward(self, preds, targs):
        # targs = np.where(targs==1)[1]
        predicted_idxs = preds.sort(descending=True)[1]
        top_3 = predicted_idxs[:, :3]
        res = mapk([[t] for t in targs.cpu().numpy()], top_3.cpu().numpy(), 3)
        return -torch.tensor(res)


class Dice(nn.Module):
    def __init__(self, threshold=0.5):
        super(Dice, self).__init__()
        self.threshold = threshold

    def forward(self, prediction, target):
        smooth = torch.tensor(1e-15).float()
        prediction = prediction.sigmoid()
        # prediction = (prediction.view(-1)).double()
        # target = target.view(-1).double()
        prediction = (prediction > self.threshold).float()
        target = target.float()
        dice = (2 * torch.sum(prediction * target, dim=(1, 2, 3)) + smooth) / \
        (torch.sum(prediction, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + smooth)
        return dice.mean()


class LossMultiLabelDice(nn.Module):
    def __init__(self, dice_weight=1):
        super(LossMultiLabelDice, self).__init__()
        self.dice_weight = dice_weight
        self.smooth = 1e-50

    def dice_coef(self, y_true, y_pred):
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)

    def dice_coef_multilabel(self, y_true, y_pred, numLabels=3):
        dice = 0

        for index in range(1, numLabels):
            dice += self.dice_coef(y_true[:, index], y_pred[:, index])
        return dice / 2

    def forward(self, outputs, targets):

        # print(outputs.size())
        targets = targets.squeeze().permute(0, 3, 1, 2)
        # print(targets.size())
        loss = 0
        if self.dice_weight:
            loss += self.dice_weight * self.dice_coef_multilabel(outputs, targets)
        return loss

class Accuracy(nn.Module):
    def __init__(self, ):
        super(Accuracy, self).__init__()

    def forward(self, prediction, target):
        prediction=torch.squeeze(prediction)
        return (prediction.argmax(dim=0) == target).float().mean()

class Binary_Accuracy(nn.Module):
    def __init__(self, ):
        super(Binary_Accuracy, self).__init__()

    def forward(self, prediction, target, treshold=0.5):
        prediction = torch.squeeze(prediction)
        prediction = torch.nn.Sigmoid()(prediction)
        prediction = (prediction > treshold).float()
        return (prediction == target.float()).float().mean()


class Accuracy50(nn.Module):
    def __init__(self, ):
        super(Accuracy50, self).__init__()

    def forward(self, prediction, target, threshold=0.5):
        prediction = nn.Softmax(dim=-1)(prediction).detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        prediction = [0 if pred[1] < threshold else 1 for pred in prediction]
        metric = (prediction == target).astype(float).mean()
        return metric


class Accuracy10(nn.Module):
    def __init__(self, ):
        super(Accuracy10, self).__init__()

    def forward(self, prediction, target, threshold=0.1):
        prediction = nn.Softmax(dim=-1)(prediction).detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        prediction = [0 if pred[1] < threshold else 1 for pred in prediction]
        metric = (prediction == target).astype(float).mean()
        return metric


class Accuracy90(nn.Module):
    def __init__(self, ):
        super(Accuracy90, self).__init__()

    def forward(self, prediction, target, threshold=0.9):
        prediction = nn.Softmax(dim=-1)(prediction).detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        prediction = [0 if pred[1] < threshold else 1 for pred in prediction]
        metric = (prediction == target).astype(float).mean()
        return metric


class roc_auc_compute_fn(nn.Module):
    def __init__(self, ):
        super(roc_auc_compute_fn, self).__init__()

    def forward(self, prediction, target):
        prediction = nn.Softmax(dim=-1)(prediction).detach().cpu().numpy()
        y_true = target.detach().cpu().numpy()
        y_pred = [pred[1] for pred in prediction]

        return roc_auc_score(y_true, y_pred)

## SIIIM
class metriccc(nn.Module):
    def __init__(self, ):
        super(metriccc, self).__init__()

    def predict(X, threshold):
        X_p = np.copy(X)
        preds = (X_p > threshold).astype('uint8')
        return preds

    def forward(self, probability, truth, threshold=0.5, reduction='none'):
        '''Calculates dice of positive and negative images seperately'''
        '''probability and truth must be torch tensors'''
        batch_size = len(truth)
        with torch.no_grad():
            probability = probability.view(batch_size, -1)
            truth = truth.view(batch_size, -1)
            assert(probability.shape == truth.shape)

            p = (probability > threshold).float()
            t = (truth > 0.5).float()

            t_sum = t.sum(-1)
            p_sum = p.sum(-1)
            neg_index = torch.nonzero(t_sum == 0)
            pos_index = torch.nonzero(t_sum >= 1)

            dice_neg = (p_sum == 0).float()
            dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

            dice_neg = dice_neg[neg_index]
            dice_pos = dice_pos[pos_index]
            dice = torch.cat([dice_pos, dice_neg])

            dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
            dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
            dice = dice.mean().item()

            num_neg = len(neg_index)
            num_pos = len(pos_index)

        return dice
