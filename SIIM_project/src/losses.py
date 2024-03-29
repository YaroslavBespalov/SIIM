import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import pydoc

class BCE(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCE, self).__init__()
        self.loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        m = nn.Sigmoid()
        targets = targets.float().view(-1)
        return self.loss(m(logits), targets)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight= torch.tensor(3.5), size_average=True):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, size_average=size_average)

    def forward(self, logits, targets):
        #m = nn.Sigmoid()
        logits = logits.float().view(-1)
        targets = targets.float().view(-1)
        return self.loss(logits, targets)

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight, size_average)

    def forward(self, logits, targets):
        targets = targets.type(torch.cuda.LongTensor).view(-1)
        return self.loss(logits, targets)


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        loss = 0
        for i in range(2):
            probs_flat = probs[:, i].contiguous().view(-1)
            targets_flat = (targets == i + 1).float().contiguous().view(-1)
            loss += self.bce_loss(probs_flat, targets_flat)
        return loss


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2):
#         super().__init__()
#         self.gamma = gamma
#         self.bce_with_logits = nn.BCEWithLogitsLoss()
#
#     def forward(self, input, target):
#         return self.bce_with_logits((1 - torch.sigmoid(input)) ** self.gamma * F.logsigmoid(input), target)

# class FocalLoss(nn.Module): # Фокал лосс кости
#     def __init__(self, gamma=2):
#         super().__init__()
#         self.gamma = gamma
#
#     def forward(self, input, target):
#         if not (target.size() == input.size()):
#             raise ValueError("Target size ({}) must be the same as input size ({})"
#                              .format(target.size(), input.size()))
#
#         max_val = (-input).clamp(min=0)
#         loss = input - input * target + max_val + \
#                ((-max_val).exp() + (-input - max_val).exp()).log()
#
#         invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
#         loss = (invprobs * self.gamma).exp() * loss
#
#         return loss.sum(dim=1).mean()



# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1., gamma=2.):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, inputs, targets, **kwargs):
#         CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
#         pt = torch.exp(-CE_loss)
#         F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
#         return F_loss.mean()


class LossBinaryDice(nn.Module):
    def __init__(self, dice_weight=1):
        super(LossBinaryDice, self).__init__()
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, outputs, targets):
        # targets = targets.squeeze().view(-1).float()
        # outputs = outputs.squeeze().view(-1).float()
        targets = targets.squeeze().float()
        outputs = outputs.squeeze().float()
        loss = self.nll_loss(outputs, targets)

        if self.dice_weight:
            smooth = torch.tensor(1e-15).float()
            target = (targets > 0.0).float()
            prediction = F.sigmoid(outputs).float()
            dice_part = (1 - (2 * torch.sum(prediction * target, dim=(1,2)) + smooth) / \
                         (torch.sum(prediction, dim=(1,2)) + torch.sum(target, dim=(1,2)) + smooth))

            loss += self.dice_weight * dice_part.mean()
        return loss


class LossMultiLabelDice(nn.Module):
    def __init__(self, dice_weight=1):
        super(LossMultiLabelDice, self).__init__()
        self.focal_loss = FocalLoss()
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
        loss = self.focal_loss(outputs, targets)
        if self.dice_weight:
            loss += self.dice_weight * self.dice_coef_multilabel(outputs, targets)
        return loss


class GeneralizedDiceLoss2D(nn.Module):
    def __init__(self):
        """
        Implementation of Generalized Dice Loss from https://arxiv.org/pdf/1707.03237.pdf

        Doesn't work!
        """
        super(GeneralizedDiceLoss2D, self).__init__()

    def forward(self, logits, masks):
        """
        Computes Generalized Dice Loss

        :param logits: tensor of shape batch_size x c x d1 x d2 ... x dk , where c is the number of classes, k is the
                       number of image dimensions
        :param masks: tensor of shape batch_size x d1 x d2 ... x dk consists of class labels (integers) of each element
        :return:
        """

        EPS = 1e-8

        one_hot_masks = F.one_hot(masks, num_classes=3).to(torch.float32)
        probas = F.softmax(logits, dim=1)
        probas = probas.permute(0, 2, 3, 1)

        weights = 1.0 / (torch.pow(torch.sum(one_hot_masks, dim=(1, 2)), 1) + EPS)

        intersection = torch.sum(torch.sum(probas * one_hot_masks, dim=(1, 2)) * weights)
        union = torch.sum(torch.sum(probas + one_hot_masks, dim=(1, 2)) * weights)

        dice_loss = 1 - 2 * (intersection + EPS) / (union + EPS)

        return dice_loss


class MeanDiceLoss2D(nn.Module):

    def __init__(self):
        """
        Implementation of Mean Dice Loss from https://arxiv.org/pdf/1707.01992.pdf
        """
        super(MeanDiceLoss2D, self).__init__()

    def forward(self, logits, masks):
        """
        Computes Generalized Dice Loss

        :param logits: tensor of shape batch_size x c x d1 x d2 ... x dk , where c is the number of classes, k is the
                       number of image dimensions
        :param masks: tensor of shape batch_size x d1 x d2 ... x dk consists of class labels (integers) of each element
        :return:
        """
        EPS = 1e-10

        one_hot_masks = F.one_hot(masks, num_classes=3).to(torch.float32)
        probas = F.softmax(logits, dim=1)
        probas = probas.permute(0, 2, 3, 1)

        intersection = torch.sum(probas * one_hot_masks, dim=(1, 2))
        union = torch.sum(torch.pow(probas, 2), dim=(1, 2)) + torch.sum(torch.pow(one_hot_masks, 2), dim=(1, 2))
        dice_loss = torch.mean(2 * (intersection + EPS) / (union + EPS), dim=1)

        return -torch.mean(dice_loss)


class CrossEntropyDiceLoss2D(nn.Module):

    def __init__(self, dice_weight=0.7):
        super(CrossEntropyDiceLoss2D, self).__init__()
        self.dice_weight = dice_weight
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = MeanDiceLoss2D()

    def forward(self, logits, targets):
        return self.ce_loss(logits, targets) \
               + self.dice_weight * self.dice_loss(logits, targets)


class MeanDiceLoss3D(nn.Module):

    def __init__(self):
        """
        Implementation of Mean Dice Loss from https://arxiv.org/pdf/1707.01992.pdf
        """
        super(MeanDiceLoss3D, self).__init__()

    def forward(self, logits, masks):
        """
        Computes Generalized Dice Loss

        :param logits: tensor of shape batch_size x c x d1 x d2 ... x dk , where c is the number of classes, k is the
                       number of image dimensions
        :param masks: tensor of shape batch_size x d1 x d2 ... x dk consists of class labels (integers) of each element
        :return:
        """
        EPS = 1e-10

        one_hot_masks = F.one_hot(masks, num_classes=3).to(torch.float32)
        probas = F.softmax(logits, dim=1)
        probas = probas.permute(0, 2, 3, 4, 1)

        intersection = torch.sum(probas * one_hot_masks, dim=(1, 2, 3))
        union = torch.sum(torch.pow(probas, 2), dim=(1, 2, 3)) + torch.sum(torch.pow(one_hot_masks, 2), dim=(1, 2, 3))
        dice_loss = torch.mean(2 * (intersection + EPS) / (union + EPS), dim=1)

        return -torch.mean(dice_loss)


class CrossEntropyDiceLoss3D(nn.Module):

    def __init__(self, dice_weight=0.7):
        super(CrossEntropyDiceLoss3D, self).__init__()
        self.dice_weight = dice_weight
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = MeanDiceLoss3D()

    def forward(self, logits, targets):
        return self.ce_loss(logits, targets) \
               + self.dice_weight * self.dice_loss(logits, targets)


## SIIM

def dice_loss_init(input, target):
    input = torch.sigmoid(input)
    smooth = 1e-15
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def dice_loss(prediction, target):
    smooth = torch.tensor(1e-15).float()
    prediction = prediction.sigmoid()
    # prediction = (prediction.view(-1)).double()
    # target = target.view(-1).double()
    # prediction = (prediction > self.threshold).float()
    target = target.float()
    dice = (2 * torch.sum(prediction * target, dim=(1, 2, 3)) + smooth) / \
    (torch.sum(prediction, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + smooth)
    return dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()

class MixedLoss(nn.Module):
    def __init__(self, alpha=5., gamma=2.):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)


    def forward(self, input, target):
        # print("FOCAL", self.focal(input, target))
        loss = (self.alpha*self.focal(input, target) + torch.log(1 - dice_loss(input, target)))
        return loss.mean()
