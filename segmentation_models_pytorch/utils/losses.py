import torch.nn as nn

from . import base
from . import functional as F
from ..base.modules import Activation


class JaccardLoss(base.Loss):

    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass


class FocalTversky(base.loss)
     def __init__(self, alpha=.7, gamma=.75, smooth=1., **kwargs):
        super().__init__(**kwargs)
        self.alpha=alpha
        self.gamma=gamma
        self.smooth=smooth

    def tversky(self,y_true, y_pred):
        y_true_pos = y_true.view(-1,1)
        y_pred_pos = y_pred.view(-1,1)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1-y_pred_pos))
        false_pos = torch.sum((1-y_true_pos)*y_pred_pos)

        return (true_pos + self.smooth)/(true_pos + self.alpha*false_neg + (1-self.alpha)*false_pos + self.smooth)

    def tversky_loss(self,y_true, y_pred):
        return 1 - tversky(y_true,y_pred)

    def forward(self,y_true,y_pred):
        y_pred = F.sigmoid(y_pred)
        pt_1 = tversky(y_true, y_pred)
        return torch.pow((1-pt_1), self.gamma)
