import torch
import torch.nn as nn


# https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, has_miner=True, weights=[1.0, 1.0], ignore_index=-1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.has_miner = has_miner
        self.weights = torch.FloatTensor(weights).cuda()
        self.ignore_index = ignore_index

    def forward(self, data, labels):
        if len(data.shape) == 4:
            data = data.flatten()
        if len(labels.shape) == 3:
            labels = labels.flatten()

        # filtering out pixels
        coord = torch.where(labels != self.ignore_index)
        labels = labels[coord]
        data = data[coord]

        if self.has_miner:
            data, labels, self.weights = self.miner(data, labels)

        loss_contrastive = torch.mean(self.weights[1] * labels * torch.pow(data, 2) +
                                      self.weights[0] * (1 - labels) *
                                      torch.pow(torch.clamp(self.margin - data, min=0.0), 2))
        return loss_contrastive

    def miner(self, data, labels):
        all_pos_values = data[labels.bool()]
        all_neg_values = data[(1 - labels).bool()]

        # get all **hard** samples of the negative class with dist < margin
        neg_hard = all_neg_values[all_neg_values < self.margin]

        if neg_hard.shape[0] == 0:
            total = torch.bincount(labels)[0] + torch.bincount(labels)[1]
            weights = torch.FloatTensor([1.0 + torch.true_divide(torch.bincount(labels)[1], total),
                                         1.0 + torch.true_divide(torch.bincount(labels)[0], total)]).cuda()
            return data, labels, weights

        neg_labels = torch.zeros(neg_hard.shape, device='cuda:0')
        pos_labels = torch.ones(all_pos_values.shape, device='cuda:0')

        total = neg_labels.shape[0] + pos_labels.shape[0]
        weights = torch.FloatTensor([1.0+torch.true_divide(pos_labels.shape[0], total),
                                     1.0+torch.true_divide(neg_labels.shape[0], total)]).cuda()

        return torch.cat([neg_hard, all_pos_values]), torch.cat([neg_labels, pos_labels]), weights
