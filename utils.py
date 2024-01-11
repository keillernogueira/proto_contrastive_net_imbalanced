import os
import argparse
import math

import numpy as np

import torch
from torch import nn

from config import NUM_WORKERS


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def str2bool(v):
    """
    Function to transform strings into booleans.

    v: string variable
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_best_models(net, optimizer, output_path, best_records, epoch, metric, num_saves=3, track_mean=None):
    if math.isnan(metric):
        metric = 0.0
    if len(best_records) < num_saves:
        best_records.append({'epoch': epoch, 'kappa': metric, 'track_mean': track_mean})

        torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
    else:
        # find min saved acc
        min_index = 0
        for i, r in enumerate(best_records):
            if best_records[min_index]['kappa'] > best_records[i]['kappa']:
                min_index = i

        # check if currect acc is greater than min saved acc
        if metric > best_records[min_index]['kappa']:
            # if it is, delete previous files
            min_step = str(best_records[min_index]['epoch'])

            os.remove(os.path.join(output_path, 'model_' + min_step + '.pth'))

            # replace min value with current
            best_records[min_index] = {'epoch': epoch, 'kappa': metric, 'track_mean': track_mean}

            # save current model
            torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
    np.save(os.path.join(output_path, 'best_records.npy'), best_records)


def kappa_with_cm(conf_matrix):
    acc = 0
    marginal = 0
    total = float(np.sum(conf_matrix))
    for i in range(len(conf_matrix)):
        acc += conf_matrix[i][i]
        marginal += np.sum(conf_matrix, 0)[i] * np.sum(conf_matrix, 1)[i]

    kappa = (total * acc - marginal) / (total * total - marginal)
    return kappa


def f1_with_cm(conf_matrix):
    precision = [0] * len(conf_matrix)
    recall = [0] * len(conf_matrix)
    f1 = [0] * len(conf_matrix)
    for i in range(len(conf_matrix)):
        precision[i] = conf_matrix[i][i] / float(np.sum(conf_matrix, 0)[i])
        recall[i] = conf_matrix[i][i] / float(np.sum(conf_matrix, 1)[i])
        f1[i] = 2 * ((precision[i]*recall[i])/(precision[i]+recall[i]))

    return np.mean(f1)


def jaccard_with_cm(conf_matrix):
    den = float(np.sum(conf_matrix[:, 1]) + np.sum(conf_matrix[1]) - conf_matrix[1][1])
    _sum_iou = conf_matrix[1][1] / den if den != 0 else 0

    return _sum_iou


def sample_weight_train_loader(train_dataset, gen_classes, batch_size):
    class_loader_weights = 1. / np.bincount(gen_classes)
    samples_weights = class_loader_weights[gen_classes]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   num_workers=NUM_WORKERS, drop_last=False, sampler=sampler)
    return train_dataloader
