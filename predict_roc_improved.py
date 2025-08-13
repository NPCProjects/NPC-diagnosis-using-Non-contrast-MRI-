import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import nibabel as nib
import scipy.misc
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from imblearn.metrics import sensitivity_score, specificity_score
from utils import criterions
import tqdm

cudnn.benchmark = True
path = os.path.dirname(__file__)
from utils.generate import generate_snapshot

patch_size = 128

def softmax_output_dice_class(output, target, num_classes):
    eps = 1e-8
    dice_scores = []

    for i in range(1, num_classes + 1):
        o = (output == i).float()
        t = (target == i).float()
        intersect = torch.sum(2 * (o * t), dim=(1, 2, 3)) + eps
        denominator = torch.sum(o, dim=(1, 2, 3)) + torch.sum(t, dim=(1, 2, 3)) + eps
        dice = intersect / denominator
        dice_scores.append(dice)

    # Post-processing for enhancing (if necessary)
    if num_classes > 3 and torch.sum(o) < 500:  # for class 4, enhancing
        o4 = o * 0.0
    else:
        o4 = o
    t4 = t
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1, 2, 3)) + eps
    denominator4 = torch.sum(o4, dim=(1, 2, 3)) + torch.sum(t4, dim=(1, 2, 3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    dice_scores.append(enhancing_dice_postpro)

    return torch.cat(dice_scores, dim=1).cpu().numpy()

def evaluate_classification(test_loader, model, feature_mask=None):
    model.is_training = False
    model.eval()
    vals_evaluation = AverageMeter()
    preds, targets = [], []
    names = []

    for data in tqdm.tqdm(test_loader):
        target = data[1]
        x = data[0].cuda()
        name = data[-1]
        names.extend(name)

        mask = torch.from_numpy(np.array(feature_mask)) if feature_mask is not None else data[2]
        mask = mask.cuda()

        pred, _, _, _, _, _, _, _ = model(x, mask)
        pred = torch.softmax(pred, dim=1)[0][1]

        preds.append(pred)
        targets.extend(target.tolist())

    for k, name in enumerate(names):
        msg = f'Subject {k+1}/{len(names)}, {name:>20}, True Label {targets[k]}, Prediction {preds[k]}'
        logging.info(msg)

    return

def test_softmax(test_loader, model, dataname='NPCCLASS', feature_mask=None):
    model.eval()
    vals_evaluation = AverageMeter()
    one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().cuda()

    # Set num_cls based on the dataset name
    num_cls = 2
    class_evaluation = ['whole', 'core', 'enhancing', 'enhancing_postpro'] if num_cls == 4 else ['necrosis', 'edema', 'non_enhancing', 'enhancing']

    for i, data in enumerate(test_loader):
        target = data[1].cuda()
        x = data[0].cuda()
        names = data[-1]

        mask = torch.from_numpy(np.array(feature_mask)) if feature_mask is not None else data[2]
        mask = mask.cuda()

        H, W, Z = x.size(2), x.size(3), x.size(4)

        # Sliding window setup
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        h_idx_list = range(0, int(np.ceil((H - patch_size) / (patch_size * (1 - 0.5))))) + [H - patch_size]
        w_idx_list = range(0, int(np.ceil((W - patch_size) / (patch_size * (1 - 0.5))))) + [W - patch_size]
        z_idx_list = range(0, int(np.ceil((Z - patch_size) / (patch_size * (1 - 0.5))))) + [Z - patch_size]

        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size]
                    pred_part = model(x_input, mask)[0]
                    pred[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += pred_part

        pred /= weight
        pred = pred[:, :, :H, :W, :Z]
        pred = torch.argmax(pred, dim=1)


        sensitivity, specificity, accuracy = softmax_output_dice_class(pred, target, 2)

        for k, name in enumerate(names):
            msg = f'Subject {i+1}/{len(test_loader)}, {k+1}/{len(names)}: {name:>20}, True Label {targets[k]}, Prediction {preds[k]}'
            logging.info(msg)

    return vals_evaluation.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
