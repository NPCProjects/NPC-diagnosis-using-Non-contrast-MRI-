import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

__all__ = ['sigmoid_dice_loss', 'softmax_dice_loss', 'GeneralizedDiceLoss', 'FocalLoss', 'dice_loss']


def dice_loss(output, target, num_cls=5, eps=1e-7):
    target = target.float()
    dice = 0.0
    for i in range(num_cls):
        num = torch.sum(output[:, i, ...] * target[:, i, ...])
        denom = torch.sum(output[:, i, ...]) + torch.sum(target[:, i, ...]) + eps
        dice += 2.0 * num / denom
    return 1.0 - dice / num_cls


def softmax_weighted_loss(output, target, num_cls=5):
    target = target.float()
    B, _, H, W, Z = output.size()
    cross_loss = 0.0
    for i in range(num_cls):
        output_i = output[:, i, :, :, :]
        target_i = target[:, i, :, :, :]
        weighted = 1.0 - torch.sum(target_i, (1, 2, 3)) / torch.sum(target, (1, 2, 3, 4))
        weighted = weighted.view(-1, 1, 1, 1).repeat(1, H, W, Z)
        cross_loss += -weighted * target_i * torch.log(torch.clamp(output_i, min=0.005, max=1))
    return torch.mean(cross_loss)


def bceloss(inputs, targets):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(inputs, targets)


def l1loss(inputs, targets):
    criterion = nn.L1Loss()
    return criterion(inputs, targets)


def softmax_loss(output, target, num_cls=5):
    target = target.float()
    cross_loss = 0.0
    for i in range(num_cls):
        output_i = output[:, i, :, :, :]
        target_i = target[:, i, :, :, :]
        cross_loss += -target_i * torch.log(torch.clamp(output_i, min=0.005, max=1))
    return torch.mean(cross_loss)


def FocalLoss(output, target, alpha=0.25, gamma=2.0):
    target[target == 4] = 3  # Merge class 4 into class 3
    output = output.view(output.size(0), output.size(1), -1).transpose(1, 2).contiguous().view(-1, output.size(2))
    target = target.view(-1)

    logpt = -F.cross_entropy(output, target, reduction='none')
    pt = torch.exp(logpt)
    loss = -((1 - pt) ** gamma) * logpt
    return loss.mean()


def dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num / den


def sigmoid_dice_loss(output, target, alpha=1e-5):
    loss1 = dice(output[:, 0, ...], (target == 1).float(), eps=alpha)
    loss2 = dice(output[:, 1, ...], (target == 2).float(), eps=alpha)
    loss3 = dice(output[:, 2, ...], (target == 4).float(), eps=alpha)
    logging.info(f'1:{1 - loss1.item():.4f} | 2:{1 - loss2.item():.4f} | 4:{1 - loss3.item():.4f}')
    return loss1 + loss2 + loss3


def softmax_dice_loss(output, target, eps=1e-5):
    loss1 = dice(output[:, 1, ...], (target == 1).float(), eps=eps)
    loss2 = dice(output[:, 2, ...], (target == 2).float(), eps=eps)
    loss3 = dice(output[:, 3, ...], (target == 4).float(), eps=eps)
    logging.info(f'1:{1 - loss1.item():.4f} | 2:{1 - loss2.item():.4f} | 4:{1 - loss3.item():.4f}')
    return loss1 + loss2 + loss3


def GeneralizedDiceLoss(output, target, eps=1e-5, weight_type='square'):
    """
    Generalized Dice Loss: A deep learning loss function for highly unbalanced segmentations.
    Weighting types available: 'square', 'identity', and 'sqrt'.
    """
    if target.dim() == 4:
        target[target == 4] = 3  # Merge class 4 into class 3
        target = expand_target(target, n_class=output.size(1))  # Expand target to match output dimensions

    output = flatten(output)[1:, ...]  # Flatten the output
    target = flatten(target)[1:, ...]  # Flatten the target

    target_sum = target.sum(-1)
    class_weights = _get_class_weights(target_sum, eps, weight_type)

    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2 * intersect[0] / (denominator[0] + eps)
    loss2 = 2 * intersect[1] / (denominator[1] + eps)
    loss3 = 2 * intersect[2] / (denominator[2] + eps)

    return 1 - 2.0 * intersect_sum / denominator_sum, [loss1.item(), loss2.item(), loss3.item()]


def _get_class_weights(target_sum, eps, weight_type):
    if weight_type == 'square':
        return 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        return 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        return 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError(f'Unsupported weight_type: {weight_type}')


def expand_target(x, n_class, mode='softmax'):
    """Converts label image to NxCxDxHxW format for multi-class segmentation."""
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    xx = torch.zeros(tuple(shape))

    if mode.lower() == 'softmax':
        for i in range(1, n_class):
            xx[:, i, ...] = (x == i)
    elif mode.lower() == 'sigmoid':
        for i in range(n_class):
            xx[:, i, ...] = (x == i)

    return xx.to(x.device)


def flatten(tensor):
    """Flattens a tensor to shape [C, N*H*W*D]."""
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.reshape(C, -1)
