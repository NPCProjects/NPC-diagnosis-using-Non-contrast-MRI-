import math
import numpy as np
import torch
import torch.nn.functional as F


class LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, mode='poly'):
        """
        Learning rate scheduler.
        Args:
            base_lr (float): Initial learning rate.
            num_epochs (int): Number of total epochs.
            mode (str): Scheduling mode ('poly' for polynomial decay).
        """
        self.mode = mode
        self.lr = base_lr
        self.num_epochs = num_epochs

    def __call__(self, optimizer, epoch):
        """
        Update the learning rate for the current epoch.
        Args:
            optimizer (torch.optim.Optimizer): The optimizer whose learning rate needs to be updated.
            epoch (int): The current epoch number.
        """
        if self.mode == 'poly':
            # Polynomial decay
            now_lr = self.lr * (1 - epoch / self.num_epochs) ** 0.9
        else:
            raise ValueError(f"Unsupported LR scheduler mode: {self.mode}")
        self._adjust_learning_rate(optimizer, now_lr)
        return now_lr

    def _adjust_learning_rate(self, optimizer, lr):
        """
        Adjust the learning rate for all parameter groups in the optimizer.
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def get_temperature(epoch):
    """Get dynamic temperature based on the epoch."""
    return 31 - (epoch + 1) if epoch <= 29 else 1


def get_params(model):
    """
    Get base and ignored parameters from the model.
    Args:
        model (torch.nn.Module): The model from which to extract parameters.
    Returns:
        base_params: Parameters that are not in the ignored layers.
        ignore_params: Parameters that belong to the ignored layers.
    """
    # Extract parameters from specific layers that need to be ignored
    ignore_layers = [model.module.decoder_all.abstraction1.fusion_conv.attention,
                     model.module.decoder_all.abstraction2.fusion_conv.attention,
                     model.module.decoder_all.abstraction3.fusion_conv.attention,
                     model.module.decoder_all.abstraction4.fusion_conv.attention]

    ignore_id = [id(param) for layer in ignore_layers for param in layer.parameters()]

    ignore_params = filter(lambda p: id(p) in ignore_id, model.parameters())
    base_params = filter(lambda p: id(p) not in ignore_id, model.parameters())

    return base_params, ignore_params


def record_loss(args, writer, mask1, loss_list, loss_name, step, mask_list, name_list, p_type):
    """
    Log the loss values for different masks and classes.
    Args:
        args: Arguments for training.
        writer: TensorBoard writer.
        mask1: The masks for which to log the loss.
        loss_list: The list of loss values.
        loss_name: The names of the losses.
        step: The current training step.
        mask_list: List of masks to match.
        name_list: Names of the classes.
        p_type: Types of the loss (e.g., 'train', 'val').
    """
    for i in range(mask1.size(0)):
        for j, mask in enumerate(mask_list):
            if torch.equal(mask1[i].int(), mask.int()):
                for k, loss in enumerate(loss_list):
                    writer.add_scalar(f"{p_type[i]}_{name_list[j]}_{loss_name[k]}", loss[i].item(), global_step=step)


def Js_div(feat1, feat2, KLDivLoss):
    """
    Compute the Jensen-Shannon divergence between two features.
    Args:
        feat1: The first feature.
        feat2: The second feature.
        KLDivLoss: The KL divergence loss function.
    """
    log_pq = torch.log((feat1 + feat2) / 2)
    return (KLDivLoss(log_pq, feat1) + KLDivLoss(log_pq, feat2)) / 2


def mutual_learning_loss(mutual_feat, mask, KLDivLoss):
    """
    Compute the mutual learning loss for multi-modal features.
    Args:
        mutual_feat: The features to compare.
        mask: The mask to apply.
        KLDivLoss: The KL divergence loss function.
    """
    mutual_loss = torch.zeros(mask.size(0)).cuda()

    for i in range(mask.size(0)):
        K = torch.sum(mask[i])
        if K == 1:
            continue
        for j in range(4):
            feat = mutual_feat[j][:, mask[i], :, :, :, :]
            feat = F.softmax(feat, dim=2)
            for k in range(K):
                for k1 in range(k + 1, K):
                    mutual_loss[i] += Js_div(feat[:, k, :, :, :, :], feat[:, k1, :, :, :, :], KLDivLoss)
        mutual_loss[i] = mutual_loss[i] / (2 * K * (K - 1))

    return mutual_loss


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """
    Custom DataLoader that supports multi-epoch data sampling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """
    A sampler that repeats the dataset indefinitely.
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
