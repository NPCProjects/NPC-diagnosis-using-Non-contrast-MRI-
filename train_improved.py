import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict
import setproctitle
from datetime import datetime

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from model.net import Model
from utils import Parser, criterions
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from predict import AverageMeter, test_softmax
from predict import evaluate_classification


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def load_checkpoint_if_needed(args, model, optimizer):
    """Load a checkpoint if the resume flag is set."""
    if args.load and os.path.isfile(args.resume):
        logging.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        logging.info(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    else:
        logging.info("No checkpoint found, starting training from scratch.")
        return 0


def create_data_loaders(args, num_cls=2):
    """Create training, validation and test data loaders."""
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls)
    valid_set = Brats_loadall_nii(transforms=args.test_transforms, root=r'/Dataset/NPC/processed/Valid',
                                  num_cls=num_cls)

    train_loader = MultiEpochsDataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=16, pin_memory=True,
                                         shuffle=True, worker_init_fn=init_fn)
    valid_loader = MultiEpochsDataLoader(dataset=valid_set, batch_size=args.batch_size, num_workers=16, pin_memory=True,
                                         shuffle=True, worker_init_fn=init_fn)

    return train_loader, valid_loader


def setup_model_and_optimizer(args, model):
    """Setup model, optimizer and learning rate scheduler."""
    model = torch.nn.DataParallel(model).cuda()  # Support multiple GPUs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
                                 eps=1e-08, amsgrad=True)
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)

    return model, optimizer, lr_schedule


def setup_tensorboard_writer(args):
    """Setup tensorboard writer."""
    if args.load:
        return SummaryWriter(os.path.join(args.savepath, '2024-06-29 1506'))
    else:
        return SummaryWriter(os.path.join(args.savepath, datetime.now().strftime('%Y-%m-%d %H%M')))


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', '--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--datapath', default=r'/Dataset/NPC/processed/Train', type=str)
    parser.add_argument('--dataname', default='NPCCLASS', type=str)
    parser.add_argument('--user', default='685', type=str)
    parser.add_argument('--savepath', default=r'/Dataset/NPC/Model_Train_Results', type=str)
    parser.add_argument('--resume', default='/Dataset/NPC/Model_Train_Results/model_320.pth', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--seed', default=1024, type=int)
    parser.add_argument('--gpu', default='2', type=str)
    parser.add_argument('--load', default=False, type=bool)
    args = parser.parse_args()
    setup(args, 'training')

    setup_logging()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # Setup model
    num_cls = 2
    model = Model(num_cls=num_cls)
    model, optimizer, lr_schedule = setup_model_and_optimizer(args, model)

    # Load checkpoint if needed
    start_epoch = load_checkpoint_if_needed(args, model, optimizer)

    # Setup data loaders
    train_loader, valid_loader = create_data_loaders(args, num_cls)

    # Setup tensorboard writer
    writer = setup_tensorboard_writer(args)

    # Training loop
    start_time = time.time()
    criterion = torch.nn.BCEWithLogitsLoss()
    ce_criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.num_epochs):
        loss_epoch = []
        model.train()
        step_lr = lr_schedule(optimizer, epoch)

        # Log learning rate
        writer.add_scalar('epoch/lr', step_lr, global_step=epoch)

        for i, data in enumerate(train_loader):
            step = i + epoch * len(train_loader)

            x, target, mask = data[:3]
            x, target, mask = x.cuda(), target.cuda(), mask.cuda()

            model.module.is_training = True
            fuse_pred, pred_all, t1ce, t1, t2, all, aux, output_features = model(x, mask)

            # Compute losses
            fuse_loss = ce_criterion(fuse_pred, target)
            all_modality_loss = ce_criterion(pred_all, target)

            loss_t1ce = torch.zeros(1).cuda()
            loss_t1 = torch.zeros(1).cuda()
            loss_t2 = torch.zeros(1).cuda()
            for stage in range(3):
                loss_t1ce += criterions.l1loss(all[stage + 2], t1ce[stage + 2])
                loss_t1 += criterions.l1loss(all[stage + 2], t1[stage + 2])
                loss_t2 += criterions.l1loss(all[stage + 2], t2[stage + 2])

            KL_t1ce = F.kl_div(F.log_softmax(aux[0], dim=1), F.softmax(pred_all, dim=1), reduction='batchmean')
            KL_t1 = F.kl_div(F.log_softmax(aux[1], dim=1), F.softmax(pred_all, dim=1), reduction='batchmean')
            KL_t2 = F.kl_div(F.log_softmax(aux[2], dim=1), F.softmax(pred_all, dim=1), reduction='batchmean')
            KL = KL_t1ce + KL_t1 + KL_t2

            loss = fuse_loss + all_modality_loss + KL

            # Backprop and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log loss values
            writer.add_scalar('step/loss', loss.item(), global_step=step)
            writer.add_scalar('step/fuse_loss', fuse_loss.item(), global_step=step)
            writer.add_scalar('step/all_modality_loss', all_modality_loss.item(), global_step=step)

        # Log epoch-wise loss
        writer.add_scalar('epoch/loss', np.mean(loss_epoch), global_step=epoch)

        # Evaluate model
        accuracy, precision, recall, f1 = evaluate_classification(train_loader, model)
        writer.add_scalar('train/accuracy', accuracy, global_step=epoch)

        accuracy, precision, recall, f1 = evaluate_classification(valid_loader, model)
        writer.add_scalar('valid/accuracy', accuracy, global_step=epoch)

        # Save model checkpoint
        if (epoch + 1) % 20 == 0 or (epoch >= (args.num_epochs - 10)):
            checkpoint_path = os.path.join(args.savepath, f"model_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            }, checkpoint_path)

    total_time = (time.time() - start_time) / 3600
    logging.info(f"Training completed in {total_time:.4f} hours")


if __name__ == '__main__':
    main()
