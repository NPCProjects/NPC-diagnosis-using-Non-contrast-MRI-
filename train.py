# coding=utf-8
import argparse
import os
import time
import logging
import setproctitle

import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from data.transforms import *
from data.datasets_nii import NPC_loadall_nii
from data.data_utils import init_fn
from model.net import Model
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from datetime import datetime
from predict import evaluate_classification
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--datapath', default=r'path_to_training_data', type=str)
parser.add_argument('--dataname', default='NPCCLASS', type=str)
parser.add_argument('--user', default='user', type=str)
parser.add_argument('--savepath', default=r'model_output_path', type=str)
parser.add_argument('--resume', default=r'path_to_resume_checkpoint', type=str)

parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--gpu', default='3,4', type=str)
parser.add_argument('--load', default=False, type=bool)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)
model_output_path = os.path.join(args.savepath, f'try_{datetime.now().strftime("%Y-%m-%d %H%M")}')

###tensorboard writer
if args.load:
    writer = SummaryWriter(os.path.join(args.savepath, args.resume))
else:
    writer = SummaryWriter(model_output_path)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


    args.dataname = 'NPCCLASS'
    num_cls = 2
    model = Model(num_cls=2)
    model = torch.nn.DataParallel(model).cuda() # multiple GPUs

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    if args.load and os.path.isfile(args.resume):
        logging.info('loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info('successfully loading checkpoint {} and training from epoch:{}'.format(args.resume, args.start_epoch))

    else:
        logging.info('re-training!')



    logging.info(str(args))


    if os.path.isfile(args.resume) and args.load:
        logging.info('loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optim_dict'])
        logging.info('successfully loading checkpoint {} and training from epoch:{}'.format(args.resume, args.start_epoch))

    else:
        logging.info('re-training!')
    train_set = NPC_loadall_nii(transforms=args.train_transforms, root=f'{args.datapath}/Train', num_cls=num_cls)
    valid_set = NPC_loadall_nii(transforms=args.test_transforms, root=f'{args.datapath}/Valid', num_cls=num_cls)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=32,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        worker_init_fn=init_fn)
    valid_loader = MultiEpochsDataLoader(
        dataset=valid_set,
        batch_size=args.batch_size,
        num_workers=32,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        worker_init_fn=init_fn)

    torch.set_default_dtype(torch.float32)
    ##########Training
    start = time.time() # records time
    torch.set_grad_enabled(True)
    logging.info('training!!!!!')
    iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)
    ce_criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.start_epoch, args.num_epochs):
        loss_epoch = []
        fuse_loss_epoch = []
        all_modality_loss_epoch = []
        KL_epoch = []
        setproctitle.setproctitle('{}'.format(args.user)) # Set process title with user name for the current epoch
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('epoch/lr', step_lr, global_step=(epoch + 1))
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i + 1) + epoch * iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True).float()
            target = target.cuda(non_blocking=True).float()
            target = target.squeeze().type(torch.int64)
            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True
            fuse_pred, pred_all, t1ce, t1, t2, all, aux, output_features = model(x, mask)

            ###Loss compute
            fuse_loss = ce_criterion(fuse_pred, target)
            all_modality_loss = ce_criterion(pred_all, target)
            KL_t1ce = F.kl_div(F.log_softmax(aux[0], dim=1), F.softmax(pred_all, dim=1), reduction='batchmean')
            KL_t1 = F.kl_div(F.log_softmax(aux[1], dim=1), F.softmax(pred_all, dim=1), reduction='batchmean')
            KL_t2 = F.kl_div(F.log_softmax(aux[2], dim=1), F.softmax(pred_all, dim=1), reduction='batchmean')
            KL = KL_t1ce + KL_t1 + KL_t2

            loss = fuse_loss + all_modality_loss + KL

            loss_epoch.append(loss.item())
            fuse_loss_epoch.append(fuse_loss.item())
            all_modality_loss_epoch.append(all_modality_loss.item())
            KL_epoch.append(KL.item())

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            ###log
            writer.add_scalar('step/loss', loss.item(), global_step=step)
            writer.add_scalar('step/fuse_loss', fuse_loss.item(), global_step=step)
            writer.add_scalar('step/all_modality_loss', all_modality_loss.item(), global_step=step)
            writer.add_scalar('step/KLloss', KL.item(), global_step=step)
            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch + 1), args.num_epochs, (i + 1), iter_per_epoch,
                                                                  loss.item())
            msg += 'fuse:{:.4f}, '.format(fuse_loss.item())
            msg += 'allmodality:{:.4f}, '.format(all_modality_loss.item())
            msg += 'KLloss:{:.4f}, '.format(KL.item())
            logging.info(msg)
        writer.add_scalar('epoch/loss', sum(loss_epoch) / len(train_loader), global_step=epoch)
        writer.add_scalar('epoch/fuse_loss', sum(fuse_loss_epoch) / len(train_loader), global_step=epoch)
        writer.add_scalar('epoch/all_modality_loss', sum(all_modality_loss_epoch) / len(train_loader), global_step=epoch)
        writer.add_scalar('epoch/KL', sum(KL_epoch) / len(train_loader), global_step=epoch)

        print('Prediction for training')
        accuracy, precision, recall, f1, _, _, _ = evaluate_classification(train_loader, model, None, None, None)

        writer.add_scalar('train/accuracy', accuracy, global_step=epoch)
        accuracy, precision, recall, f1, _, _, _ = evaluate_classification(valid_loader, model, None, None, None)
        writer.add_scalar('valid/accuracy', accuracy, global_step=epoch)
        model.module.is_training = True
        model.train()
        logging.info('train time per epoch: {}'.format(time.time() - b))


        ##########model save
        file_name = os.path.join(model_output_path, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            file_name)

        if (epoch + 1) % 50 == 0 or (epoch >= (args.num_epochs - 10)):
            file_name = os.path.join(model_output_path, 'model_{}.pth'.format(epoch + 1))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

    msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
    logging.info(msg)


if __name__ == '__main__':
    main()
