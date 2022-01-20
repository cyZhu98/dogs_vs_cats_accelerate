import os
import time
import numpy as np
import random
import warnings
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import OneCycleLR
import torch.backends.cudnn as cudnn
from accelerate import Accelerator
import timm
from transformers import get_linear_schedule_with_warmup, set_seed

from loader import getLoader
from config import return_args
from utils import *

best_acc = 0


def main():
    args = return_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    main_worker(args)


def main_worker(args):
    global best_acc
    accelerator = Accelerator()

    train_loader = getLoader(args, 'train')
    val_loader = getLoader(args, 'val')

    model = timm.create_model(
        'swin_base_patch4_window7_224', pretrained=True, num_classes=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    lr_scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    for epoch in tqdm(range(args.epochs), disable=not accelerator.is_local_main_process):
        train_one_epoch(train_loader, model, criterion, optimizer,
                        lr_scheduler, accelerator, epoch, args)
        acc = validate(val_loader, model, criterion,
                       optimizer, accelerator, args)
        if acc > best_acc:
            best_acc = acc
            if args.save:
                if not os.path.exists('save'):
                    os.mkdir('save')
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), os.path.join(
                    'save', f'ep{epoch}_acc{acc*1000:.0f}.pth'))


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, accelerator, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        img, label = data
        output = model(img)
        loss = criterion(output, label)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        acc1 = accuracy(output, label, topk=(1, ))
        losses.update(loss.item(), img.size(0))
        top1.update(acc1[0].item(), img.size(0))
        # top5.update(acc5[0], img.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, accelerator)


@torch.no_grad()
def validate(val_loader, model, criterion, optimizer, accelerator, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    model.eval()
    end = time.time()
    for i, data in enumerate(val_loader):
        img, label = data
        output = model(img)
        loss = criterion(output, label)

        acc1 = accuracy(output, label, topk=(1, ))
        losses.update(loss.item(), img.size(0))
        top1.update(acc1[0].item(), img.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, accelerator)

    progress.display_summary(accelerator)

    return top1.avg


if __name__ == '__main__':
    main()
