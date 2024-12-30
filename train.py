# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, progress_bar
from train_utils import *
from efficiency.function import set_seed
import torch.backends.cudnn as cudnn
from backpack import extend
import time

def train(epoch):
    set_seed(args.seed)
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(trainloader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batch_size + len(images),
            total_samples=len(trainloader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=False):
    global best_acc
    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    total = 0
    for  batch_idx, (images, labels) in enumerate(cifar100_test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum()
        progress_bar(batch_idx, len(cifar100_test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    # Save checkpoint.
    acc = 100.*correct/total
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)

    state = {
        'epoch': epoch,  # Save the next epoch number
        'net': net.state_dict(),
        'best_acc': best_acc,
        'current_acc': acc,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'is_best': is_best
    }

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, last_checkpoint_path)
    if is_best:
        torch.save(state, best_checkpoint_path)
        print(f"Updated best checkpoint: {best_checkpoint_path}")



    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    finish = time.time()
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    eval_time = finish - start
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return test_loss / len(cifar100_test_loader.dataset), acc, eval_time




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default = 'resnet18', help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--adaptive_lr',  action='store_true', help='Rescale learning rate')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--data_dir', default='../data', type=str)
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar10','cifar100', 'imagenet'],
                        help='Dataset to train on (cifar100 or imagenet)')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train')
    parser.add_argument('--algorithm', default='sgd', type=str, choices=['sgd', 'divebatch','adabatch'],)
    parser.add_argument('--batch_size', '-bs', default=128, type=int, help='Batch size')
    parser.add_argument('--resize_freq', default=10, type=int, help='Resize frequency for DiveBatch')
    parser.add_argument('--max_batch_size', default=2048, type=int, help='Maximum batch size for DiveBatch')
    parser.add_argument('--delta', default=0.1, type=float, help='Delta for GradDiversity')
    parser.add_argument('--log_dir', default='./logs', type=str, help='Directory to save logs')
    parser.add_argument('--checkpoint_dir', default='/projects/bdeb/chenyuen0103/checkpoint', type=str, help='Directory to save checkpoints')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    args = parser.parse_args()

    net = get_network(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    if device == 'cuda': 
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
                



    if args.algorithm == 'divebatch':
        def extend_all_modules(model):
            """
            Recursively extend all submodules of the model with BackPACK.
            """
            for module in model.modules():
                extend(module)
        extend_all_modules(net)






    checkpoint_dir = os.path.join(args.checkpoint_dir, args.net, args.dataset)
    # print(f"Checkpoint directory: {checkpoint_dir}")

    best_acc = 0  # best test accuracy
    start_epoch = 1  # start from epoch 0 or last checkpoint epoch


    log_dir = os.path.join(args.log_dir, args.net, args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)


    if args.algorithm == 'sgd':
        log_file = os.path.join(log_dir, f'{args.algorithm}_lr{args.lr}_bs{args.batch_size}_s{args.seed}.csv')
    elif args.algorithm == 'divebatch':
        log_file = os.path.join(log_dir, f'{args.algorithm}_lr{args.lr}_bs{args.batch_size}_rf{args.resize_freq}_mbs{args.max_batch_size}_delta{args.delta}_s{args.seed}.csv')
    elif args.algorithm == 'adabatch':
        log_file = os.path.join(log_dir, f'{args.algorithm}_lr{args.lr}_bs{args.batch_size}_rf{args.resize_freq}_mbs{args.max_batch_size}_s{args.seed}.csv')

    if args.algorithm != 'sgd' and args.adaptive_lr:
        log_file = log_file.replace('.csv', '_rescale.csv')

    fieldnames = [
        'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
        'learning_rate', 'batch_size', 'epoch_time', 'eval_time', 'abs_time', 'memory_allocated', 'memory_reserved', 'gradient_diversity'
    ]

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.net, args.dataset)
    print(f"Checkpoint directory: {checkpoint_dir}")
    if args.algorithm == 'sgd':
        checkpoint_file = f"{args.algorithm}_lr{args.lr}_bs{args.batch_size}_s{args.seed}_ckpt.pth"
    elif args.algorithm == 'divebatch':
        checkpoint_file = f"{args.algorithm}_lr{args.lr}_bs{args.batch_size}_rf{args.resize_freq}_mbs{args.max_batch_size}_delta{args.delta}_s{args.seed}_ckpt.pth"
    elif args.algorithm == 'adabatch':
        checkpoint_file = f"{args.algorithm}_lr{args.lr}_bs{args.batch_size}_rf{args.resize_freq}_mbs{args.max_batch_size}_s{args.seed}_ckpt.pth"
    # Check if the log file already exists


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    batch_size = args.batch_size
    best_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file.replace('.pth', '_best.pth'))
    last_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    if args.algorithm != 'sgd' and args.adaptive_lr:
        last_checkpoint_path = last_checkpoint_path.replace('.pth', '_rescale.pth')
        best_checkpoint_path = best_checkpoint_path.replace('.pth', '_rescale.pth')
        log_file = log_file.replace('.csv', '_rescale.csv')
        
    log_exists = os.path.exists(log_file)
    # Resume from checkpoint
    file_mode = 'w'
    if args.resume:
        if os.path.exists(last_checkpoint_path):
            checkpoint = torch.load(last_checkpoint_path)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            file_mode = 'a'
            row = None
            # load the batch size from the csv file
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pass
                batch_size = int(row['batch_size'])

        #data preprocessing:
    trainloader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=1,
        batch_size=batch_size,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=1,
        batch_size=batch_size,
        shuffle=True
    )
    iter_per_epoch = len(trainloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)


    with open(log_file, file_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # Write the header only if the file is new
        if not log_exists or not args.resume:
            writer.writeheader()

    # if args.resume:
    #     recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
    #     if not recent_folder:
    #         raise Exception('no recent folder were found')

    #     checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    # else:
    #     checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    # #so the only way is to create a new tensorboard log
    # writer = SummaryWriter(log_dir=os.path.join(
    #         settings.LOG_DIR, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(1, 3, 32, 32)
    # if args.gpu:
    #     input_tensor = input_tensor.cuda()
    # writer.add_graph(net, input_tensor)

    # #create checkpoint folder to save model
    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)
    # checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    # best_acc = 0.0
    # if args.resume:
    #     best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
    #     if best_weights:
    #         weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
    #         print('found best acc weights file:{}'.format(weights_path))
    #         print('load best training file to test acc...')
    #         net.load_state_dict(torch.load(weights_path))
    #         best_acc = eval_training(tb=False)
    #         print('best acc is {:0.2f}'.format(best_acc))

    #     recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
    #     if not recent_weights_file:
    #         raise Exception('no recent weights file were found')
    #     weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
    #     print('loading weights file {} to resume training.....'.format(weights_path))
    #     net.load_state_dict(torch.load(weights_path))

    #     resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))



    if not os.path.exists(log_file) or file_mode == 'w':
        with open(log_file, file_mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    # Training and Testing
    trainer_cls = SGDTrainer if args.algorithm == 'sgd' else DiveBatchTrainer
    trainer_args = {
        "model": net,
        "optimizer": optimizer,
        "criterion": criterion,
        "device": device,
    }
    if args.algorithm == 'divebatch':
        trainer_args.update({
            "resize_freq": args.resize_freq,
            "max_batch_size": args.max_batch_size,
        })
    trainer = trainer_cls(**trainer_args)
    old_grad_diversity = 1.0 if args.algorithm == 'divebatch' else None


    batch_size = args.batch_size
    abs_start_time = time.time()
    for epoch in range(1, settings.EPOCH + 1):
        if args.resume:
            if epoch <= start_epoch:
                continue

        # train(epoch)
        train_metrics = trainer.train_epoch(trainloader, epoch)
        val_loss, val_acc, eval_time = eval_training(epoch)


        log_metrics(
            log_file=log_file,
            epoch=epoch,
            train_loss=train_metrics["train_loss"],
            train_acc=train_metrics["train_accuracy"],
            val_loss=val_loss,
            val_acc=val_acc,
            lr=optimizer.param_groups[0]['lr'],
            batch_size=batch_size,
            epoch_time=train_metrics.get("epoch_time", 0),
            eval_time=eval_time,
            abs_time = time.time() - abs_start_time,
            memory_allocated=torch.cuda.memory_allocated() if device == 'cuda' else 0,
            memory_reserved=torch.cuda.memory_reserved() if device == 'cuda' else 0,
            grad_diversity=train_metrics.get("grad_diversity")
        )

        #start to save best performance model after learning rate decay to 0.01

        if epoch > settings.MILESTONES[1] and best_acc < val_acc:
            weights_path = best_checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = val_acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = last_checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
        if epoch >= args.warm:
            scheduler.step(epoch)
        elif epoch < args.warm:
            warmup_scheduler.step()
        if epoch % args.resize_freq == 0 and batch_size < args.max_batch_size and args.algorithm in ['divebatch', 'adabatch']:
            old_batch_size = batch_size
            if args.algorithm == 'divebatch':
                grad_diversity = train_metrics.get("grad_diversity")
                rescale_ratio = max((grad_diversity / old_grad_diversity),1)
            elif args.algorithm == 'adabatch':
                rescale_ration = 2

            batch_size = int(min(old_batch_size * rescale_ratio, args.max_batch_size))
            
            if batch_size != old_batch_size:
                # Update the batch size argument
                # batch_size = new_batch_size
                # print(f"Updating DataLoader with new batch size: {batch_size}")
                # trainer.accum_steps = new_batch_size // args.batch_size
                # Recreate DataLoader with the new batch size
                print(f"Recreating trainloader with batch size: {batch_size}...")
                trainloader = torch.utils.data.DataLoader(
                    trainloader.dataset,
                    batch_size=batch_size,
                    shuffle=True, 
                    num_workers=1, 
                    pin_memory=True
                )

            if args.adaptive_lr:
                # rescale the learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= rescale_ratio

f.close()



