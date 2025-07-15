# coding: utf-8
import os
import time
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from lr_scheduler import *
from model import *
from audio_dataset import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 5
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def data_loader(args):
    dsets = {x: MyDataset(x, args.dataset) for x in ['train', 'val', 'test']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.workers, generator=torch.Generator(device='cuda')) for x in ['train', 'val', 'test']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
    print('\nStatistics: train: {}, val: {}, test: {}'.format(dset_sizes['train'], dset_sizes['val'], dset_sizes['test']))
    return dset_loaders, dset_sizes


def reload_model(model, logger, path=""):
    if not bool(path):
        logger.info('train from scratch')
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('*** model has been successfully loaded! ***')
        return model


def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def train_test(model, dset_loaders, criterion, epoch, phase, optimizer, args, logger, use_gpu, save_path):

    if phase == 'val' or phase == 'test':
        model.eval()
    if phase == 'train':
        model.train()
    if phase == 'train':
        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
        logger.info('Current Learning rate: {}'.format(showLR(optimizer)))

    running_loss, running_corrects, running_all = 0., 0., 0.

    if phase == 'train':
        for batch_idx, (inputs, targets) in enumerate(dset_loaders[phase]):
            inputs = inputs.float()
            inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            if args.every_frame:
                outputs = torch.mean(outputs, 1)
            _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data * inputs.size(0)
            for i in range(preds.size(0)):
                if preds[i] == targets.data[i]:
                    running_corrects += 1
            running_all += len(inputs)
            if batch_idx == 0:
                since = time.time()
            elif batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase])-1):
                print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tAcc:{:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                    running_all,
                    len(dset_loaders[phase].dataset),
                    100. * batch_idx / (len(dset_loaders[phase])-1),
                    running_loss / running_all,
                    running_corrects / running_all,
                    time.time()-since,
                    (time.time()-since)*(len(dset_loaders[phase])-1) / batch_idx - (time.time()-since))),
        print
        logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}'.format(
            phase,
            epoch,
            running_loss / len(dset_loaders[phase].dataset),
            running_corrects / len(dset_loaders[phase].dataset))+'\n')
        torch.save(model.state_dict(), save_path+'/'+args.mode+'_'+ 'audio_' +str(epoch+1)+'.pt')
        return model
    if phase == 'val' or phase == 'test':
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dset_loaders[phase]):
                inputs = inputs.float()
                inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                if args.every_frame:
                    outputs = torch.mean(outputs, 1)
                _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)
                loss = criterion(outputs, targets)
                running_loss += loss.data * inputs.size(0)
                for i in range(preds.size(0)):
                    if preds[i] == targets.data[i]:
                        running_corrects += 1
                running_all += len(inputs)
                if batch_idx == 0:
                    since = time.time()
                elif batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase])-1):
                    print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tAcc:{:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                        running_all,
                        len(dset_loaders[phase].dataset),
                        100. * batch_idx / (len(dset_loaders[phase])-1),
                        running_loss / running_all,
                        running_corrects / running_all,
                        time.time()-since,
                        (time.time()-since)*(len(dset_loaders[phase])-1) / batch_idx - (time.time()-since))),
            print
            logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}'.format(
                phase,
                epoch,
                running_loss / len(dset_loaders[phase].dataset),
                running_corrects / len(dset_loaders[phase].dataset))+'\n')


def test_adam(args, use_gpu):
    if args.every_frame and args.mode != 'temporalConv':
        save_path = './' + args.mode + '_every_frame'
    elif not args.every_frame and args.mode != 'temporalConv':
        save_path = './' + args.mode + '_last_frame'
    elif args.mode == 'temporalConv':
        save_path = './' + args.mode
    else:
        raise Exception('No model is found!')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # logging info
    filename = save_path+'/'+args.mode+'_'+ 'audio_' +str(args.lr)+'.txt'
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    model = audio_model(mode=args.mode, inputDim=512, hiddenDim=512, nClasses=args.nClasses, frameLen=60, every_frame=args.every_frame)
    # reload model
    model = reload_model(model, logger, args.path)
    model = model.to(device)
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0005)
    if args.mode == 'temporalConv' or args.mode == 'finetuneGRU':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)
    elif args.mode == 'backendGRU':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.gru.parameters():
            param.requires_grad = True
        optimizer = optim.Adam([
            {'params': model.gru.parameters(), 'lr': args.lr}
            ], lr=0., weight_decay=0.)
    else:
        raise Exception('No model is found!')

    dset_loaders, dset_sizes = data_loader(args)
    scheduler = AdjustLR(optimizer, [args.lr], sleep_epochs=1, half=5, verbose=1)
    if args.test:
        train_test(model, dset_loaders, criterion, 0, 'val', optimizer, args, logger, use_gpu, save_path)
        train_test(model, dset_loaders, criterion, 0, 'test', optimizer, args, logger, use_gpu, save_path)
        return
    for epoch in range(0,args.epochs):
        scheduler.step(epoch)
        model = train_test(model, dset_loaders, criterion, epoch, 'train', optimizer, args, logger, use_gpu, save_path)
        train_test(model, dset_loaders, criterion, epoch, 'val', optimizer, args, logger, use_gpu, save_path)


def main():
    # Settings
    parser = argparse.ArgumentParser(description='AVE Speech Dataset')
    parser.add_argument('--nClasses', default=101, type=int, help='the number of classes')
    parser.add_argument('--path', default='', help='path to model')
    parser.add_argument('--dataset', default='audio', help='path to dataset')
    parser.add_argument('--mode', default='finetuneGRU', help='temporalConv, backendGRU, finetuneGRU')
    parser.add_argument('--every-frame', default=True, action='store_true', help='predicition based on every frame')
    parser.add_argument('--lr', default=0.0003, type=float, help='initial learning rate')
    parser.add_argument('--batch-size', default=36, type=int, help='mini-batch size (default: 36)')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs')
    parser.add_argument('--interval', default=10, type=int, help='display interval')
    parser.add_argument('--test', default=False, action='store_true', help='perform on the test phase')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_gpu = torch.cuda.is_available()
    test_adam(args, use_gpu)


if __name__ == '__main__':
    main()

