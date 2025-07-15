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
from EVAdataset_CSR import *
from cvtransforms import *
from lr_scheduler import *
from audio_only.audio_model_BGRU_CSR import audiosr as audio_model_sr
from lip_only.lip_model_CSR import lipreading as video_model_sr
from emg_only.emg_model_BGRU_CSR import EMGNet as emg_model_sr
from concat_model_trans_CSR import TRANSFORMERConcat as concat_model_sr

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


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(tgt, device):
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    tgt_padding_mask = (tgt == 0).transpose(0, 1)
    return tgt_mask, tgt_padding_mask #(S, S), (Batch, S)

def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def ctc_decode(y):
    result = []
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]


def train_test(audio_model, lip_model, emg_model, concat_model, dset_loaders, crit_ctc, crit_ce, epoch, phase, optimizer, args, logger, use_gpu, save_path):
    if phase == 'val' or phase == 'test':
        concat_model.eval()
        audio_model.eval()
        lip_model.eval()
        emg_model.eval()
    if phase == 'train':
        concat_model.train()
        audio_model.eval()
        lip_model.eval()
        emg_model.eval()
    if phase == 'train':
        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
        logger.info('Current Learning rate: {}'.format(showLR(optimizer)))

    train_wer = []
    running_loss, running_all = 0., 0.
    loss_list = []
    wer = []

    if phase == 'train':
        for batch_idx, (audio, emg, lip, annos, annos_len, fusion_len, annos_attn) in enumerate(dset_loaders[phase]):
            emg = emg.reshape(emg.shape[0], emg.shape[2], emg.shape[3], -1)
            
            lip = lip.view(lip.size(0), -1, lip.size(3), lip.size(4))
            batch_img = ColorNormalize(lip.cpu().numpy())
            batch_img = np.reshape(batch_img, (batch_img.shape[0], 1, batch_img.shape[1], batch_img.shape[2], batch_img.shape[3]))
            lip = torch.from_numpy(batch_img)

            audio, emg, lip = audio.float(), emg.float(), lip.float()
            audio, emg, lip, annos, annos_attn = Variable(audio.cuda()), Variable(emg.cuda()), Variable(lip.cuda()), Variable(annos.cuda()), Variable(annos_attn.cuda())
            audio, emg, lip, annos, annos_len, fusion_len, annos_attn = audio.cuda(), emg.cuda(), lip.cuda(), annos.cuda(), annos_len.cuda(), fusion_len.cuda(), annos_attn.cuda()

            txt_attn_input = annos_attn[:, :-1]
            txt_attn_output = annos_attn[:, 1:]
            tgt_mask, tgt_padding_mask = create_mask(txt_attn_input.transpose(0, 1), device='cuda')

            audio_outputs = audio_model(audio) 
            video_outputs = lip_model(lip) 
            emg_outputs = emg_model(emg) 
            inputs = torch.cat((audio_outputs, video_outputs, emg_outputs), dim=1)

            y_ctc, y_ce = concat_model(inputs, txt_attn_input, tgt_mask, tgt_padding_mask)
            y_ce = y_ce.contiguous().view(-1, y_ce.size(-1))

            loss_ctc = crit_ctc(y_ctc.transpose(0, 1).log_softmax(-1), annos, fusion_len.view(-1), annos_len.view(-1))
            loss_ce = crit_ce(y_ce, txt_attn_output.contiguous().view(-1))
            loss = 0.2 * loss_ctc + 0.8 * loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data 
            pred_txt = ctc_decode(y_ctc)
            truth_txt = [MyDataset.arr2txt(annos[_], start=1) for _ in range(annos.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
            running_all += len(inputs)

            if batch_idx == 0:
                since = time.time()
            elif batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase])-1):
                print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tWER:{:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                    running_all,
                    len(dset_loaders[phase].dataset),
                    100. * batch_idx / (len(dset_loaders[phase])-1),
                    running_loss / running_all,
                    np.array(train_wer).mean(),
                    time.time()-since,
                    (time.time()-since)*(len(dset_loaders[phase])-1) / batch_idx - (time.time()-since))),
        print
        logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tWER:{:.4f}'.format(
            phase,
            epoch,
            running_loss / len(dset_loaders[phase].dataset),
            np.array(train_wer).mean()))
        torch.save(concat_model.state_dict(), save_path+'/Transformer_CSR_'+str(epoch+1)+'.pt')
        return audio_model, lip_model, emg_model, concat_model


    if phase == 'val' or phase == 'test':

        with torch.no_grad():
            for batch_idx, (audio, emg, lip, annos, annos_len, fusion_len, annos_attn) in enumerate(dset_loaders[phase]):
                emg = emg.reshape(emg.shape[0], emg.shape[2], emg.shape[3], -1)

                lip = lip.view(lip.size(0), -1, lip.size(3), lip.size(4))
                batch_img = ColorNormalize(lip.cpu().numpy())
                batch_img = np.reshape(batch_img, (batch_img.shape[0], 1, batch_img.shape[1], batch_img.shape[2], batch_img.shape[3]))
                lip = torch.from_numpy(batch_img)
                
                audio, emg, lip = audio.float(), emg.float(), lip.float()
                audio, emg, lip, annos, annos_attn = Variable(audio.cuda()), Variable(emg.cuda()), Variable(lip.cuda()), Variable(annos.cuda()), Variable(annos_attn.cuda())
                audio, emg, lip, annos, annos_len, fusion_len, annos_attn = audio.cuda(), emg.cuda(), lip.cuda(), annos.cuda(), annos_len.cuda(), fusion_len.cuda(), annos_attn.cuda()

                txt_attn_input = annos_attn[:, :-1]
                txt_attn_output = annos_attn[:, 1:]
                tgt_mask, tgt_padding_mask = create_mask(txt_attn_input.transpose(0, 1), device='cuda')

                audio_outputs = audio_model(audio)
                video_outputs = lip_model(lip)
                emg_outputs = emg_model(emg)
                inputs = torch.cat((audio_outputs, video_outputs, emg_outputs), dim=1)

                y_ctc, y_ce = concat_model(inputs, txt_attn_input, tgt_mask, tgt_padding_mask)

                loss = crit_ctc(y_ctc.transpose(0, 1).log_softmax(-1), annos, fusion_len.view(-1), annos_len.view(-1)).detach().cpu().numpy()
                running_loss += loss 
                running_all += len(inputs)
                loss_list.append(loss)
                pred_txt = ctc_decode(y_ctc)
                truth_txt = [MyDataset.arr2txt(annos[_], start=1) for _ in range(annos.size(0))]
                wer.extend(MyDataset.wer(pred_txt, truth_txt)) 
                if batch_idx == 0:
                    since = time.time()
                elif batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase])-1):
                    print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tWER:{:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                        running_all,
                        len(dset_loaders[phase].dataset),
                        100. * batch_idx / (len(dset_loaders[phase])-1),
                        running_loss / running_all,
                        np.array(wer).mean(),
                        time.time()-since,
                        (time.time()-since)*(len(dset_loaders[phase])-1) / batch_idx - (time.time()-since))),
            print
            logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tWER:{:.4f}'.format(
                phase,
                epoch,
                running_loss / len(dset_loaders[phase].dataset),
                np.array(wer).mean()))



def test_adam(args, use_gpu):
    save_path = '/ai/mm/fusion_AVEdataset'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # logging info
    filename = save_path+'/'+'Transformer_CSR_'+str(args.lr)+'.txt'
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # create model
    audio_model = audio_model_sr(mode=args.mode, inputDim=512, hiddenDim=512, nClasses=args.nClasses, frameLen=60, every_frame=args.every_frame)
    lip_model = video_model_sr(mode=args.mode, inputDim=512, hiddenDim=512, nClasses=args.nClasses, frameLen=60, every_frame=args.every_frame)
    emg_model = emg_model_sr(mode=args.mode, inputDim=256, hiddenDim=512, nClasses=args.nClasses, frameLen=40, every_frame=args.every_frame)
    concat_model = concat_model_sr(mode=args.mode, inputDim=101, hiddenDim=512, nLayers=2, nClasses=args.nClasses, frameLen=60, every_frame=args.every_frame)

    # reload model
    print('reload audio model')
    audio_model = reload_model(audio_model, logger, args.audio_path)
    print("reload video model")
    lip_model = reload_model(lip_model, logger, args.lip_path)
    print("reload emg model")
    emg_model = reload_model(emg_model, logger, args.emg_path)
    print("reload fusion model")
    concat_model = reload_model(concat_model, logger, args.concat_path)

    # define loss function and optimizer
    crit_ctc = nn.CTCLoss(zero_infinity=True)
    crit_ce = nn.CrossEntropyLoss(ignore_index=0)

    for param in audio_model.parameters():
        param.requires_grad = False
    for param in lip_model.parameters():
        param.requires_grad = False
    for param in emg_model.parameters():
        param.requires_grad = False
    for param in concat_model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(concat_model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    dset_loaders, dset_sizes = data_loader(args)
    scheduler = AdjustLR(optimizer, [args.lr], sleep_epochs=3, half=5, verbose=1)
    if args.test:
        train_test(audio_model, lip_model, emg_model, concat_model, dset_loaders, crit_ctc, crit_ce, 0, 'val', optimizer, args, logger, use_gpu, save_path)
        train_test(audio_model, lip_model, emg_model, concat_model, dset_loaders, crit_ctc, crit_ce, 0, 'test', optimizer, args, logger, use_gpu, save_path)
        return
    for epoch in range(0,args.epochs):
        scheduler.step(epoch)
        audio_model, lip_model, emg_model, concat_model = train_test(audio_model, lip_model, emg_model, concat_model, dset_loaders, crit_ctc, crit_ce, epoch, 'train', optimizer, args, logger, use_gpu, save_path)
        train_test(audio_model, lip_model, emg_model, concat_model, dset_loaders, crit_ctc, crit_ce, epoch, 'val', optimizer, args, logger, use_gpu, save_path)


def main():
    # Settings
    parser = argparse.ArgumentParser(description='AVE Speech Dataset')
    parser.add_argument('--nClasses', default=101, type=int, help='the number of classes')
    parser.add_argument('--audio-path', default='', help='path to pre-trained audio model')
    parser.add_argument('--lip-path', default='', help='path to pre-trained lip model')
    parser.add_argument('--emg-path', default='', help='path to pre-trained emg model')
    parser.add_argument('--concat-path', default='', help='path to pre-trained concat model')
    parser.add_argument('--dataset', default='', help='path to dataset')
    parser.add_argument('--mode', default='finetuneGRU', help='temporalConv, backendGRU, finetuneGRU')
    parser.add_argument('--every-frame', default=True, action='store_true', help='predicition based on every frame')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size (default: 36)')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs')
    parser.add_argument('--interval', default=10, type=int, help='display interval')
    parser.add_argument('--test', default=False, action='store_true', help='perform on the test phase')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_gpu = torch.cuda.is_available()
    test_adam(args, use_gpu)


if __name__ == '__main__':
    main()

