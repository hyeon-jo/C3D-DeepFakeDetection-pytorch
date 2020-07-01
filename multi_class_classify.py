"""Train 3D ConvNets to action classification."""
import os
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim

from datasets.faceforensics import FaceForensicsDataset
from models.resnet import generate_model
from utils import AverageMeter, calculate_accuracy, calculate_accuracy_per_class


def load_pretrained_weights(ckpt_path):
    """load pretrained weights and adjust params name."""
    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path)
    pretrained_weights = pretrained_weights['state_dict']
    for name, params in pretrained_weights.items():
        if not 'fc' in name:
            # name = name[name.find('.')+1:]
            adjusted_weights[name] = params
            print('Pretrained weight name: [{}]'.format(name))
    return adjusted_weights


def load_pretrained_resnet(model, pretrain_path, n_finetune_classes):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path)

        model.load_state_dict(pretrain['state_dict'])
        tmp_model = model
        tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                 n_finetune_classes)

    return model


def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]


def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True

        if add_flag:
            parameters.append({'params': v})

    return parameters


def train(args, model, criterion, optimizer, device, train_dataloader, epoch):
    torch.set_grad_enabled(True)
    model.train()

    losses = AverageMeter()
    accuracy = AverageMeter()
    for i, data in enumerate(train_dataloader, 1):
        # get inputs
        clips, _, idxs = data
        inputs = clips.to(device).half()
        targets = idxs.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward and backward
        outputs = model(inputs) # return logits here
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # compute loss and acc
        losses.update(loss.item())
        accuracy.update(calculate_accuracy(outputs, targets))
        del(inputs, targets, outputs)
        # print statistics and write summary every N batch
        if i % args.pf == 0:
            print('[TRAIN] Epoch: [{}][{:3d}/{}], loss: {:.3f}({:.3f}), acc: {:.4f}({:.4f})'.format(
                epoch, i, len(train_dataloader), losses.val, losses.avg, accuracy.val, accuracy.avg))

    return losses.avg


def validate(args, model, criterion, device, val_dataloader, epoch):
    torch.set_grad_enabled(False)
    model.eval()
    
    total_loss = 0.0
    accuracy = AverageMeter()
    # acc_per_class = [AverageMeter()] * 5

    for i, data in enumerate(val_dataloader):
        # get inputs
        clips, _, idxs = data
        inputs = clips.to(device).half()
        targets = idxs.to(device)
        # forward
        outputs = model(inputs) # return logits here
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        accuracy.update(calculate_accuracy(outputs, targets))
        # for j in range(0, 5):
        #     acc_per_class[j].update(calculate_accuracy_per_class(outputs, targets, j))
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(val_dataloader)
    print('[VAL] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, accuracy.avg))
    # print('[Accuracy per classes]\n'
    #       'Pristine: {:.4f}\t'
    #       'Deepfakes: {:.4f}\t'
    #       'Face2Face: {:.4f}\t'
    #       'FaceSwap: {:.4f}\t'
    #       'NeuralTexture: {:.4f}'.format(
    #     acc_per_class[0].avg, acc_per_class[1].avg, acc_per_class[2].avg, acc_per_class[3].avg, acc_per_class[4].avg))
    return avg_loss


def parse_args():
    pretrain_default = '/home/hyeon/deepfake/pretrained/r3d18_K_200ep.pth'
        # '/disk3/hyeon/eccv2020/ucf/fsop/c3d_cl16_it8_tl3_3clips_01221118/best_model_260.pt'
    parser = argparse.ArgumentParser(description='Video Classification')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model', type=str, default='c3d', help='c3d/r3d/r21d')
    parser.add_argument('--dataset', type=str, default='faceforensics', help='ucf101/hmdb51')
    parser.add_argument('--split', type=str, default='1', help='dataset split')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--gpu', type=str, default='0, 1, 2, 3, 4, 5, 6, 7', help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', default='/disk3/hyeon/deepfake_logs/', type=str, help='log directory')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--desp', type=str, default='attn_multi_class', help='additional description')
    parser.add_argument('--epochs', type=int, default=500, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=128, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=20, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=10, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    parser.add_argument('--pretrain', type=str, default=pretrain_default)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    # torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)

    ########### model ##############
    if args.dataset == 'ucf101':
        class_num = 101
    elif args.dataset == 'hmdb51':
        class_num = 51
    elif args.dataset == 'faceforensics':
        class_num = 5

    if args.model == 'c3d':
        print(class_num)
        model = generate_model(model_depth=18,
                               n_classes=700,
                               n_input_channels=3,
                               shortcut_type='B',
                               conv1_t_size=7,
                               conv1_t_stride=1,
                               no_max_pool=True,
                               widen_factor=1.0).to(device)

    if args.mode == 'train':  ########### Train #############
        if args.ckpt:  # resume training
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(args.ckpt))
            log_dir = os.path.dirname(args.ckpt)
            parameters = model.parameters()
        else:
            if args.pretrain:
                model = load_pretrained_resnet(model, args.pretrain, class_num)
                model = nn.DataParallel(model).to(device)
                parameters = get_fine_tuning_parameters(model, 'layer4')
            else:
                model = nn.DataParallel(model)
                parameters = model.parameters()

            if args.desp:
                exp_name = '{}_cl{}_{}_{}'.format(args.model, args.cl, args.desp, time.strftime('%m%d%H%M'))
            else:
                exp_name = '{}_cl{}_{}'.format(args.model, args.cl, time.strftime('%m%d%H%M'))
            log_dir = os.path.join(args.log, exp_name)
        model = model.to(device)
        model.half()
        os.makedirs(log_dir, exist_ok=True)

        train_transforms = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        if args.dataset == 'faceforensics':
            train_dataset = FaceForensicsDataset('/disk1/dataset/FaceForensics_dist', args.cl, True, train_transforms)
            val_size = 800
            random.seed(args.seed)
            train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset)-val_size, val_size))
            val_dataset.train = False


        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                    num_workers=args.workers, pin_memory=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)

        print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))

        ### loss funciton, optimizer and scheduler ###

        # print([name for name, param in model.named_parameters() if 'attention' in name or 'layer4' in name or 'fc' in name])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-06, patience=20, factor=0.1)

        prev_best_val_loss = float('inf')
        prev_best_model_path = None
        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            time_start = time.time()
            train_loss = train(args, model, criterion, optimizer, device, train_dataloader, epoch)
            print('Epoch time: {:.2f} s\tTrain loss: {:.4f}.'.format(time.time() - time_start, train_loss))
            val_loss = validate(args, model, criterion, device, val_dataloader, epoch)
            scheduler.step(val_loss)
            # save model every 20 epoches
            if epoch % 20 == 0:
                torch.save(model.state_dict(), os.path.join(log_dir, 'model_{}.pt'.format(epoch)))
            # save model for the best val
            if val_loss < prev_best_val_loss:
                model_path = os.path.join(log_dir, 'best_model_{}.pt'.format(epoch))
                torch.save(model.state_dict(), model_path)
                prev_best_val_loss = val_loss
                if prev_best_model_path:
                    os.remove(prev_best_model_path)
                prev_best_model_path = model_path
