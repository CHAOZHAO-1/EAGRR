
# !/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader_1d
import resnet18_1d as models
import torch.nn as nn
import time
import numpy as np
import random
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Training settings


def train(model):
    src_iter = iter(src_loader)
    Train_Loss_list = []
    Train_Accuracy_list = []
    Test_Loss_list = []
    Test_Accuracy_list = []

    start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_decay)  # 全参优化器

    for i in range(1, args.iteration + 1):
        model.train()
        LEARNING_RATE = args.lr / math.pow((1 + 10 * (i - 1) / (args.iteration)), 0.75)
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE
        if (i - 1) % 100 == 0:
            print(f'learning rate: {LEARNING_RATE:.6f}')

        try:
            src_data, src_label = next(src_iter)
        except Exception:
            src_iter = iter(src_loader)
            src_data, src_label = next(src_iter)

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()

        cls_label = src_label[:, 0]
        dom_label = src_label[:, 1]

        optimizer.zero_grad()

        pred_loss, dir_loss= model(src_data, cls_label, dom_label)

        loss = pred_loss + 1 * dir_loss

        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print(
                f'Train iter: {i} [{100. * i / args.iteration:.0f}%]\tLoss: {loss.item():.6f}\tcls_loss: {pred_loss.item():.6f}\tregularization_loss: {dir_loss.item():.6f}')

        if i % (args.log_interval * 10) == 0:
            train_correct, train_loss = test_source(model, src_loader)
            test_correct, test_loss = test_target(model, tgt_test_loader)

            Train_Accuracy_list.append(train_correct.cpu().numpy() / len(src_loader.dataset))
            Train_Loss_list.append(train_loss)
            Test_Accuracy_list.append(test_correct.cpu().numpy() / len(tgt_test_loader.dataset))
            Test_Loss_list.append(test_loss)



    end = time.time()
    Train_Time = end - start
    name1 = 'store'
    name2 = args.experiment_date + '_' + str(taskindex) + '_' + args.method_name + str(repeat)
    log(name1, name2, Train_Loss_list, Train_Accuracy_list, Test_Loss_list, Test_Accuracy_list, Train_Time)


def test_source(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_test_label = tgt_test_label[:, 0]
            # print(tgt_test_data)
            tgt_pred = model(tgt_test_data, tgt_test_label, tgt_test_label, mode='pred')
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(tgt_name, test_loss, correct,
                                                                               len(test_loader.dataset),
                                                                               100. * correct / len(
                                                                                   test_loader.dataset)))
    return correct, test_loss


def test_target(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            # print(tgt_test_data)
            tgt_pred = model(tgt_test_data, tgt_test_label, tgt_test_label, mode='pred')
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(tgt_name, test_loss, correct,
                                                                               len(test_loader.dataset),
                                                                               100. * correct / len(
                                                                                   test_loader.dataset)))
    return correct, test_loss


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total:{} Trainable:{}'.format(total_num, trainable_num))


def parse_args():
    parser = argparse.ArgumentParser(description='Training 1D CNN with FFT for Domain Adaptation')
    parser.add_argument('--iteration', type=int, default=8000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--dataset', type=str, default='HTBF')
    parser.add_argument('--class_num', type=int, default=9)

    parser.add_argument('--FFT', type=str, default='FRE')
    parser.add_argument('--method_name', type=str, default='EAGRR')
    parser.add_argument('--experiment_date', type=str, default='2025.6.26')

    parser.add_argument('--start_task', type=int, default=1)
    parser.add_argument('--end_task', type=int, default=2)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--log_interval', type=int, default=10)

    parser.add_argument('--repeat_num', type=int, default=1)

    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--l2_decay', type=float, default=5e-4)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    src_tar = np.array([

        [0, 1, 2],
        [0, 2, 1],
        [1, 2, 0],

    ])

    for taskindex in range(args.start_task, args.end_task):

        src = np.array(src_tar[taskindex][:2])
        tgt = np.array(src_tar[taskindex][2:])

        for repeat in range(args.repeat_num):



            root_path = 'D:\\ZHAOCHAO\\' + args.dataset + str(args.class_num) + '.mat'


            #
            tgt_name = "testing"

            cuda = not args.no_cuda and torch.cuda.is_available()
            torch.manual_seed(args.seed)
            if cuda:
                torch.cuda.manual_seed(args.seed)

            kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

            src_loader = data_loader_1d.load_training(root_path, src, args.FFT, args.class_num,
                                                      args.batch_size, kwargs)

            tgt_test_loader = data_loader_1d.load_testing(root_path, tgt, args.FFT, args.class_num,
                                                          args.batch_size, kwargs)

            src_dataset_len = len(src_loader.dataset)

            src_loader_len = len(src_loader)
            model = models.EAGRR(num_classes=args.class_num)

            print(model)
            if cuda:
                model.cuda()
            train(model)



