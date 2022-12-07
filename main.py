"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""
from __future__ import print_function
import argparse
import os
#import cPickle as pickle
import pickle
import random
import numpy as np
import csv

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from model import RN2, RN, CNN_MLP


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN2','RN', 'CNN_MLP'], default='RN2', 
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')
parser.add_argument('--relation-type', type=str, default='binary',
                    help='what kind of relations to learn. options: binary, ternary (default: binary)')
parser.add_argument('--state_desc', type=str, default='state-desc',
                    help='what kind of representations of images. options: state desc, pixel (default: state-desc)')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

summary_writer = SummaryWriter()

if args.model=='CNN_MLP': 
  model = CNN_MLP(args)
elif args.model=='RN2':
  model = RN2(args)
else:
  model = RN(args)
  
model_dirs = './model'
bs = args.batch_size
input_img = torch.FloatTensor(bs, 3, 75, 75)
input_state = torch.FloatTensor(bs, 6, 7)
input_qst = torch.FloatTensor(bs, 11)
label = torch.LongTensor(bs)

if args.cuda:
    model.cuda()
    input_img = input_img.cuda()
    input_state = input_state.cuda()
    input_qst = input_qst.cuda()
    label = label.cuda()

input_img = Variable(input_img)
input_qst = Variable(input_qst)
input_state = Variable(input_state)
label = Variable(label)

def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    state = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[3][bs*i:bs*(i+1)]))

    input_img.data.resize_(img.size()).copy_(img)
    input_state.data.resize_(state.size()).copy_(state)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data):
    img = [e[0] for e in data]
    state = [e[1] for e in data]
    qst = [e[2] for e in data]
    ans = [e[3] for e in data]
    return (img,state,qst,ans)

    
def train(epoch, rel, is_state_desc):
    model.train()
    random.shuffle(rel)

    rel = cvt_data_axis(rel)
    
    acc_rels = []
    
    l_binary = []


    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(rel, batch_idx)
        
        if not is_state_desc:
           accuracy_rel, loss_binary = model.train_(input_img, None, input_qst, label) 
        else:    
            accuracy_rel, loss_binary = model.train_(input_img, input_state, input_qst, label)
        acc_rels.append(accuracy_rel.item())
        l_binary.append(loss_binary.item())



        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] '
                  'Relations accuracy: {:.0f}% '.format(
                   epoch,
                   batch_idx * bs * 2,
                   len(rel[0]) * 2,
                   100. * batch_idx * bs / len(rel[0]),
                   accuracy_rel))
        

    avg_acc_binary = sum(acc_rels) / len(acc_rels)

    summary_writer.add_scalars('Accuracy/train', {
        'binary': avg_acc_binary
    }, epoch)


    avg_loss_binary = sum(l_binary) / len(l_binary)


    summary_writer.add_scalars('Loss/train', {
        'binary': avg_loss_binary
    }, epoch)

    return avg_acc_binary

def test(epoch, rel):
    model.eval()

    rel = cvt_data_axis(rel)

    accuracy_rels = []

    loss_binary = []

    for batch_idx in range(len(rel[0]) // bs):


        tensor_data(rel, batch_idx)
        acc_bin, l_bin = model.test_(input_img, input_state, input_qst, label)
        accuracy_rels.append(acc_bin.item())
        loss_binary.append(l_bin.item())




    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)

    print('\n Test set: Binary accuracy: {:.0f}% \n'.format(
        accuracy_rel))

    summary_writer.add_scalars('Accuracy/test', {
        'binary': accuracy_rel
    }, epoch)


    loss_binary = sum(loss_binary) / len(loss_binary)


    summary_writer.add_scalars('Loss/test', {
        'binary': loss_binary
    }, epoch)

    return accuracy_rel

    
def load_data():
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr2.pickle')
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)

    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, state, relations, norelations in train_datasets:
        img = np.swapaxes(img, 0, 2)

        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,state,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,state,qst,ans))

    for img, state, relations, norelations in test_datasets:
        img = np.swapaxes(img, 0, 2)

        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,state,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,state,qst,ans))
    
    return (rel_train, rel_test, norel_train, norel_test)
    

rel_train, rel_test, norel_train, norel_test = load_data()

try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))

if args.resume:
    #filename = os.path.join(model_dirs, args.resume)
    filename = os.path.join("./saved_model/", args.resume)
    print(filename)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

with open(f'./{args.model}_{args.seed}_log.csv', 'w') as log_file:
    csv_writer = csv.writer(log_file, delimiter=',')
    csv_writer.writerow(['epoch', 'train_acc_rel',
                     'train_acc_norel', 'test_acc_rel', 'test_acc_norel'])

    print(f"Training {args.model} {f'({args.relation_type})' if args.model == 'RN' else ''} model...")
    is_state_desc = False
    if args.state_desc == 'state-desc': 
        is_state_desc = True
    print(is_state_desc)
    for epoch in range(1, args.epochs + 1):

        train_acc_binary= train(
            epoch, rel_train, is_state_desc)
        test_acc_binary= test(
            epoch, rel_test)

        csv_writer.writerow([epoch, train_acc_binary,
                          test_acc_binary])
        model.save_model(epoch)