'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.module import _addindent

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
import scipy.cluster.hierarchy as hcluster
import scipy.cluster.hierarchy as hac
import scipy.cluster.hierarchy as fclusterdata
import time
from sklearn.preprocessing import normalize

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    total =0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])


        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
            total+=params
            print(params)
            print('total is ',total)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    # best_acc = 0.0
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving weight')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_pruned_retrain.t7')
        best_acc = acc

    return best_acc

def cluster_weights_agglo(weight, threshold, average=True):
    t0 = time.time()
    weight = weight.T
    weight = normalize(weight, norm='l2', axis=1)
    threshold =  1.0-threshold   # Conversion to distance measure
    clusters = hcluster.fclusterdata(weight, threshold, criterion="distance", metric='cosine', depth=1, method='centroid')
    z = hac.linkage(weight, metric='cosine', method='complete')
    labels = hac.fcluster(z, threshold, criterion="distance")

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    #print(n_clusters_)
    elapsed_time = time.time() - t0
    # print(elapsed_time)

    a=np.array(labels)
    sort_idx = np.argsort(a)
    a_sorted = a[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_items = a_sorted[unq_first]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    first_ele = [unq_idx[idx][-1] for idx in xrange(len(unq_idx))]
    return n_clusters_, first_ele

cfg =  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M',512]

class VGG19X(nn.Module):
    def __init__(self):
        super(VGG19X, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(cfg[-1], 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg[:-1]:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--nbepochs', default=100, type=int, help='number of epochs')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
# best_acc = 0  # best test accuracy
# start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

tau_values = [0.54] # Error is 6.33 %

nb_remanining_filters_all = []
test_acc_c1= []

for threshold in tau_values:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    print('best_acc is ', best_acc)
    start_epoch = checkpoint['epoch']
    best_acc = 0.

    print('==> Constructing pruned network..')
    # print(torch_summarize(net))
    ii = 0
    first_ele = None
    nb_remanining_filters = []
    total_flop_after_pruning = 0
    rr = 1
    for layer in net.modules():
        #print(layer)
        if isinstance(layer, nn.ReLU):
            rr+=1
        if isinstance(layer, nn.MaxPool2d):
            rr+=1
        if isinstance(layer, nn.Conv2d):
            weight = layer.weight.data.cpu().numpy()
            bias = layer.bias.data.cpu().numpy()
            if first_ele is not None:
                weight_layers_rearranged = np.transpose(weight, (1, 0, 2, 3))
                weight_layers_rearranged_pruned = weight_layers_rearranged[first_ele]
                weight_layers_rearranged_pruned = np.transpose(weight_layers_rearranged_pruned, (1, 0, 2, 3))
            else:
                weight_layers_rearranged_pruned = weight

            weight_layers_rearranged = np.reshape(weight_layers_rearranged_pruned, [weight_layers_rearranged_pruned.shape[0], -1])
            n_clusters_,first_ele = cluster_weights_agglo(weight_layers_rearranged.T, threshold)
            first_ele = sorted(first_ele)
            nb_remanining_filters.append(n_clusters_)

            weight_pruned = weight_layers_rearranged[first_ele]
            bias_pruned = bias[first_ele]
            weight_pruned = np.reshape(weight_pruned, [n_clusters_, weight_layers_rearranged_pruned.shape[1],weight_layers_rearranged_pruned.shape[2],weight_layers_rearranged_pruned.shape[3]])

            params_1 = np.shape(weight_pruned)
            layer.out_channels = params_1[0]
            layer.in_channels = params_1[1]

            weight_tensor = torch.from_numpy(weight_pruned)
            bias_tensor = torch.from_numpy(bias_pruned)
            layer.weight = torch.nn.Parameter(weight_tensor)
            layer.bias = torch.nn.Parameter(bias_tensor)

            params_1 = np.shape(weight_pruned)
            C1_1 = int(params_1[0])
            C2_1 = int(params_1[1])
            K1_1 = int(params_1[2])
            K2_1 = int(params_1[3])
            x = Variable(torch.randn(1,3, 32, 32))
            nett_1 = nn.Sequential(*list(net.features.children())[:rr])
            out_1 = nett_1(x)
            img_size_1 = out_1.size()
            # print('feature map size is:', img_size_1)
            # print('weight size is:', params_1)

            H_1 = img_size_1[2]
            W_1 = img_size_1[3]
            if ii==0:
                H_1 = 32
                W_1 = 32

            flops_1 = C1_1*C2_1*K1_1*K2_1*H_1*W_1
            print('flop is ',flops_1, '\n')
            total_flop_after_pruning +=flops_1
            # print(ii)
            ii+=1
            rr+=1

        if isinstance(layer, nn.BatchNorm2d) and first_ele is not None:
            bnorm_weight = layer.weight.data.cpu().numpy()
            bnorm_weight = bnorm_weight[first_ele]
            bnorm_bias = layer.bias.data.cpu().numpy()
            bnorm_bias = bnorm_bias[first_ele]

            bnorm_tensor = torch.from_numpy(bnorm_weight)
            bias_tensor = torch.from_numpy(bnorm_bias)
            layer.weight = torch.nn.Parameter(bnorm_tensor)
            layer.bias = torch.nn.Parameter(bias_tensor)

            layer.num_features = int(np.shape(bnorm_weight)[0])
            bnorm_rm = layer.running_mean.cpu().numpy()
            bnorm_rm = bnorm_rm[first_ele]
            bnorm_rv = layer.running_var.cpu().numpy()
            bnorm_rv = bnorm_rv[first_ele]
            running_mean = torch.from_numpy(bnorm_rm)
            layer.running_mean = running_mean
            running_var = torch.from_numpy(bnorm_rv)
            layer.running_var = running_var
            rr+=1

        if isinstance(layer, nn.Linear):
            weight_linear = layer.weight.data.cpu().numpy()
            weight_linear_rearranged = np.transpose(weight_linear, (1, 0))
            weight_linear_rearranged_pruned = weight_linear_rearranged[first_ele]
            weight_linear_rearranged_pruned = np.transpose(weight_linear_rearranged_pruned, (1, 0))
            layer.in_features = int(np.shape(weight_linear_rearranged_pruned)[1])
            linear_tensor = torch.from_numpy(weight_linear_rearranged_pruned)
            layer.weight = torch.nn.Parameter(linear_tensor)

            params_linear = np.shape(weight_linear_rearranged_pruned)
            C1_1 = params_linear[0]
            C2_1 = params_linear[1]

            flops_1 = C1_1*C2_1
            total_flop_after_pruning +=flops_1

    print('flops after pruning:',total_flop_after_pruning)
    print(nb_remanining_filters)
    print(torch_summarize(net))

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(start_epoch, start_epoch+args.nbepochs):
        train(epoch)
        acc_best = test(epoch)

print('error isL', 100-acc_best)
print(nb_remanining_filters)
