print("Epsilon : {}, Robust Accuracy: {}".format(eps, robust_acc))
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import sys
import time
import math

import torch.nn.init as init
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import random_split

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import time
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DDN:
    """
    DDN attack: decoupling the direction and norm of the perturbation to achieve a small L2 norm in few steps.
    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    gamma : float, optional
        Factor by which the norm will be modified. new_norm = norm * (1 + or - gamma).
    init_norm : float, optional
        Initial value for the norm.
    quantize : bool, optional
        If True, the returned adversarials will have quantized values to the specified number of levels.
    levels : int, optional
        Number of levels to use for quantization (e.g. 256 for 8 bit images).
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.
    callback : object, optional
        Visdom callback to display various metrics.
    """
    def __init__(self,
                 steps: int,
                 gamma: float = 0.05,
                 init_norm: float = 1.,
                 quantize: bool = True,
                 levels: int = 256,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu'),
                 callback: Optional = None) -> None:
        self.steps = steps
        self.gamma = gamma
        self.init_norm = init_norm
        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm
        self.device = device
        self.callback = callback
    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.
        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.
        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.
        """
        inputs_max = inputs.max()
        inputs_min = inputs.min()
        inputs = (inputs-inputs_min) / (inputs_max-inputs_min)
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros_like(inputs, requires_grad=True)
        norm = torch.full((batch_size,), self.init_norm, device=self.device, dtype=torch.float)
        worst_norm = torch.max(inputs, 1 - inputs).view(batch_size, -1).norm(p=2, dim=1)
        # Setup optimizers
        optimizer = optim.SGD([delta], lr=1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps, eta_min=0.01)
        best_l2 = worst_norm.clone()
        best_delta = torch.zeros_like(inputs)
        adv_found = torch.zeros(inputs.size(0), dtype=torch.uint8, device=self.device)
        for i in range(self.steps):
            scheduler.step()
            l2 = delta.data.view(batch_size, -1).norm(p=2, dim=1)
            # adv = inputs + delta
            adv = (inputs + delta) * (inputs_max-inputs_min) + inputs_min
            logits = model(adv)
            pred_labels = logits.argmax(1)
            ce_loss = F.cross_entropy(logits, labels, reduction='sum')
            loss = multiplier * ce_loss
            is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
            is_smaller = l2 < best_l2
            is_both = is_adv * is_smaller
            adv_found[is_both] = 1
            best_l2[is_both] = l2[is_both]
            best_delta[is_both] = delta.data[is_both]
            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            if self.callback:
                cosine = F.cosine_similarity(-delta.grad.view(batch_size, -1),
                                             delta.data.view(batch_size, -1), dim=1).mean().item()
                self.callback.scalar('ce', i, ce_loss.item() / batch_size)
                self.callback.scalars(
                    ['max_norm', 'l2', 'best_l2'], i,
                    [norm.mean().item(), l2.mean().item(),
                     best_l2[adv_found].mean().item() if adv_found.any() else norm.mean().item()]
                )
                self.callback.scalars(['cosine', 'lr', 'success'], i,
                                      [cosine, optimizer.param_groups[0]['lr'], adv_found.float().mean().item()])
            optimizer.step()
            norm.mul_(1 - (2 * is_adv.float() - 1) * self.gamma)
            norm = torch.min(norm, worst_norm)
            delta.data.mul_((norm / delta.data.view(batch_size, -1).norm(2, 1)).view(-1, 1, 1, 1))
            delta.data.add_(inputs)
            if self.quantize:
                delta.data.mul_(self.levels - 1).round_().div_(self.levels - 1)
            delta.data.clamp_(0, 1).sub_(inputs)
        if self.max_norm:
            best_delta.renorm_(p=2, dim=0, maxnorm=self.max_norm)
            if self.quantize:
                best_delta.mul_(self.levels - 1).round_().div_(self.levels - 1)
        # return inputs + best_delta
        return (inputs + best_delta)* (inputs_max-inputs_min) + inputs_min



def create_dataset_dots(k=100):
    X_1 = np.zeros([1, 1, k, k])
    X_1[:, :, k//2, k//2] = 1
    X_2 = np.zeros([1, 1, k, k])
    X_2[:, :, k//2, k//2] = -1
    Xs = np.concatenate([X_1, X_2], 0)
    ys = np.hstack([np.zeros(1), np.ones(1)])
    return Xs, ys


class simple_Conv_2D(nn.Module):
    def __init__(self, n_hidden, kernel_size=28):
        super(simple_Conv_2D, self).__init__()
        padding_size = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(1, n_hidden, kernel_size=kernel_size, padding=padding_size, padding_mode='circular'),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(n_hidden, 2)
    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class simple_FC_2D(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(simple_FC_2D, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input, n_hidden),
            nn.ReLU()
        )
        self.classifier = nn.Linear(n_hidden, 2)
    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out
# Training
def train(epoch, net, batch):
    print('\nEpoch: %d' % epoch)
    inputs, targets = batch
    batch_idx = 0
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    progress_bar(batch_idx, 1, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)


def test(epoch, net, model_name, batch):
    global best_acc
    inputs, targets = batch
    batch_idx = 0
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, 1, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if epoch == start_epoch+199:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('./ckpt'):
            os.mkdir('./ckpt')
        torch.save(state, './ckpt/%s.pth'%model_name)
        best_acc = acc



def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
term_width = 65

dists = {'FC':[], 'Conv':[]}
losses = {'FC':[], 'Conv':[]}
for model in ['FC', 'Conv']:
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    for i in range(15):
        k = i*2 + 1
        # batch = create_dataset_sin_cos(k)
        batch = create_dataset_dots(k)
        n_hidden = 100
        kernel_size = k
        # Model
        if model == 'FC':
            net = simple_FC_2D(k*k, n_hidden)
        elif model == 'Conv':
            net = simple_Conv_2D(n_hidden, kernel_size)
        print('Number of parameters: %d'%sum(p.numel() for p in net.parameters()))
        net = net.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        torch_batch = [torch.FloatTensor(batch[0]), torch.LongTensor(batch[1])]
        for epoch in range(start_epoch, start_epoch+3000):
            loss_epoch = train(epoch, net, torch_batch)
            test(epoch, net, 'simple_%s_%d_%d'%(model, n_hidden, k), torch_batch)
            if epoch == 999:
                optimizer.param_groups[0]['lr'] = 0.5
            if epoch == 1999:
                optimizer.param_groups[0]['lr'] = 0.1
            if epoch > 2999 and epoch % 1000 == 0:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
            # scheduler.step()
        inputs, targets = torch_batch
        n_steps = [1000]
        for n_step in n_steps:
            attacker = DDN(steps=n_step, device=torch.device('cuda'))
            adv_norm = 0.
            adv_correct = 0
            total = 0
            inputs, targets = inputs.to(device), targets.to(device)
            adv = attacker.attack(net, inputs.to(device), labels=targets.to(device), targeted=False)
            adv_outputs = net(adv)
            _, adv_predicted = adv_outputs.max(1)
            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()
            adv_norm += l2_norm(adv - inputs.to(device)).sum().item()
            print('DDN (n-step = {:.0f}) done in Success: {:.2f}%, Mean L2: {:.4f}.'.format(
                n_step,
                100.*adv_correct/total,
                adv_norm/total
            ))
        dists[model].append(adv_norm/total)
        losses[model].append(loss_epoch)
        if k == 15:
            torchvision.utils.save_image(inputs[0]/2.+0.5, 'clean1.png')
            torchvision.utils.save_image(inputs[1]/2.+0.5, 'clean2.png')
            torchvision.utils.save_image(adv[0]/2.+0.5, '%s_adv1.png'%model)
            torchvision.utils.save_image(adv[1]/2.+0.5, '%s_adv2.png'%model)
FCN_dists = dists['FC']
CNN_dists = dists['Conv']

CAND_COLORS = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f', '#ff7f00','#cab2d6','#6a3d9a', '#90ee90', '#9B870C', '#2f4554', '#61a0a8', '#d48265', '#c23531']
colors1 = CAND_COLORS[::2]
colors2 = CAND_COLORS[1::2]

font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)

with plt.style.context('bmh'):
    fig = plt.figure(dpi=250, figsize=(10, 5.5))
    plt.clf()
    ks = [i*2 + 1 for i in range(15)]
    ax = plt.subplot(111)
    plt.plot(ks, FCN_dists[:15], marker='o', linewidth=3, markersize=8, label='FCN', color=colors2[0], alpha=0.8)
    plt.plot(ks, CNN_dists[:15], marker='^', linewidth=3, markersize=8, label='CNN', color=colors2[2], alpha=0.8)
    plt.plot(ks, [1.0/(i*2+1) for i in range(15)], linestyle='dotted', marker='*', linewidth=3, markersize=8, label='CNTKKKK', color=colors2[1], alpha=0.8)
    # plt.tight_layout()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.178), ncol=3)
    # plt.ylabel('average distance of attack')
    # plt.xlabel('image width')
    plt.savefig('figure1.png')