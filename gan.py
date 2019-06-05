import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.utils import save_image

import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from maxout import Maxout

lzsize = 1

class Dloss(nn.Module):
    def __init__(self):
        super(Dloss, self).__init__()

    def forward(self, x, z):
        if x.shape != z.shape:
            raise ArithmeticError('Sizes input to DLoss are not same.')
        #if torch.max(x) >= 1 or torch.max(z) >= 1: raise ArithmeticError('Is it true D returns probability?')
        x = x.view(-1, 1)
        z = z.view(-1, 1)
        return -torch.sum(torch.log(x) + torch.log(1 - z)) / len(x)


class GLoss(nn.Module):
    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, x):
        #if torch.max(z) >= 1: raise ArithmeticError('Is it true D returns probability?')
        x = x.view(-1, 1)
        return -torch.sum(torch.log(x)) / len(x)*10


# network generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.convtr2d1=nn.ConvTranspose2d(1,8,4,2)
        self.convtr2d2=nn.ConvTranspose2d(8,4,4,2)
        self.convtr2d3=nn.ConvTranspose2d(4,4,4,2)
        self.convtr2d4=nn.ConvTranspose2d(4,2,4,2,4)
        self.convtr2d5=nn.ConvTranspose2d(2,1,4,1,5)
        self.batchnomal1=nn.BatchNorm2d(8)
        self.batchnomal2 = nn.BatchNorm2d(4)
        self.batchnomal3 = nn.BatchNorm2d(4)
        self.batchnomal4 = nn.BatchNorm2d(2)

    def forward(self, x):
        x = self.convtr2d1(x)
        x=self.batchnomal1(x)
        x=F.relu(x)
        x=self.convtr2d2(x)
        x=self.batchnomal2(x)
        x=F.relu(x)
        x=self.convtr2d3(x)
        x=self.batchnomal3(x)
        x=F.relu(x)
        x=self.convtr2d4(x)
        x=self.batchnomal4(x)
        x=F.relu(x)
        x=self.convtr2d5(x)
        x=F.tanh(x)
        return x


# network of discriminator
# TODO USE MAXOUT AND DROPOUT
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv2d1=nn.Conv2d(1,4,4,2,1)
        self.conv2d2=nn.Conv2d(4,8,4,2,1)
        self.conv2d3=nn.Conv2d(8,16,4,2,1)
        self.conv2d4=nn.Conv2d(16,32,4,2,1)
        self.conv2d5=nn.Conv2d(32,1,1)
        self.batchnomal1=nn.BatchNorm2d(8)
        self.batchnomal2 = nn.BatchNorm2d(16)
        self.batchnomal3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.conv2d1(x)
        x = F.leaky_relu(x)
        x=self.conv2d2(x)
        x=self.batchnomal1(x)
        x=F.leaky_relu(x)
        x=self.conv2d3(x)
        x=self.batchnomal2(x)
        x=F.leaky_relu(x)
        x=self.conv2d4(x)
        x=self.batchnomal3(x)
        x=F.leaky_relu(x)
        x=self.conv2d5(x)
        x=F.sigmoid(x)
        return x


def main():
    parse = argparse.ArgumentParser('My gan trainnig')
    parse.add_argument('--batchsize', type=int, default=128)
    parse.add_argument('--epoch', type=int, default=100)
    parse.add_argument('--nocuda', action='store_false', default=True)
    parse.add_argument('--k', type=int, default=1)
    parse.add_argument('--lrg',type=float,default=0.01)
    parse.add_argument('--lrd',type=float,default=0.00001)
    args = parse.parse_args()
    device = 'cuda' if torch.cuda.is_available() and args.nocuda else 'cpu'
    # load data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x:x/255*5)
                       ])),
        batch_size=args.batchsize, shuffle=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion_D = Dloss()
    criterion_G = GLoss()
    optimizer_g = optim.Adam(generator.parameters())
    optimizer_d=optim.Adam(discriminator.parameters())
    # train
    # for epoch
    count = 0
    lossglist=[]
    lossdlist=[]
    for e in range(args.epoch):
        # sample train data x

        for i,(x, label) in enumerate(train_loader,0):
            # for k
            # sample noise minibatch z
            discriminator.zero_grad()
            z = torch.rand(x.shape[0],1,lzsize,lzsize).to(device)*5
            # update discriminator by sgd
            loss_D = criterion_D(discriminator.forward(x.to(device)), discriminator.forward(generator.forward(z)))
            loss_D.backward()
            lossdlist.append(loss_D)
            optimizer_d.step()

            count += 1
            if count < args.k:
                continue
            count = 0
            # TODO print loss mean

            # sample noise minibatch z by sgd
            z = torch.rand(args.batchsize, 1,lzsize, lzsize).to(device) * 5
            # update generator
            generator.zero_grad()
            loss_G = criterion_G(discriminator.forward(generator.forward(z)))
            loss_G.backward()
            lossglist.append(loss_G)
            optimizer_g.step()
            print("e:{} {:2.1f}% loss_D:{} loss_G:{}".format(e,i*args.batchsize*100/len(train_loader.dataset),loss_D,loss_G))

            #test
            print(
                "t:",torch.mean(discriminator.forward(x.to(device))),
                "f:",torch.mean((torch.rand(x.shape).to(device))),
                "tf",torch.mean(discriminator.forward(generator(torch.rand(x.shape).to(device)*5)))
            )
        if not os.path.exists('output'):
            os.mkdir('output')
        save_image((generator(torch.rand(1, lzsize, lzsize).to(device) * 5)), 'output/' + str(e) + '.png')
    plt.plot(range(len(lossdlist)),lossdlist,label='Loss_D')
    plt.plot(range(len(lossglist)),lossglist,label='Loss_G')
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.legend()
    import datetime
    plt.savefig(str(datetime.datetime.now())+'-loss.png')


if __name__ == '__main__':
    main()
