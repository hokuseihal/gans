import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.utils import save_image

import argparse
import matplotlib.pyplot as plt
from maxout import Maxout

imagesize = 28

class Dloss(nn.Module):
    def __init__(self):
        super(Dloss, self).__init__()

    def forward(self, x, z):
        if x.shape != z.shape:
            raise ArithmeticError('Sizes input to DLoss are not same.')
        #if torch.max(x) >= 1 or torch.max(z) >= 1: raise ArithmeticError('Is it true D returns probability?')
        x = x.view(-1, 1)
        z = z.view(-1, 1)
        return torch.sum(torch.log(x) + torch.log(1 - z)) / len(x)


class GLoss(nn.Module):
    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, x):
        #if torch.max(z) >= 1: raise ArithmeticError('Is it true D returns probability?')
        x = x.view(-1, 1)
        return torch.sum(torch.log(x)) / len(x)*10


# network generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(imagesize ** 2, 32 ** 2)
        self.fc2 = nn.Linear(32 ** 2, 32 ** 2)
        self.fc3 = nn.Linear(32 ** 2, imagesize ** 2)

    def forward(self, x):
        x = x.view(-1, imagesize ** 2)
        x = F.relu(F.dropout(self.fc1(x)))
        x = F.relu(F.dropout(self.fc2(x)))
        x = F.sigmoid(self.fc3(x))*5
        x = x.view(-1, imagesize, imagesize)
        return x


# network of discriminator
# TODO USE MAXOUT AND DROPOUT
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc4 = nn.Linear(imagesize ** 2, 16 ** 2)
        self.fc5 = nn.Linear(128, 8 ** 2)
        self.fc6 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(-1, imagesize ** 2)
        x = Maxout(2)(F.dropout(self.fc4(x)))
        x = Maxout(2)(F.dropout(self.fc5(x)))
        x = F.sigmoid(self.fc6(x))
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
    #optimizer_g = optim.Adam(generator.parameters())
    #optimizer_d=optim.Adam(discriminator.parameters())
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
            z = torch.rand(x.shape).to(device)*5
            # update discriminator by sgd
            loss_D = criterion_D(discriminator.forward(x.to(device)), discriminator.forward(generator.forward(z)))
            loss_D.backward()
            lossdlist.append(loss_D)
            #optimizer_d.step()
            for param in discriminator.parameters():
                param.data+=args.lrd*param.grad.data

            count += 1
            if count <= args.k:
                continue
            count = 0
            # TODO print loss mean

            # sample noise minibatch z by sgd
            z = torch.rand(args.batchsize, imagesize, imagesize).to(device)*5
            # update generator
            generator.zero_grad()
            loss_G = criterion_G(discriminator.forward(generator.forward(z)))
            loss_G.backward()
            lossglist.append(loss_G)
            #optimizer_g.step()
            for param in generator.parameters():
                param.data+=args.lrg*param.grad.data
            print("e:{} {:2.1f}% loss_D:{} loss_G:{}".format(e,i*args.batchsize*100/len(train_loader.dataset),loss_D,loss_G))

            #test
            print(
                "t:",torch.mean(discriminator.forward(x.to(device))),
                "f:",torch.mean((torch.rand(x.shape).to(device))),
                "tf",torch.mean(discriminator.forward(generator(torch.rand(x.shape).to(device)*5)))
            )
        if not os.path.exists('output'):
            os.mkdir('output')
        save_image((generator(torch.rand(1,imagesize,imagesize).to(device)*5)),'output/'+str(e)+'.png')
    plt.plot(range(len(lossdlist)),lossdlist,label='Loss_D')
    plt.plot(range(len(lossglist)),lossglist,label='Loss_G')
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.legend()
    import datetime
    plt.savefig(str(datetime.datetime.now())+'-loss.png')


if __name__ == '__main__':
    main()
