import argparse
import datetime
import os

import matplotlib as mpl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torchvision import transforms
from torchvision.utils import save_image

mpl.use('Agg')
import matplotlib.pyplot as plt

foldername = str(datetime.datetime.now()) + 'output'
if not os.path.exists(foldername):
    os.mkdir(foldername)

parse = argparse.ArgumentParser('My gan trainnig')
parse.add_argument('--batchsize', type=int, default=128)
parse.add_argument('--epoch', type=int, default=50)
parse.add_argument('--nocuda', action='store_false', default=True)
parse.add_argument('--k', type=int, default=5)
parse.add_argument('--lrg', type=float, default=0.001)
parse.add_argument('--lrd', type=float, default=0.0001)
parse.add_argument('--dataset', help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake', default='mnist')
parse.add_argument('--dataroot', type=str, default='dataset')
parse.add_argument('--tmark',type=float,default=0.75)
parse.add_argument('--tfgmark',type=float,default=0.75)
args = parse.parse_args()

###init
imagesize = 64
if args.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(imagesize),
                                   transforms.CenterCrop(imagesize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc = 3
elif args.dataset == 'lsun':
    dataset = dset.LSUN(root=args.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(imagesize),
                            transforms.CenterCrop(imagesize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc = 3
elif args.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(imagesize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc = 3

elif args.dataset == 'mnist':
    dataset = dset.MNIST(root=args.dataroot, download=True,
                         transform=transforms.Compose([
                             transforms.Resize(imagesize),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,)),
                         ]))
    nc = 1

lzsize = 1
nz = 100
ngf = 64
ndf = 64
zshape = (args.batchsize, nz, lzsize, lzsize)
xshape = (args.batchsize, nc, imagesize, imagesize)
device = 'cuda' if torch.cuda.is_available() and args.nocuda else 'cpu'


# classes
# define errors
class Dloss(nn.Module):
    def __init__(self):
        super(Dloss, self).__init__()

    def forward(self, x, z):
        x = x.view(-1, 1)
        return -torch.mean(torch.log(x) + torch.log(1 - z))


class GLoss(nn.Module):
    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, x):
        x = x.view(-1, 1)
        return -torch.mean(torch.log(x))


# network of generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        x = self.main(x)
        return x


# network of discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x

def main():
    # load data
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    # main init
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion_D = Dloss()
    criterion_G = GLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    lossglist = []
    lossdlist = []
    tlist = []
    flist = []
    tflist = []
    tmark = args.tmark
    tfgmark = args.tfgmark
    s = 0
    loss_G = torch.tensor(0)

    # train
    for e in range(args.epoch):
        for i, (x, label) in enumerate(train_loader, 0):
            x = x.to(device)
            tfg = 0

            # train dicriminator
            s += 1
            discriminator.zero_grad()
            z = torch.rand(zshape).to(device)
            loss_D = criterion_D(discriminator(x), discriminator(generator(z)))
            loss_D.backward()
            optimizer_d.step()
            t = torch.mean(discriminator(x.to(device))).item()
            print("lossD:%4f t:%4f" % (loss_D.item(), t))
            lossglist.append(loss_G.item())
            lossdlist.append(loss_D.item())
            if t < tmark: continue
            tf = torch.mean(discriminator(generator(z))).item()
            # train generator
            while tfg < tfgmark:
                s += 1
                generator.zero_grad()
                z = torch.rand(args.batchsize, nz, 1, 1).to(device)
                loss_G = criterion_G(discriminator(generator(z)))
                loss_G.backward()
                optimizer_g.step()
                z = torch.rand(args.batchsize, nz, 1, 1).to(device)
                tfg = torch.mean(discriminator(generator(z))).item()
                print("lossG:%4f tf:%4f" % (loss_G.item(), tfg))
                lossglist.append(loss_G.item())
                lossdlist.append(loss_D.item())

            # score
            print("e:%d s:%d | %d/%d loss_D:%4f loss_G:%4f" % (
                e, s, i * args.batchsize, len(train_loader.dataset), loss_D, loss_G))
            t = torch.mean(discriminator(x.to(device))).item()
            tlist.append(t)
            f = torch.mean(discriminator(torch.rand(xshape).to(device))).item()
            flist.append(f)
            
            tflist.append(tf)
            # test
            print("t:%4f f:%4f tf%4f->%4f" % (t, f, tfg, tf))

            # stop if nan
            if loss_G != loss_G: exit(1)
            # save generate images
            save_image((generator(torch.rand(64, nz, lzsize, lzsize).to(device))).detach(),
                       foldername + '/' + str(s) + '.png',
                       normalize=True)

    # draw graph
    #fig = plt.figure()
    #ax1 = fig.add_subplot(211)
    #ax1.plot(range(len(tlist)), tlist, label='t')
    #ax1.plot(range(len(flist)), flist, label='f')
    #ax1.plot(range(len(tflist)), tflist, label='tf')
    #ax2 = fig.add_subplot(211)
    plt.plot(range(len(lossglist)), lossglist, label='Loss_G')
    plt.plot(range(len(lossdlist)), lossdlist, label='Loss_D')
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(foldername+'/'+'loss.png')


if __name__ == '__main__':
    main()
