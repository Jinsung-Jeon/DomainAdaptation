import argparse
import os
from dataloader import get_svhn, get_mnist
from models import Generator, Discriminator, Classifier
from utils import weights_init_normal
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=96, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the noise input")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes in the dataset")
parser.add_argument("--sample_interval", type=int, default=300, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'

lambda_adv = 1
lambda_task = 0.1

# Calculate output of image discriminator (PatchGAN)
patch = int(opt.img_size / (2 ** 4))
patch = (1, patch, patch)


generator = Generator(opt.latent_dim, opt.channels, opt.img_size, opt.n_residual_blocks)
discriminator = Discriminator(opt.channels)
classifier = Classifier(opt.channels, opt.img_size, opt.n_classes)

generator = nn.DataParallel(generator)
generator.cuda()
discriminator = nn.DataParallel(discriminator)
discriminator.cuda()
classifier = nn.DataParallel(classifier)
classifier.cuda()

adversarial_loss = torch.nn.MSELoss().cuda()
task_loss = torch.nn.CrossEntropyLoss().cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
classifier.apply(weights_init_normal)

os.makedirs("data", exist_ok=True)

train_source = get_mnist(train=True)
train_target = get_svhn(split='train')

optimizer_G = torch.optim.Adam(
    itertools.chain(generator.parameters(), classifier.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------

# Keeps 100 accuracy measurements
task_performance = []
target_performance = []

for epoch in range(opt.n_epochs):
    for i, ((imgs_A, labels_A), (imgs_B, labels_B)) in enumerate(zip(train_source, train_target)):

        batch_size = 96

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, *patch).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, *patch).fill_(0.0), requires_grad=False)

        # Configure input
        imgs_A = Variable(imgs_A.type(FloatTensor).expand(batch_size, 3, opt.img_size, opt.img_size))
        labels_A = Variable(labels_A.type(LongTensor))
        imgs_B = Variable(imgs_B.type(FloatTensor))

        optimizer_G.zero_grad()

        # Sample noise
        z = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.latent_dim))))

        # Generate a batch of images
        fake_B = generator(imgs_A, z)

        # Perform task on translated source image
        label_pred = classifier(fake_B)

        # Calculate the task loss
        task_loss_ = (task_loss(label_pred, labels_A) + task_loss(classifier(imgs_A), labels_A)) / 2

        # Loss measures generator's ability to fool the discriminator
        g_loss = lambda_adv * adversarial_loss(discriminator(fake_B), valid) + lambda_task * task_loss_

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        print(len(valid))
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(imgs_B), valid)
        fake_loss = adversarial_loss(discriminator(fake_B.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        

        d_loss.backward()
        optimizer_D.step()

        # ---------------------------------------
        #  Evaluate Performance on target domain
        # ---------------------------------------

        # Evaluate performance on translated Domain A
        acc = np.mean(np.argmax(label_pred.data.cpu().numpy(), axis=1) == labels_A.data.cpu().numpy())
        task_performance.append(acc)
        if len(task_performance) > 100:
            task_performance.pop(0)

        # Evaluate performance on Domain B
        pred_B = classifier(imgs_B)
        target_acc = np.mean(np.argmax(pred_B.data.cpu().numpy(), axis=1) == labels_B.numpy())
        target_performance.append(target_acc)
        if len(target_performance) > 100:
            target_performance.pop(0)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [CLF acc: %3d%% (%3d%%), target_acc: %3d%% (%3d%%)]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_source),
                d_loss.item(),
                g_loss.item(),
                100 * acc,
                100 * np.mean(task_performance),
                100 * target_acc,
                100 * np.mean(target_performance),
            )
        )

        batches_done = len(train_source) * epoch + i
        if batches_done % opt.sample_interval == 0:
            sample = torch.cat((imgs_A.data[:5], fake_B.data[:5], imgs_B.data[:5]), -2)
            save_image(sample, "images/%d.png" % batches_done, nrow=int(math.sqrt(batch_size)), normalize=True)
