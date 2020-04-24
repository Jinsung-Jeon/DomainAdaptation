"""Adversarial adaptation to train target encoder."""

import os
import numpy as nn
import torch
import torch.optim as optim
from torch import nn
import params
from plot_all_epoch_stats import plot_all_epoch_stats
from utils import make_variable

def train_tgt(tgt_encoder, src_classifier, critic, src_data_loader, tgt_data_loader, tgt_data_loader_eval, src_data_loader_eval, eval_tgt):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    src_classifier.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_critic_c = optim.Adam(list(tgt_encoder.parameters()) + list(src_classifier.parameters()), lr=params.d_learning_rate, betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(list(tgt_encoder.parameters()) + list(src_classifier.parameters()) + list(critic.parameters()), lr=params.d_learning_rate, betas=(params.beta1, params.beta2))

    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))
    print("loader done!")
    ####################
    # 2. train network #
    ####################
    all_epoch_stats = []
    for epoch in range(params.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        epoch_stats = []
        for step, ((images_src, images_src_labels), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################
            p = float(step + epoch * len_dataloader) / params.num_epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # make images variable
            images_src = make_variable(images_src)
            images_src_labels = make_variable(images_src_labels)
            images_tgt = make_variable(images_tgt)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_critic_c.zero_grad()

            # extract and concat features
            feat_src = src_classifier(tgt_encoder(images_src))
            loss_src = criterion(feat_src, images_src_labels)
            loss_src.backward()
            optimizer_critic_c.step()
            feat_tgt = src_classifier(tgt_encoder(images_tgt))
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            
            # predict on discriminator
            pred_concat = critic(feat_concat.detach(), alpha)

            # prepare real and fake label
            label_src = make_variable(torch.ones(feat_src.size(0)).long())
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()
            ############################
            # 2.2 train target encoder #
            ############################
            '''
            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            
            # extract and target features
            feat_tgt = src_classifier(tgt_encoder(images_tgt))

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()
            optimizer_critic.step()
            '''
            #######################
            # 2.3 print step info #
            #######################
            if ((step+1) == 230):
            #if ((step + 1) % params.log_step == 0):
                tot_loss, acc = eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
                tot_loss_s, acc_s = eval_tgt(tgt_encoder, src_classifier, src_data_loader_eval)
                epoch_stats.append((step, len_data_loader, tot_loss, acc))
                print("Epoch [{}/{}] Step [{}/{}]:t_loss={:.5f}  acc={:.5f}".format(epoch + 1,params.num_epochs,step + 1,len_data_loader,loss_critic,acc))
                print("Epoch [{}/{}] Step [{}/{}]:t_loss={:.5f}  acc={:.5f}".format(epoch + 1,params.num_epochs,step + 1,len_data_loader,tot_loss_s,acc_s))
        all_epoch_stats.append(epoch_stats)
        plot_all_epoch_stats(all_epoch_stats, params.outf)
        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % params.save_step == 0):
            torch.save(critic.state_dict(), os.path.join(
                params.model_root,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                "ADDA-target-encoder-{}.pt".format(epoch + 1)))

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "ADDA-critic-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        params.model_root,
        "ADDA-target-encoder-final.pt"))
