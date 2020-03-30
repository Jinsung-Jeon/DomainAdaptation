"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable


def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0
    tot_loss = 0
    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    with torch.no_grad():
        for (images, labels) in data_loader:
            images = make_variable(images)
            labels = make_variable(labels).squeeze_()

            preds = classifier(encoder(images))
            loss = criterion(preds, labels).item()
            tot_loss += loss
            _, pred_cls = torch.max(preds, 1)
            #pred_cls = preds.item.max(1)[1]
            #acc = pred_cls.eq(labels.data).cpu().sum()
            acc += (pred_cls == labels).sum().float()

        tot_loss /= len(data_loader)
        acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(tot_loss, acc))
