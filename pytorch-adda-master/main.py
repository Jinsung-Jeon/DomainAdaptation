"""Main script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder, ResNetEncoder
from utils import get_data_loader, init_model, init_random_seed, get_dataset
import os
import torch.nn as nn
import torch
from SSHead import extractor_from_layer3
from parse_tasks import parse_tasks

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader = get_data_loader(params.src_dataset)
    src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    tgt_data_loader = get_data_loader(params.tgt_dataset)
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, split='test')

    #ResNetEncoder = torch.nn.DataParallel(ResNetEncoder(depth=26))
    #LeNetClassifier = torch.nn.DataParallel(LeNetClassifier())
    #Discriminator = torch.nn.DataParallel(Discriminator(input_dims=params.d_input_dims, hidden_dims=params.d_hidden_dims, output_dims=params.d_output_dims))
    src_encoder = init_model(net=ResNetEncoder(depth=26),
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=LeNetClassifier(),
                                restore=params.src_classifier_restore)
    tgt_encoder = init_model(net=ResNetEncoder(depth=26),
                             restore=params.tgt_encoder_restore)
    critic = init_model(Discriminator(input_dims=params.d_input_dims, hidden_dims=params.d_hidden_dims, output_dims=params.d_output_dims),
                        restore=params.d_model_restore)
    ext = init_model(net = extractor_from_layer3(ResNetEncoder(depth=26)),
                     restore = params.extrac_restore)

    sc_tr_dataset = get_dataset(params.src_dataset)
    sc_te_dataset = get_dataset(params.src_dataset, train=False)
    tg_tr_dataset = get_dataset(params.tgt_dataset)
    tg_te_dataset = get_dataset(params.tgt_dataset, split='test')

    sstasks = parse_tasks(params, ext, sc_tr_dataset, sc_te_dataset, tg_tr_dataset, tg_te_dataset)
    # load models
    parameters = list(tgt_encoder.parameters())
    for sstask in sstasks:
        parameters += list(sstask.head.parameters())
    # train source model
    print("=== Training classifier for source domain ===")

    if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):
        src_encoder, src_classifier = train_src(
            src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())
        src_classifier.load_state_dict(src_classifier.state_dict())

    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        train_tgt(tgt_encoder, src_classifier, critic, src_data_loader, tgt_data_loader, tgt_data_loader_eval,eval_tgt,src_data_loader_eval, ext, scheduler,sstasks)
    '''
    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
    '''