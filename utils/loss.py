import torch.nn.functional as F
import torch.nn as nn

def loss_fn_kd(outputs, teacher_outputs, args):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = args.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=-1),
                             F.softmax(teacher_outputs/T, dim=-1)) * (T * T)
    return KD_loss