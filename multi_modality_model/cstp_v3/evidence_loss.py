import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import digamma, gamma
import math
#from config import loss, threshold_kl

threshold_kl = 30
loss = 'ce'
def relu_evidence(y):
    # return F.softplus(y)
    # return torch.exp(y)
    return F.elu(y) + 1


def kl(alpha, num_classes):
    ones = torch.ones([1, num_classes], dtype=torch.float32).cuda()
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        torch.sum((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha)), dim=1, keepdim=True)
    )
    kl = first_term + second_term

    return kl


def loss_function(logits, p, global_step, W, c, annealing_step, kl_coefficient):
    if loss == 'ce':
        A, B = ce_loss(logits, p, global_step, W, c, annealing_step, kl_coefficient)
    elif loss == 'likelihood':
        A, B = likelihood_loss(logits, p, global_step, W, c, annealing_step, kl_coefficient)
    elif loss == 'mse':
        A, B = mse_loss(logits, p, global_step, W, c, annealing_step, kl_coefficient)
    return A, B


def ce_loss(logits, p, global_step=100, W=128, c=128, annealing_step=500, kl_coefficient=1):
    evidence = relu_evidence(logits)
    #print("evidence shape", evidence.shape)
    alpha = evidence + W/c

    S = torch.sum(alpha, dim=1, keepdim=True)
    #print("S", S.shape)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    #print("torch.digamma(S)",torch.digamma(S).shape)
    #print("orch.digamma(alpha)",torch.digamma(alpha).shape)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    if global_step >= threshold_kl:
        annealing_coef = min(kl_coefficient, (global_step-threshold_kl+1) / annealing_step)
    else:
        annealing_coef = 0.0

    alp = E * (1 - label) + 1
    B = annealing_coef * kl(alp, c)
    #ce_loss_value = (A+B).squeeze()
    ce_loss_mean = (A+B).mean()
    return ce_loss_mean


def likelihood_loss(logits, p, global_step, W, c, annealing_step, kl_coefficient):
    evidence = relu_evidence(logits)
    alpha = evidence + W/c

    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)

    if global_step >= threshold_kl:
        annealing_coef = min(kl_coefficient, (global_step-threshold_kl+1) / annealing_step)
    else:
        annealing_coef = 0.0

    alp = E * (1 - label) + 1
    B = annealing_coef * kl(alp, c)

    return A, B


def mse_loss(logits, p, global_step, W, c, annealing_step, kl_coefficient):
    evidence = relu_evidence(logits)
    alpha = evidence + W/c
   
    S = torch.sum(alpha, dim=1, keepdim=True)  # 沿着行求和
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    loglike_err = torch.sum((label - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglike_var = torch.sum((alpha * (S - alpha)) / (S * S * (S + 1)), dim=1, keepdim=True)
    A = loglike_err + loglike_var
   
    if global_step >= threshold_kl:
        annealing_coef = min(kl_coefficient, (global_step-threshold_kl+1) / annealing_step)
    else:
        annealing_coef = 0.0

    alp = E * (1 - label) + 1
    B = annealing_coef * kl(alp, c)

    return A, B


def loss_function_eval(evidence, p, global_step, W, c, annealing_step, kl_coefficient):
    if loss == 'ce':
        A, B = ce_loss_eval(evidence, p, global_step, W, c, annealing_step, kl_coefficient)
    elif loss == 'likelihood':
        A, B = likelihood_loss_eval(evidence, p, global_step, W, c, annealing_step, kl_coefficient)
    elif loss == 'mse':
        A, B = mse_loss_eval(evidence, p, global_step, W, c, annealing_step, kl_coefficient)
    return A, B


def ce_loss_eval(evidence, p, global_step, W, c, annealing_step, kl_coefficient):
    # evidence = relu_evidence(logits)
    alpha = evidence + W/c

    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(kl_coefficient, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * kl(alp, c)

    return A, B


def likelihood_loss_eval(evidence, p, global_step, W, c, annealing_step, kl_coefficient):
    # evidence = relu_evidence(logits)
    alpha = evidence + W/c

    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)

    annealing_coef = min(kl_coefficient, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * kl(alp, c)

    return A, B


def mse_loss_eval(evidence, p, global_step, W, c, annealing_step, kl_coefficient):
    # evidence = relu_evidence(logits)
    alpha = evidence + W/c
   
    S = torch.sum(alpha, dim=1, keepdim=True)  # 沿着行求和
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    loglike_err = torch.sum((label - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglike_var = torch.sum((alpha * (S - alpha)) / (S * S * (S + 1)), dim=1, keepdim=True)
    A = loglike_err + loglike_var
   
    annealing_coef = min(kl_coefficient, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * kl(alp, c)

    return A, B