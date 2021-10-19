# -*- coding: utf-8 -*-
# import sys
# sys.path.append("..")
import argparse
import numpy as np
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Function
import time
import os
from gdn_v3 import GDN, IGDN
import torchvision.transforms as transforms
from tqdm import tqdm
from network_swinir import RSTB,PatchEmbed,PatchUnEmbed
from pytorch_msssim import ms_ssim as compute_ms_ssim
import matplotlib.pyplot as plt
from PIL import Image

def checkType(X):
    if isinstance(X, (np.ndarray, list, tuple)):
        return torch.FloatTensor(X)
    elif isinstance(X, (torch.TensorType, torch.FloatType)):
        return X
    else:
        return X
        print("Type Error")


class Residualblock(nn.Module):
    """Builds the residual block"""
    def __init__(self, num_filters):
        super(Residualblock, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(num_filters, num_filters//2, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num_filters // 2, num_filters//2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num_filters//2, num_filters, 1, 1),
            )

    def forward(self, inputs):
        x = self.transform(inputs)
        x = x+inputs
        return x


class NonLocalAttentionBlock(nn.Module):
    """Builds the non-local attention block"""
    def __init__(self, num_filters):
        super(NonLocalAttentionBlock, self).__init__()
        self.trunk_branch = nn.Sequential(
            Residualblock(num_filters),
            Residualblock(num_filters),
            Residualblock(num_filters)
            )
        self.attention_branch = nn.Sequential(
            Residualblock(num_filters),
            Residualblock(num_filters),
            Residualblock(num_filters)
        )
        self.conv = nn.Conv2d(num_filters, num_filters, 1, 1)

    def forward(self, inputs):
        trunk_branch = self.trunk_branch(inputs)
        attention_branch = self.attention_branch(inputs)
        attention_branch = self.conv(attention_branch)
        attention_branch = nn.Sigmoid()(attention_branch)
        x = inputs + torch.mul(attention_branch, trunk_branch)
        return x

class ConcatenatedResidualModule(nn.Module):
    def __init__(self, num_filters):
        super(ConcatenatedResidualModule, self).__init__()
        self.block0 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.LeakyReLU(),
        )

    def forward(self,inputs):
        outputs0 = self.block0(inputs)
        outputs0 = inputs + outputs0
        outputs1 = self.block1(outputs0)
        outputs = outputs0+outputs1+inputs
        return outputs

# Main analysis transform model with GDN
class analysisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters,windowsize,depths=[8,8,8,8],num_heads=[8,8,8,8]):
        super(analysisTransformModel, self).__init__()
        self.shortcut0 = nn.Conv2d(in_dim, num_filters, 1, 2)
        self.conv_down_sample_0 = nn.Sequential(
            nn.Conv2d(in_dim, num_filters, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            GDN(num_filters)
            )
        self.CRM0 = ConcatenatedResidualModule(num_filters)
        self.shortcut1 = nn.Conv2d(num_filters, num_filters, 1, 2)
        self.conv_down_sample_1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            GDN(num_filters)
        )
        self.RSTB_0_0 = RSTB(dim=num_filters, input_resolution=(24, 24), depth=depths[0], num_heads=num_heads[0],
                             window_size=windowsize,
                             mlp_ratio=2, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                             drop_path=0, norm_layer=nn.LayerNorm, downsample=None,
                             use_checkpoint=False, patch_size=1, resi_connection='3conv')
        # self.Attention0 = NonLocalAttentionBlock(num_filters)
        self.CRM1 = ConcatenatedResidualModule(num_filters)
        self.shortcut2 = nn.Conv2d(num_filters, num_filters, 1, 2)
        self.conv_down_sample_2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            GDN(num_filters)
        )
        self.CRM2 = ConcatenatedResidualModule(num_filters)
        self.conv_down_sample_3 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, 2, 1),
            GDN(num_filters)
        )
        self.RSTB_2_0 = RSTB(dim=num_filters, input_resolution=(96, 96), depth=depths[2], num_heads=num_heads[2],
                             window_size=windowsize,
                             mlp_ratio=2, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                             drop_path=0, norm_layer=nn.LayerNorm, downsample=None,
                             use_checkpoint=False, patch_size=1, resi_connection='3conv')
        # self.Attention0 = NonLocalAttentionBlock(num_filters)

    def forward(self, x):
        shortcut0 = self.shortcut0(x)
        x = self.conv_down_sample_0(x)
        x += shortcut0
        x = self.CRM0(x)
        shortcut1 = self.shortcut1(x)
        x = self.conv_down_sample_1(x)
        x += shortcut1

        x_size = (x.shape[2], x.shape[3])
        x = self.RSTB_0_0(x, x_size)
        x = self.CRM1(x)
        shortcut2 = self.shortcut2(x)
        x = self.conv_down_sample_2(x)
        x += shortcut2
        x = self.CRM2(x)
        x = self.conv_down_sample_3(x)
        x_size = (x.shape[2], x.shape[3])
        x = self.RSTB_2_0(x, x_size)

        return x

# Main synthesis transform model with IGDN
class synthesisTransformModel(nn.Module):
    def __init__(self, num_filters,windowsize,depths=[8,8,8,8],num_heads=[8,8,8,8]):
        super(synthesisTransformModel, self).__init__()
        self.RSTB_0_0 = RSTB(dim=num_filters, input_resolution=(24,24), depth=depths[0], num_heads=num_heads[0], window_size=windowsize,
                 mlp_ratio=2, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0, norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, patch_size=1, resi_connection='3conv')
        self.CRM0 = ConcatenatedResidualModule(num_filters)
        self.shortcut0 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*4, 1, 1),
            Depth2Space(2),
        )
        self.conv_up_sample_0 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*4, 3, 1, 1),
            nn.LeakyReLU(),
            Depth2Space(2),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            IGDN(num_filters)
        )
        self.CRM1 = ConcatenatedResidualModule(num_filters)
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*4, 1, 1),
            Depth2Space(2),
        )
        self.conv_up_sample_1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 4, 3, 1, 1),
            nn.LeakyReLU(),
            Depth2Space(2),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            IGDN(num_filters)
        )
        self.RSTB_2_0 = RSTB(dim=num_filters, input_resolution=(96, 96), depth=depths[2], num_heads=num_heads[2], window_size=windowsize,
                           mlp_ratio=2, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                           drop_path=0, norm_layer=nn.LayerNorm, downsample=None,
                           use_checkpoint=False, patch_size=1, resi_connection='3conv')

        self.CRM2 = ConcatenatedResidualModule(num_filters)
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 4, 1, 1),
            Depth2Space(2),
        )
        self.conv_up_sample_2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 4, 3, 1, 1),
            nn.LeakyReLU(),
            Depth2Space(2),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            IGDN(num_filters)
        )

        self.CRM3 = ConcatenatedResidualModule(num_filters)
        self.conv_up_sample_3 = nn.Sequential(
            nn.Conv2d(num_filters, 3 * 4, 3, 1, 1),
            Depth2Space(2),
        )

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.RSTB_0_0(x, x_size)
        x = self.CRM0(x)
        shortcut0 = self.shortcut0(x)
        x = self.conv_up_sample_0(x)
        x += shortcut0

        x = self.CRM1(x)
        shortcut1 = self.shortcut1(x)
        x = self.conv_up_sample_1(x)
        x += shortcut1

        x_size = (x.shape[2], x.shape[3])
        x = self.RSTB_2_0(x, x_size)
        x = self.CRM2(x)
        shortcut2 = self.shortcut2(x)
        x = self.conv_up_sample_2(x)
        x += shortcut2

        x = self.CRM3(x)
        x = self.conv_up_sample_3(x)
        return x


# Space-to-depth & depth-to-space module
# same to TensorFlow implementations
class Space2Depth(nn.Module):
    def __init__(self, r):
        super(Space2Depth, self).__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c * (r**2)
        out_h = h//2
        out_w = w//2
        x_view = x.view(b, c, out_h, r, out_w, r)
        x_prime = x_view.permute(0, 3, 5, 1, 2, 4).contiguous().view(
            b, out_c, out_h, out_w)
        return x_prime

class Depth2Space(nn.Module):
    def __init__(self, r):
        super(Depth2Space, self).__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c // (r**2)
        out_h = h * 2
        out_w = w * 2
        x_view = x.view(b, r, r, out_c, h, w)
        x_prime = x_view.permute(0, 3, 4, 1, 5, 2).contiguous().view(
            b, out_c, out_h, out_w)
        return x_prime

# Hyper analysis transform (w/o GDN)
class h_analysisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters):
        super(h_analysisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_dim, num_filters, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters, num_filters, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters, num_filters, 3, 2, 1),
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x

# Hyper synthesis transform (w/o GDN)
class h_synthesisTransformModel(nn.Module):
    def __init__(self, num_filters):
        super(h_synthesisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters, int(1.5*num_filters), 3, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(int(1.5*num_filters), int(1.5*num_filters), kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(int(1.5*num_filters), int(2 * num_filters), 3, 1, 1),
            nn.LeakyReLU(),
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x

# Hyper synthesis transform (w/o GDN)
class entropy_parameter(nn.Module):
    def __init__(self, num_filters):
        super(entropy_parameter, self).__init__()
        self.num_filters = num_filters
        self.transform = nn.Sequential(
            nn.Conv2d(4 * num_filters, 640, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(640, 640 * 2, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(640 * 2, num_filters * 30, 1, 1),
        )
        self.masked = MaskedConv2d("A", in_channels=num_filters
                                   , out_channels=num_filters*2,
                                   kernel_size=5, stride=1,
                                   padding=2)

    def forward(self, y_hat,phi):
        context_info = self.masked(y_hat)
        x = torch.cat([phi, context_info], dim=1)
        x = self.transform(x)
        prob0, mean0, scale0, prob1, mean1, scale1,\
        prob2, mean2, scale2, prob3, mean3, scale3,\
        prob4, mean4, scale4, prob5, mean5, scale5,\
        prob6, mean6, scale6, prob7, mean7, scale7,\
        prob8, mean8, scale8, prob_m0, prob_m1, prob_m2 = \
            torch.split(x, split_size_or_sections=self.num_filters, dim=1)
        scale0 = torch.abs(scale0)
        ...
        scale8 = torch.abs(scale8)
        softmax = torch.nn.Softmax(dim=-1)
        probs = torch.stack([prob0, prob1, prob2], dim=-1)
        probs = softmax(probs)
        probs_lap = torch.stack([prob3, prob4, prob5], dim=-1)
        probs_lap = softmax(probs_lap)
        probs_log = torch.stack([prob6, prob7, prob8], dim=-1)
        probs_log = softmax(probs_log)
        probs_mix = torch.stack([prob_m0, prob_m1, prob_m2], dim=-1)
        probs_mix = softmax(probs_mix)
        # To merge them together
        means = torch.stack([mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8], dim=-1)
        variances = torch.stack([scale0, scale1, scale2, scale3, scale4, scale5, scale6, scale7, scale8], dim=-1)

        return means,variances,probs,probs_lap,probs_log,probs_mix

class MaskedConv2d(nn.Conv2d):
    '''
    Implementation of the Masked convolution from the paper
    Van den Oord, Aaron, et al. "Conditional image generation with pixelcnn decoders." Advances in neural information processing systems. 2016.
    https://arxiv.org/pdf/1606.05328.pdf
    '''

    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ('A', 'B')
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)



class Bitparm(nn.Module):
    '''
    save params
    '''
    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)

class BitEstimator(nn.Module):
    '''
    Estimate bit
    '''

    def __init__(self, num_filters):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(num_filters)
        self.f2 = Bitparm(num_filters)
        self.f3 = Bitparm(num_filters)
        self.f4 = Bitparm(num_filters, True)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)

class RateDistortionLoss(nn.Module):
    def __init__(self):
        super(RateDistortionLoss, self).__init__()

    def cumulative_normal(self, x,mu,sigma):
        """
        Calculates CDF of Normal distribution with parameters mu and sigma at point x
        """
        half = 0.5
        const = (2 ** 0.5)
        return half * (1 + torch.erf((x - mu) / (const * sigma)))

    def log_cdf_laplace(self,x):

        lower_solution = torch.log(torch.Tensor([.5]).cuda()) + x
        safe_exp_neg_x = torch.exp(-torch.abs(x))
        upper_solution = torch.log1p(-0.5 * safe_exp_neg_x)

        return torch.where(x < 0., lower_solution, upper_solution)

    def cumulative_laplace(self, x,mu,sigma):
        """
        Calculates CDF of Laplace distribution with parameters mu and sigma at point x
        """
        x = self.log_cdf_laplace(x)
        return 0.5 - 0.5 * (x - mu).sign() * torch.expm1(-(x - mu).abs() / (sigma))

    def cumulative_logistic(self, x,mu,sigma):
        """
        Calculates CDF of Logistic distribution with parameters mu and sigma at point x
        """
        return torch.sigmoid((x - mu) / sigma)

    def latent_rate(self,y_hat,mu, sigma, probs, probs_lap, probs_log, probs_mix,mode='train'):
        sigma = sigma.clamp(1e-10, 1e10)
        half = 0.5
        likelihoods_0 = self.cumulative_normal(y_hat + half, mu[...,0], sigma[...,0]) - self.cumulative_normal(y_hat - half,
                                                                                                          mu[...,0],
                                                                                                          sigma[...,0])
        likelihoods_1 = self.cumulative_normal(y_hat + half, mu[...,1], sigma[...,1]) - self.cumulative_normal(y_hat - half,
                                                                                                          mu[...,1],
                                                                                                          sigma[...,1])
        likelihoods_2 = self.cumulative_normal(y_hat + half, mu[...,2], sigma[...,2]) - self.cumulative_normal(y_hat - half,
                                                                                                          mu[...,2],
                                                                                                          sigma[...,2])
        likelihoods_3 = self.cumulative_laplace(y_hat + half, mu[...,3], sigma[...,3]) - self.cumulative_laplace(
            y_hat - half, mu[...,3], sigma[...,3])
        likelihoods_4 = self.cumulative_laplace(y_hat + half, mu[...,4], sigma[...,4]) - self.cumulative_laplace(
            y_hat - half, mu[...,4], sigma[...,4])
        likelihoods_5 = self.cumulative_laplace(y_hat + half, mu[...,5], sigma[...,5]) - self.cumulative_laplace(
            y_hat - half, mu[...,5], sigma[...,5])
        likelihoods_6 = self.cumulative_logistic(y_hat + half, mu[...,6], sigma[...,6]) - self.cumulative_logistic(
            y_hat - half, mu[...,6], sigma[...,6])
        likelihoods_7 = self.cumulative_logistic(y_hat + half, mu[...,7], sigma[...,7]) - self.cumulative_logistic(
            y_hat - half, mu[...,7], sigma[...,7])
        likelihoods_8 = self.cumulative_logistic(y_hat + half, mu[...,8], sigma[...,8]) - self.cumulative_logistic(
            y_hat - half, mu[...,8], sigma[...,8])
        likelihoods = probs_mix[..., 0] * (probs[..., 0] * likelihoods_0 + probs[..., 1] * likelihoods_1 +
                                           probs[...,2] * likelihoods_2) + \
                      probs_mix[..., 1] * (probs_lap[..., 0] * likelihoods_3 + probs_lap[...,1] * likelihoods_4 +
                                           probs_lap[...,2] * likelihoods_5) + \
                      probs_mix[..., 2] * (probs_log[..., 0] * likelihoods_6 + probs_log[...,1] * likelihoods_7 +
                                           probs_log[...,2] * likelihoods_8)
        # =======REVISION: Robust version ==========
        edge_min = probs_mix[..., 0] * (
                       probs[..., 0] * self.cumulative_normal(y_hat + half, mu[...,0], sigma[...,0]) +
                       probs[..., 1] * self.cumulative_normal(y_hat + half, mu[...,1], sigma[...,1]) +
                       probs[..., 2] * self.cumulative_normal(y_hat + half, mu[...,2], sigma[...,2])) + \
                   probs_mix[..., 1] * (
                       probs_lap[..., 0] * self.cumulative_laplace(y_hat + half, mu[...,3], sigma[...,3]) +
                       probs_lap[..., 1] * self.cumulative_laplace(y_hat + half, mu[...,4], sigma[...,4]) +
                       probs_lap[..., 2] * self.cumulative_laplace(y_hat + half, mu[...,5], sigma[...,5])) + \
                   probs_mix[..., 2] * (
                       probs_log[..., 0] * self.cumulative_logistic(y_hat + half, mu[...,6], sigma[...,6]) +
                       probs_log[..., 1] * self.cumulative_logistic(y_hat + half, mu[...,7], sigma[...,7]) +
                       probs_log[..., 2] * self.cumulative_logistic(y_hat + half, mu[...,8], sigma[...,8]))

        edge_max = probs_mix[..., 0] * (
                       probs[..., 0] * (1.0 - self.cumulative_normal(y_hat - half, mu[...,0], sigma[...,0])) +
                       probs[..., 1] * (1.0 - self.cumulative_normal(y_hat - half, mu[...,1], sigma[...,1])) +
                       probs[..., 2] * (1.0 - self.cumulative_normal(y_hat - half, mu[...,2], sigma[...,2]))) + \
                   probs_mix[..., 1] * (
                       probs_lap[..., 0] * (1.0 - self.cumulative_laplace(y_hat - half, mu[...,3], sigma[...,3])) +
                       probs_lap[..., 1] * (1.0 - self.cumulative_laplace(y_hat - half, mu[...,4],sigma[...,4])) +
                       probs_lap[..., 2] * (1.0 - self.cumulative_laplace(y_hat - half, mu[...,5],sigma[...,5]))) + \
                   probs_mix[..., 2] * (
                       probs_log[..., 0] * (1.0 - self.cumulative_logistic(y_hat - half, mu[...,6], sigma[...,6])) +
                       probs_log[..., 1] * (1.0 - self.cumulative_logistic(y_hat - half, mu[...,7],sigma[...,7])) +
                       probs_log[..., 2] * (1.0 - self.cumulative_logistic(y_hat - half, mu[...,8],sigma[...,8])))
        likelihoods = torch.where(y_hat < -254.5, edge_min, torch.where(y_hat > 255.5, edge_max, likelihoods))
        likelihood_lower_bound = 1e-8
        likelihood_upper_bound = 1.0
        likelihoods = torch.clamp(likelihoods, min=likelihood_lower_bound, max=likelihood_upper_bound)
        return -torch.sum(torch.log2(likelihoods), dim=(1, 2, 3))

    def hyperlatent_rate(self, probs_z):
        """
        Calculate hyperlatent rate

        Since we assume that each latent is modelled a Non-parametric convolved with Unit Uniform distribution we calculate latent rate
        as a difference of the CDF of the distribution at two different points shifted by -1/2 and 1/2 (limit points of Uniform distribution)

        See apeendix 6.2
        J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston,
        “Variational image compression with a scale hyperprior,” 6th Int. Conf. on Learning Representations, 2018. [Online].
        Available: https://openreview.net/forum?id=rkcQFMZRb.
        """
        likelihoods = torch.abs(probs_z)
        likelihood_lower_bound = 1e-8
        likelihood_upper_bound = 1.0
        likelihoods = torch.clamp(likelihoods, min=likelihood_lower_bound, max=likelihood_upper_bound)
        return -torch.sum(torch.log2(likelihoods), dim=(1, 2, 3))

    def forward(self, x, x_hat, y_hat, probs_z, means, variances, probs, probs_lap, probs_log, probs_mix, lam,num_pixels,model_type):

        mse = torch.mean((x - x_hat) ** 2, [0, 1, 2, 3])
        ms_ssim = 1-compute_ms_ssim(x, x_hat, data_range=255, size_average=True)
        latent_rate = self.latent_rate(y_hat,means, variances, probs, probs_lap, probs_log, probs_mix)
        latent_rate = torch.mean(latent_rate)/num_pixels
        hyperlatent_rate = self.hyperlatent_rate(probs_z)
        hyperlatent_rate = torch.mean(hyperlatent_rate)/num_pixels
        if model_type==0:
            loss = lam * mse + latent_rate + hyperlatent_rate
        else:
            loss = lam * ms_ssim + latent_rate + hyperlatent_rate
        return loss,mse,ms_ssim,latent_rate,hyperlatent_rate


# differentiable rounding function
class BypassRound(Function):
    @staticmethod
    def forward(ctx, inputs):
        return torch.round(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


bypass_round = BypassRound.apply
# Main network
# The current hyper parameters are for higher-bit-rate compression (2x)
# Stage 1: train the main encoder & decoder, fine hyperprior
# Stage 2: train the whole network w/o info-agg sub-network
# Stage 3: disable the final layer of the synthesis transform and enable info-agg net
# Stage 4: End-to-end train the whole network w/o the helping (auxillary) loss

class Net(nn.Module):
    def __init__(self, channel,windowsize):
        super(Net, self).__init__()
        self.a_model = analysisTransformModel(in_dim=3,num_filters=channel,windowsize=windowsize,depths=[6,6,6,6],num_heads=[8,8,8,8])
        self.s_model = synthesisTransformModel(num_filters=channel,windowsize=windowsize,depths=[6,6,6,6],num_heads=[8,8,8,8])
        self.ha_model = h_analysisTransformModel(in_dim=channel,num_filters=channel)
        self.hs_model = h_synthesisTransformModel(num_filters=channel)
        self.entropy_parameter = entropy_parameter(num_filters=channel)
        self.bitEstimator_z = BitEstimator(num_filters=channel)


    def _forward_impl(self, inputs, mode='train'):
        y = self.a_model(inputs)
        if mode == 'train':
            noise = torch.rand_like(y) - 0.5
            y_tilde = y + noise
            y_hat = y_tilde
        else:
            y_hat = bypass_round(y)

        z = self.ha_model(y)

        if mode == 'train':
            noise = torch.rand_like(z) - 0.5
            z_tilde = z + noise
            z_hat = z_tilde
        else:
            z_hat = bypass_round(z)

        phi = self.hs_model(z_hat)
        means,variances,probs,probs_lap,probs_log,probs_mix = self.entropy_parameter(y_hat,phi)
        x_hat = self.s_model(y_hat)
        probs_z = self.bitEstimator_z(z_hat + 0.5) - self.bitEstimator_z(z_hat - 0.5)
        return x_hat,y_hat, z_hat,probs_z,means,variances,probs,probs_lap,probs_log,probs_mix

    def forward(self, images, mode='train'):
        return self._forward_impl(images,mode)






