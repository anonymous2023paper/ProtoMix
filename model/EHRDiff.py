from torch.nn.functional import pad
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from transformers import get_cosine_schedule_with_warmup
import os
import torch
import pickle
import logging
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch.nn as nn
import prettytable as pt
import torch.nn.functional as func

from models import backbone, transformer

from utils_toy.loader import load_data
from utils_toy.option import add_common_args, set_environment, get_schedule_step

from torch.optim import RMSprop
 
from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel
import argparse

import torch.nn.functional as F
from einops import rearrange, reduce
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel
from pyhealth.tokenizer import Tokenizer

def inverse_multi_hot_encode(multi_hot_encoded,device):
    list_3 = [[torch.nonzero(tt.eq(1)).view(-1).numpy().tolist() for tt in t] for t in multi_hot_encoded]
    max_len = max(len(item) for sublist in list_3 for item in sublist)
    padded_list = [[list_1 + (max_len - len(list_1))*[0] for list_1 in list_2] for list_2 in list_3]
    res = torch.tensor(padded_list,device=device)
    return res


def plot_dim_dist(train_data, syn_data, model_setting, best_corr):
    train_data_mean = np.mean(train_data, axis=0)
    temp_data_mean = np.mean(syn_data, axis=0)
    corr = pearsonr(temp_data_mean, train_data_mean)
    nzc = sum(temp_data_mean[i] > 0 for i in range(temp_data_mean.shape[0]))

    fig, ax = plt.subplots(figsize=(8, 6))
    slope, intercept = np.polyfit(train_data_mean, temp_data_mean, 1)
    fitted_values = [slope * i + intercept for i in train_data_mean]
    identity_values = [1 * i + 0 for i in train_data_mean]

    ax.plot(train_data_mean, fitted_values, 'b', alpha=0.5)
    ax.plot(train_data_mean, identity_values, 'r', alpha=0.5)
    ax.scatter(train_data_mean, temp_data_mean, alpha=0.3)
    ax.set_title('corr: %.4f, none-zero columns: %d, slope: %.4f' % (corr[0], nzc, slope))
    ax.set_xlabel('Feature prevalence of real data')
    ax.set_ylabel('Feature prevalence of synthetic data')

     
    fig.savefig('figs/{}.png'.format('Cur_res'))

    flag = False
    if corr[0] > best_corr:
        best_corr = corr[0]
        flag = True
         
        fig.savefig('figs/{}.png'.format('Best_res'))

    plt.close(fig)
    return corr[0], nzc, flag


def pretrain_ehrdiff(do_train=True, train_loader=None, dataset=None, x_key=None):
    parser = argparse.ArgumentParser()
    add_common_args(parser)
     
     
    parser.add_argument("--ehr_dim", default=19256, type=int,
                        help="data dimension of EHR data")
    parser.add_argument("--check_steps", default=5000, type=int,
                        help="the interval for printing the training loss, *batch")
    parser.add_argument("--time_dim", default=384, type=int,
                        help="dimension of time embedding")
    parser.add_argument("--mlp_dims", nargs='+', default=[1024, 384, 384, 384, 1024], type=int,
                        help="hidden dims for the mlp backbone")
    parser.add_argument("--lr", default=3e-4, type=float,
                        help="learning_rate")
    parser.add_argument("--weight_decay", default=0., type=float,
                        help="weigth decay value for the optimizer")
    parser.add_argument("--if_drop_last", default=True, type=bool,
                        help="parameter for the dataloader")
     
     
    parser.add_argument("--num_sample_steps", default=32, type=int,
                        help="init parameters for number of discretized time steps")
    parser.add_argument("--sigma_min", default=0.02, type=float,
                        help="init parameters for sigma_min")
    parser.add_argument("--sigma_max", default=80, type=float,
                        help="init parameters for sigma_max")
    parser.add_argument("--rho", default=7, type=float,
                        help="init parameters for rho")
    parser.add_argument("--sigma_data", default=0.14, type=float,
                        help="init parameters for sigma_data")
    parser.add_argument("--p_mean", default=-1.2, type=float,
                        help="init parameters for p_mean")
    parser.add_argument("--p_std", default=1.2, type=float,
                        help="init parameters for p_std")
    parser.add_argument("--eval_samples", default=41000, type=int,
                        help="number of samples wanted for evaluation")
     

    args = parser.parse_args()

    set_environment(args, 'runs/ehrdiff')

    args.backbone = 'GRACE'
    args.generate_path = 'mortality-generate'
    args.output_path = 'ehrdiff_model'

     
    model = LinearModel(z_dim=args.ehr_dim, time_dim=args.time_dim, unit_dims=args.mlp_dims)
    model.to(args.device)

    diffusion = Diffusion(
        model,
        num_sample_steps=args.num_sample_steps,
         
        dim=args.ehr_dim,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_data=args.sigma_data,
        rho=args.rho,
        P_mean=args.p_mean,
        P_std=args.p_std,
    )

    optimizer = torch.optim.AdamW(
        [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}])
    if args.if_drop_last:
        total_steps = len(train_loader.dataset) // args.batch_size * args.epoch_num
    else:
        total_steps = (len(train_loader.dataset) // args.batch_size + 1) * args.epoch_num
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=total_steps * args.warm_ratio, \
                                                num_training_steps=total_steps)

    data_map = sequence_data_generate(
        dataset=dataset,
        feature_keys=x_key,
        label_key="label",
        mode="binary",
    ).to(args.device)

     
    logging.info('Training begin...')
    best_loss = float('inf')

    train_dm_loss = 0
    train_cnt = 0
    train_steps = 0
    best_corr = 0

    print(args.epoch_num)
    for epoch in range(args.epoch_num):
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            batch_size = args.batch_size
             

            x = data_map(**batch)
            x = x.to(args.device)

            loss_dm = diffusion(x)
            train_dm_loss += loss_dm.item()
            train_cnt += batch_size

            train_steps += 1

            loss_dm.backward()
            optimizer.step()
            scheduler.step()

        logging.info('[%d] dm_loss: %.10f' % (
            epoch + 1, train_dm_loss / train_cnt))

     
    logging.info('Training end...')

    args.batch_size = 64 * 5
    outs_tensor = diffusion.sample(batch_size=args.batch_size).detach().cpu().round().long()
     
    feature_tensor = dict(zip(data_map.feature_keys,
             (outs_tensor[:, :7, :], outs_tensor[:, 7:14, :], outs_tensor[:, 14:, :])))
    patient_emb = []
    patient_mask = []
    for feature_key in data_map.feature_keys:
        feature_tensor[feature_key][:,:,data_map.feat_tokenizers[feature_key].get_vocabulary_size():] = 0
        feature_indices = inverse_multi_hot_encode(feature_tensor[feature_key],args.device)
        x = data_map.embeddings[feature_key](feature_indices)
         
        x = torch.sum(x, dim=2)
         
        mask = torch.any(x != 0, dim=2)

        patient_emb.append(x)
        patient_mask.append(mask)
    patient_emb = torch.cat(patient_emb, dim=1)
    patient_mask = torch.cat(patient_mask, dim=1)

    return [patient_emb.detach()],[patient_mask.detach()]


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def exists(val):
    return val is not None


class Block(nn.Module):
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None):
        super().__init__()

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, dim_in),
            )

        self.out_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_in, dim_out),
        )

    def forward(self, x, time_emb=None):
        if time_emb is not None:
            t_emb: torch.Tensor = self.time_mlp(time_emb)
            h = x + torch.tile(t_emb.unsqueeze(1), (1, x.shape[1], 1))
        else:
            h = x
        out = self.out_proj(h)
        return out


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Diffusion(nn.Module):
    def __init__(
            self,
            net,
            *,
            dim,
            num_sample_steps,
            sigma_min,
            sigma_max,
            sigma_data,
            rho,
            P_mean,
            P_std,
    ):
        super().__init__()

        self.net = net
        self.dim = dim

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps

    @property
    def device(self):
        return next(self.net.parameters()).device

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    def preconditioned_network_forward(self, noised_ehr, sigma, clamp=False):
        batch, device = noised_ehr.shape[0], noised_ehr.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1')

        net_out = self.net(
            self.c_in(padded_sigma) * noised_ehr,
            self.c_noise(sigma),
        )

        out = self.c_skip(padded_sigma) * noised_ehr + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(0, 1)
        return out

    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (
                self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.)
        return sigmas

    @torch.no_grad()
    def sample(self, batch_size: int = 32, num_sample_steps: int = None,
               clamp: bool = True) -> torch.Tensor:
         
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (batch_size, 21, self.dim)

         
        sigmas = self.sample_schedule(num_sample_steps)

         
        sigmas_and_sigmas_next = list(zip(sigmas[:-1], sigmas[1:]))

         
        init_sigma = sigmas[0]

         
        ehr = init_sigma * torch.randn(shape, device=self.device)

         
        for sigma, sigma_next in sigmas_and_sigmas_next:

            sigma, sigma_next = map(lambda t: t.item(), (sigma, sigma_next))

            model_output = self.preconditioned_network_forward(ehr, sigma, clamp=clamp)

            denoised_over_sigma = (ehr - model_output) / sigma

             
            ehr_next = ehr + (sigma_next - sigma) * denoised_over_sigma

             
            if sigma_next != 0:
                 
                model_output_next = self.preconditioned_network_forward(ehr_next, sigma_next,
                                                                        clamp=clamp)

                 
                denoised_prime_over_sigma = (ehr_next - model_output_next) / sigma_next

                 
                ehr_next = ehr + 0.5 * (sigma_next - sigma) * (
                        denoised_over_sigma + denoised_prime_over_sigma)

             
            ehr = ehr_next

         
        return ehr

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()

    def forward(self, ehr: torch.Tensor) -> torch.Tensor:
         
        batch_size = ehr.shape[0]

         
        sigmas = self.noise_distribution(batch_size)

         
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1')

         
        noise = torch.randn_like(ehr)

         
        noised_ehr = ehr + padded_sigmas * noise

         
        denoised = self.preconditioned_network_forward(noised_ehr, sigmas)

         
        losses = F.mse_loss(denoised, ehr, reduction='none')

         
        losses = reduce(losses, 'b ... -> b', 'mean')

         
        losses = losses * self.loss_weight(sigmas)

         
        return losses.mean()


class LinearModel(nn.Module):
    def __init__(
            self, *,
            z_dim,
            time_dim,
            unit_dims,
    ):
        super().__init__()

        num_linears = len(unit_dims)
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(z_dim),
            nn.Linear(z_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.block_in = Block(dim_in=z_dim, dim_out=unit_dims[0], time_emb_dim=time_dim)
        self.block_mid = nn.ModuleList()
        for i in range(num_linears - 1):
            self.block_mid.append(Block(dim_in=unit_dims[i], dim_out=unit_dims[i + 1]))
        self.block_out = Block(dim_in=unit_dims[-1], dim_out=z_dim)

    def forward(self, x, time_steps):

        t_emb = self.time_embedding(time_steps)
        x = self.block_in(x, t_emb)

        num_mid_blocks = len(self.block_mid)
        if num_mid_blocks > 0:
            for block in self.block_mid:
                x = block(x)

        x = self.block_out(x)
        return x


class sequence_data_generate(BaseModel):
    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            embedding_dim: int = 128,
            hidden_dim: int = 128,
            **kwargs
    ):
        super(sequence_data_generate, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

         
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")

         
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
         
        self.embeddings = nn.ModuleDict()
         
        self.linear_layers = nn.ModuleDict()

         
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
             
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "RNN only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "RNN only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                    input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "RNN only supports 2-dim or 3-dim float and int as input types"
                )
             
             
            self.add_feature_transform_layer(feature_key, input_info)


    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        patient_emb = []
        patient_mask = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

             
            if (dim_ == 2) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                 
                x = torch.tensor(x, dtype=torch.long, device=self.device)

                 
                batch_size, dim1 = x.size()
                 
                num_categories = 19256

                 
                multi_hot_encoded = torch.zeros(batch_size, num_categories, device=self.device)
                multi_hot_encoded.scatter_(1, x, 1)
                multi_hot_encoded[:, 0] = 0

                x = multi_hot_encoded

            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                 
                x = torch.tensor(x, dtype=torch.long, device=self.device)

                 
                batch_size, dim1, dim2 = x.size()
                 
                num_categories = 19256

                 
                multi_hot_encoded = torch.zeros(batch_size, dim1, num_categories,
                                                device=self.device)
                multi_hot_encoded.scatter_(2, x, 1)
                multi_hot_encoded[:, :, 0] = 0

                x = multi_hot_encoded

             
            elif (dim_ == 2) and (type_ in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                 
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                 
                x = self.linear_layers[feature_key](x)
                 
                mask = mask.bool().to(self.device)

             
            elif (dim_ == 3) and (type_ in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                 
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                 
                x = torch.sum(x, dim=2)
                x = self.linear_layers[feature_key](x)
                 
                mask = mask[:, :, 0]
                mask = mask.bool().to(self.device)

            else:
                raise NotImplementedError

            patient_emb.append(x)
             
        max_len = max([ts.shape[2] for ts in patient_emb])
        paded_patient_emb = [pad(ts, (0, max_len - ts.shape[2])) for ts in patient_emb]
        patient_emb = torch.cat(paded_patient_emb, dim=1)

        return patient_emb
