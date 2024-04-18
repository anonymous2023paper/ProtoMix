 
import os
import torch
import time as timem
from torch import nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.utils.data as Data
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import math
lab = medi = inputs = [1]

class TContextGGANN(nn.Module):
    def __init__(self, dim_lab, dim_inputs, dim_med, dim_embd1, num_layer=2, loop_num=1, \
                 nodecay=True, basic=True, qkv_diff=True, dp_rate=0.5, pass_v=True):
        super(TContextGGANN, self).__init__()
        self.dim_lab = dim_lab
        self.dim_med = dim_med
        self.dim_inputs = dim_inputs
        self.dim_embd = dim_lab
        self.num_layer = num_layer

        self.embd_lab = nn.Linear(dim_lab, dim_lab)
        self.embd_inputs = nn.Linear(dim_inputs, dim_inputs)
        self.embd_med = nn.Linear(dim_med, dim_med)
        self.embd_med2 = nn.Linear(dim_med, dim_med)

        self.gats = nn.ModuleList()
        for _ in range(self.num_layer):
            self.gats.append(TGGANN(dim_embd=self.dim_embd, loop_num=loop_num, nodecay=nodecay, basic=basic))

        self.saLayer = SelfAttentionLayer(dim_embd=self.dim_embd, dp_rate=dp_rate, qkv_diff=qkv_diff, pass_v=pass_v,
                                          out=True)

        self.beta_fc = nn.Linear(self.dim_embd, dim_embd1)
        self.output = nn.Linear(in_features=dim_embd1, out_features=1)

    def reset_params(self):
        torch.nn.init.kaiming_normal_(self.embd_lab.weight)
        self.embd_lab.bias.data.zero_()
        torch.nn.init.kaiming_normal_(self.embd_inputs.weight)
        self.embd_inputs.bias.data.zero_()
        torch.nn.init.kaiming_normal_(self.embd_med.weight)
        self.embd_med.bias.data.zero_()
        for l in range(self.num_layer):
            self.gats[l].reset_params()
        self.saLayer.reset_params()
        torch.nn.init.kaiming_normal_(self.beta_fc.weight)
        self.beta_fc.bias.data.zero_()
        torch.nn.init.kaiming_normal_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self, batch):
        lab_batch, input_batch, med_batch, lab_mask = batch
        batch_size = lab_batch.shape[0]
        lab_mask = lab_mask.repeat([1, lab_batch.shape[-1]])
         
         
        length = 1
        decay = None
         
         
        lab_batch = torch.tensor(lab_batch, dtype=torch.float32).cuda()
        input_batch = torch.tensor(input_batch, dtype=torch.float32).cuda()
        med_batch = torch.tensor(med_batch, dtype=torch.float32).cuda()
        lab_mask = torch.tensor(lab_mask, dtype=torch.float32).cuda()

        ih_l = self.embd_lab(med_batch)
        ih_i = self.embd_inputs((input_batch != 0).to(torch.float32).cuda())
        ih_m  = self.embd_med(med_batch)
        ih_e = self.embd_med2(lab_batch)

        h_e, h_m, h_l, h_i = ih_e, ih_m, ih_l, ih_i
        for l in range(self.num_layer - 1):
            h_e, h_m, h_l, h_i = self.gats[l](med_batch, input_batch, lab_batch, decay, None, length, h_e, h_m,
                                              h_l, h_i)
            h_e, h_m, h_l, h_i = F.leaky_relu(h_e), F.leaky_relu(h_m), F.leaky_relu(h_l), F.leaky_relu(h_i)
        h_e, h_m, h_l, h_i = self.gats[self.num_layer - 1](med_batch, input_batch, lab_batch, decay, None, length,
                                                           h_e, h_m, h_l, h_i)

        h_e = self.saLayer(h_e, h_l, h_i, h_m, length)

        beta = torch.tanh(self.beta_fc(h_e))

         
         
         
         
         

        logit = self.output(beta)
        logit = F.softmax(logit)
        return logit


class FocalLoss(nn.Module):
    def __init__(self,weight=None,gamma=2):
        super(FocalLoss, self).__init__()
        self.weight=weight
        self.gamma=gamma
    def forward(self, pred_y, targets):
        CE_loss=F.cross_entropy(pred_y, targets, weight=self.weight)
        mask=targets.float()*(pred_y[:,0]**self.gamma)+(1-targets.float())*(pred_y[:,1]**self.gamma)
        return torch.mean(mask*CE_loss)


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim_embd=len(lab + medi + inputs), dp_rate=0.5, qkv_diff=True, pass_v=True, out=False):
        super(SelfAttentionLayer, self).__init__()
        self.dp_rate = dp_rate
        self.dim_embd = dim_embd
        self.qkv_diff = qkv_diff
        self.pass_v = pass_v
        self.out = out

        self.Wq = nn.Sequential(
            nn.Linear(dim_embd, dim_embd, bias=False),
            nn.Dropout(p=dp_rate)
        )
        self.Wk = nn.Sequential(
            nn.Linear(dim_embd*3, dim_embd, bias=False),
            nn.Dropout(p=dp_rate)
        )
        self.Wv = nn.Sequential(
            nn.Linear(dim_embd*3, dim_embd, bias=False),
            nn.Dropout(p=dp_rate)
        )

    def reset_params(self):
        torch.nn.init.kaiming_normal_(self.Wq[0].weight)
        torch.nn.init.kaiming_normal_(self.Wk[0].weight)
        torch.nn.init.kaiming_normal_(self.Wv[0].weight)

    def forward(self, h_e, h_l, h_i, h_m, length):
        batch_size = len(h_e)
         

        if self.qkv_diff:
            h_c = torch.cat([h_l, h_i, h_m], dim=1)

            Q = self.Wq(h_e)
            K = self.Wk(h_c)
            V = self.Wv(h_c)
            if self.pass_v:
                P = V
            else:
                P = K
            Z = torch.matmul(F.softmax(torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(K.shape[-1]), dim=-1), V)
             

            h_e = h_e + Z
             
             
             
        else:
            h_c = torch.cat([h_e, h_l, h_i, h_m], dim=1)
            Q = self.Wq(h_c)
            K = self.Wk(h_c)
            V = self.Wv(h_c)

            Z = torch.matmul(F.softmax(torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(K.shape[-1]), dim=-1), V)
            for b, blength in enumerate(length):
                Z[b][blength:] = 0

            h_e = h_e + Z[:, :max(length), :]
            h_l = h_l + Z[:, max(length):max(length) + len(lab), :]
            h_i = h_i + Z[:, max(length) + len(lab):max(length) + len(lab + inputs), :]
            h_m = h_m + Z[:, max(length) + len(lab + inputs):, :]

        if not self.out:
            return h_e, h_l, h_i, h_m
        else:
            return h_e


class TGGANN(nn.Module):
    def __init__(self, dim_embd, loop_num, nodecay=False, basic=True):
        super(TGGANN, self).__init__()
         
        self.dim_embd = dim_embd
        self.loop_num = loop_num
        self.nodecay = nodecay
        self.basic = basic
        if self.basic:
            self.nodecay = True
        if not self.nodecay and not self.basic:
             
            self.W_decay_enc = nn.Linear(dim_embd, dim_embd)
             
            self.W_decay_med = nn.Linear(dim_embd, dim_embd)
             
            self.W_decay_inputs = nn.Linear(dim_embd, dim_embd)
         
        self.vars_r_in = torch.nn.ParameterList()
         
        weight_lab_r = nn.Parameter(torch.ones(dim_embd * 2, dim_embd))
        self.vars_r_in.append(weight_lab_r)
         
        weight_inputs_r = nn.Parameter(torch.ones(dim_embd * 2, dim_embd))
        self.vars_r_in.append(weight_inputs_r)
         
        weight_med_r = nn.Parameter(torch.ones(dim_embd * 2, dim_embd))
        self.vars_r_in.append(weight_med_r)
         
        weight_enc_r = nn.Parameter(torch.ones(dim_embd * 2, dim_embd))
        self.vars_r_in.append(weight_enc_r)
         
        self.vars_out = torch.nn.ParameterList()
         
        weight_lab_r = nn.Parameter(torch.ones(dim_embd, dim_embd))
        self.vars_out.append(weight_lab_r)
         
        weight_inputs_r = nn.Parameter(torch.ones(dim_embd, dim_embd))
        self.vars_out.append(weight_inputs_r)
         
        weight_med_r = nn.Parameter(torch.ones(dim_embd, dim_embd))
        self.vars_out.append(weight_med_r)
         
        weight_enc_r = nn.Parameter(torch.ones(dim_embd, dim_embd))
        self.vars_out.append(weight_enc_r)
         
        self.vars_z_in = torch.nn.ParameterList()
         
        weight_lab_z = nn.Parameter(torch.ones(2 * dim_embd, 2 * dim_embd))
        self.vars_z_in.append(weight_lab_z)
         
        weight_inputs_z = nn.Parameter(torch.ones(2 * dim_embd, 2 * dim_embd))
        self.vars_z_in.append(weight_inputs_z)
         
        weight_med_z = nn.Parameter(torch.ones(2 * dim_embd, 2 * dim_embd))
        self.vars_z_in.append(weight_med_z)
         
        weight_enc_z = nn.Parameter(torch.ones(2 * dim_embd, 2 * dim_embd))
        self.vars_z_in.append(weight_enc_z)
         
        self.out_h_enc = nn.Linear(dim_embd, dim_embd, bias=False)
        self.out_x_enc = nn.Linear(dim_embd, dim_embd, bias=False)

        self.out_h_lab = nn.Linear(dim_embd, dim_embd, bias=False)
        self.out_x_lab = nn.Linear(dim_embd, dim_embd, bias=False)

        self.out_h_input = nn.Linear(dim_embd, dim_embd, bias=False)
        self.out_x_input = nn.Linear(dim_embd, dim_embd, bias=False)

        self.out_h_med = nn.Linear(dim_embd, dim_embd, bias=False)
        self.out_x_med = nn.Linear(dim_embd, dim_embd, bias=False)
         

    def reset_params(self):
        if not self.nodecay and not self.basic:
             
            torch.nn.init.kaiming_normal_(self.W_decay_enc.weight)
            self.W_decay_enc.bias.data.zero_()
             
            torch.nn.init.kaiming_normal_(self.W_decay_med.weight)
            self.W_decay_med.bias.data.zero_()
             
            torch.nn.init.kaiming_normal_(self.W_decay_inputs.weight)
            self.W_decay_inputs.bias.data.zero_()
        for i in range(len(self.vars_r_in)):
            torch.nn.init.kaiming_normal_(self.vars_r_in[i])
        for i in range(len(self.vars_out)):
            torch.nn.init.kaiming_normal_(self.vars_out[i])
        for i in range(len(self.vars_z_in)):
            torch.nn.init.kaiming_normal_(self.vars_z_in[i])
        torch.nn.init.kaiming_normal_(self.out_h_enc.weight)
        torch.nn.init.kaiming_normal_(self.out_x_enc.weight)
        torch.nn.init.kaiming_normal_(self.out_h_lab.weight)
        torch.nn.init.kaiming_normal_(self.out_x_lab.weight)
        torch.nn.init.kaiming_normal_(self.out_h_input.weight)
        torch.nn.init.kaiming_normal_(self.out_x_input.weight)
        torch.nn.init.kaiming_normal_(self.out_h_med.weight)
        torch.nn.init.kaiming_normal_(self.out_x_med.weight)

    def forward(self, enc_med, enc_inputs, enc_lab, decay_mask_lower, time, lengths, h_e_r=None, h_m_r=None, h_l_r=None,
                h_i_r=None):
        h_e = h_e_r
        h_m = h_m_r
        h_l = h_l_r
        h_i = h_i_r
        batch_size = enc_med.shape[0]
        if h_e is None:
            h_e = torch.zeros(batch_size, enc_med.shape[1], self.dim_embd)
        if h_m is None:
            h_m = torch.zeros(batch_size, enc_med.shape[-1], self.dim_embd)
        if h_l is None:
            h_l = torch.zeros(batch_size, enc_lab.shape[-1], self.dim_embd)
        if h_i is None:
            h_i = torch.zeros(batch_size, enc_inputs.shape[-1], self.dim_embd)
        if self.vars_out[0].is_cuda and not h_e.is_cuda:
            h_e = h_e.cuda()
        if self.vars_out[0].is_cuda and not h_m.is_cuda:
            h_m = h_m.cuda()
        if self.vars_out[0].is_cuda and not h_l.is_cuda:
            h_l = h_l.cuda()
        if self.vars_out[0].is_cuda and not h_i.is_cuda:
            h_i = h_i.cuda()
        for loop in range(self.loop_num):
            if not self.nodecay and not self.basic:
                 
                h_e_temp = lower_(h_e, lengths)
                h_e_r_s = torch.tanh(self.W_decay_enc(h_e_temp))
                h_e_r_s_ = h_e_r_s * decay_mask_lower
                h_e_r_t = h_e_temp - h_e_r_s
                h_e_r_adj = h_e_r_s_ + h_e_r_t
                 
                h_nm_temp = torch.matmul(lower_(enc_med, lengths), h_m)
                h_nm_r_s = torch.tanh(self.W_decay_med(h_nm_temp))
                h_nm_r_s_ = h_nm_r_s * decay_mask_lower
                h_nm_r_t = h_nm_temp - h_nm_r_s
                h_nm_adj = h_nm_r_s_ + h_nm_r_t
                 
                h_ni_temp = torch.matmul(lower_(enc_inputs, lengths), h_i)
                h_ni_r_s = torch.tanh(self.W_decay_inputs(h_ni_temp))
                h_ni_r_s_ = h_ni_r_s * decay_mask_lower
                h_ni_r_t = h_ni_temp - h_ni_r_s
                h_ni_adj = h_ni_r_s_ + h_ni_r_t
            elif self.nodecay and not self.basic:
                 
                h_e_r_adj = lower_(h_e, lengths)
                 
                h_nm_adj = torch.matmul(lower_(enc_med, lengths), h_m)
                 
                h_ni_adj = torch.matmul(lower_(enc_inputs, lengths), h_i)
             
            h_lab2enc = F.leaky_relu(torch.matmul((enc_lab * h_l), self.vars_out[0]))             
            h_inputs2enc = F.leaky_relu(torch.matmul((enc_inputs * h_i), self.vars_out[1]))
            h_med2enc = F.leaky_relu(torch.matmul((enc_med * h_m), self.vars_out[2]))
            h_enc2lab = F.leaky_relu(torch.matmul((enc_lab * h_e), self.vars_out[3]))
            h_enc2med = F.leaky_relu(torch.matmul((enc_med *  h_e), self.vars_out[3]))
            h_enc2inputs = F.leaky_relu(torch.matmul((enc_inputs * h_e), self.vars_out[3]))
            if not self.basic:
                h_enc2nmed = F.leaky_relu(
                    torch.matmul(torch.matmul(upper_(enc_med).transpose(-1, -2), h_e), self.vars_out[3]))
                h_enc2ninputs = F.leaky_relu(
                    torch.matmul(torch.matmul(upper_(enc_inputs).transpose(-1, -2), h_e), self.vars_out[3]))
             
            if not self.basic:
                h_e_r_c = torch.cat([torch.unsqueeze(h_e_r_adj, dim=2), torch.unsqueeze(h_lab2enc, dim=2), \
                                     torch.unsqueeze(h_inputs2enc, dim=2), torch.unsqueeze(h_med2enc, dim=2), \
                                     torch.unsqueeze(h_nm_adj, dim=2), torch.unsqueeze(h_ni_adj, dim=2)], dim=2)
                h_l_r_c = torch.cat([torch.unsqueeze(h_enc2lab, dim=2)], dim=2)
                h_i_r_c = torch.cat([torch.unsqueeze(h_enc2inputs, dim=2), torch.unsqueeze(h_enc2ninputs, dim=2)],
                                    dim=2)
                h_m_r_c = torch.cat([torch.unsqueeze(h_enc2med, dim=2), torch.unsqueeze(h_enc2nmed, dim=2)], dim=2)
            else:
                h_e_r_c = torch.cat([torch.unsqueeze(h_lab2enc, dim=2), torch.unsqueeze(h_inputs2enc, dim=2),
                                     torch.unsqueeze(h_med2enc, dim=2)], dim=2)
                h_l_r_c = torch.cat([torch.unsqueeze(h_enc2lab, dim=2)], dim=2)
                h_i_r_c = torch.cat([torch.unsqueeze(h_enc2inputs, dim=2)], dim=2)
                h_m_r_c = torch.cat([torch.unsqueeze(h_enc2med, dim=2)], dim=2)
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
            h_e_r_outs = F.sigmoid(torch.matmul(torch.cat([h_e, h_e], dim=-1), self.vars_z_in[3]))
            h_l_r_outs = F.sigmoid(torch.matmul(torch.cat([h_l, h_l], dim=-1), self.vars_z_in[0]))
            h_i_r_outs = F.sigmoid(torch.matmul(torch.cat([h_i, h_i], dim=-1), self.vars_z_in[1]))
            h_m_r_outs = F.sigmoid(torch.matmul(torch.cat([h_m, h_m], dim=-1), self.vars_z_in[2]))
             
            h_e_r_r, h_e_r_z = torch.chunk(h_e_r_outs, 2, -1)
            h_l_r_r, h_l_r_z = torch.chunk(h_l_r_outs, 2, -1)
            h_i_r_r, h_i_r_z = torch.chunk(h_i_r_outs, 2, -1)
            h_m_r_r, h_m_r_z = torch.chunk(h_m_r_outs, 2, -1)
             
            h_e_ = torch.tanh(self.out_h_enc(h_e_r_r * h_e) + self.out_x_enc(h_e))
            h_l_ = torch.tanh(self.out_h_lab(h_l_r_r * h_l) + self.out_x_lab(h_l))
            h_i_ = torch.tanh(self.out_h_input(h_i_r_r * h_i) + self.out_x_input(h_i))
            h_m_ = torch.tanh(self.out_h_med(h_m_r_r * h_m) + self.out_x_med(h_m))
             
             
            h_e = ((1.0 - h_e_r_z) * h_e + h_e_r_z * torch.tanh(h_e_))
            h_l = ((1.0 - h_l_r_z) * h_l + h_l_r_z * torch.tanh(h_l_))
            h_m = ((1.0 - h_m_r_z) * h_m + h_m_r_z * torch.tanh(h_m_))
            h_i = ((1.0 - h_i_r_z) * h_i + h_i_r_z * torch.tanh(h_i_))
        return h_e, h_m, h_l, h_i


def upper(m):
    nm=torch.zeros(m.shape)
    if len(m.shape)>1:
        for i in range(m.shape[0]-1):
            nm[i,:]=m[i+1,:]
    if m.is_cuda:
        nm=nm.cuda()
    return nm

def upper_(m):
    eye=torch.eye(m.shape[-2])
    if m.is_cuda:
        eye=eye.cuda()
    return torch.matmul(upper(eye),m)

def lower(m):
    nm=torch.zeros(m.shape)
    if len(m.shape)>1:
        for i in range(1,m.shape[0]):
            nm[-i,:]=m[-i-1,:]
    if m.is_cuda:
        nm=nm.cuda()
    return nm

def lower_(m,length):
    eye=torch.eye(m.shape[-2])
    batch_size=m.shape[0]
    if m.is_cuda:
        eye=eye.cuda()
    result=torch.matmul(lower(eye),m)
    for b in range(batch_size):
        blength=length[b]
        for r in range(blength,max(length)):
            result[b][r]=0
    return torch.matmul(lower(eye),m)

def lower_decay(tlist, max_length):
    decay_mask=[]
    for b in range(len(tlist)):
        t=tlist[b].clone().detach()
        t_=t.clone().detach()
        for i in range(1,len(t_)):
            t_[i]=t[i-1]
        gap=t-t_
        decay=1.0/torch.log(math.e+gap*1.0)
        padding=torch.zeros(max_length-len(decay))
        decay=torch.cat([decay,padding],dim=0)
        decay_mask.append(torch.unsqueeze(decay, dim = 0))
    torch.cat(decay_mask, dim = 0)
    return torch.unsqueeze(decay_mask,dim=-1)

def upper_decay(tlist, max_length):
    decay_mask=[]
    for b in range(len(tlist)):
        t=tlist[b]
        t_=t.clone().detach()
        for i in range(0,len(t_)-1):
            t_[i]=t[i+1]
        gap=t_-t
        decay=1.0/torch.log(math.e+gap*1.0)
        padding=torch.zeros(max_length-len(decay))
        decay=torch.cat([decay,padding],dim=0)
        if b==0:
            decay_mask=torch.unsqueeze(decay,dim=0)
        else:
            decay_mask=torch.cat([decay_mask,torch.unsqueeze(decay,dim=0)],dim=0)
    return torch.unsqueeze(decay_mask,dim=-1)


