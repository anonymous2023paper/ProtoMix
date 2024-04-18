from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel




class RNNLayer(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "GRU",     
        num_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = False,
    ):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.dropout_layer = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        rnn_module = getattr(nn, rnn_type)
        self.rnn = rnn_module(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if bidirectional:
            self.down_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:

        x = self.dropout_layer(x)
        batch_size = x.size(0)
        if mask is None:
            lengths = torch.full(
                size=(batch_size,), fill_value=x.size(1), dtype=torch.int64
            )
        else:
            lengths = torch.sum(mask.int(), dim=-1).cpu()
        lengths = torch.where(lengths<=0, torch.ones_like(lengths), lengths)
        x = rnn_utils.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(x)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
        if not self.bidirectional:
            last_outputs = outputs[torch.arange(batch_size), (lengths - 1), :]
            return outputs, last_outputs
        else:
            outputs = outputs.view(batch_size, outputs.shape[1], 2, -1)
            f_last_outputs = outputs[torch.arange(batch_size), (lengths - 1), 0, :]
            b_last_outputs = outputs[:, 0, 1, :]
            last_outputs = torch.cat([f_last_outputs, b_last_outputs], dim=-1)
            outputs = outputs.view(batch_size, outputs.shape[1], -1)
            last_outputs = self.down_projection(last_outputs)
            outputs = self.down_projection(outputs)
            return outputs, last_outputs


class RNN(BaseModel):

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
        super(RNN, self).__init__(
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

        self.rnn = nn.ModuleDict()
        for feature_key in feature_keys:
            self.rnn[feature_key] = RNNLayer(
                input_size=embedding_dim, hidden_size=hidden_dim, **kwargs
            )
        output_size = 1
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_emb = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            
            if (dim_ == 2) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                
                x = self.embeddings[feature_key](x)
                
                mask = torch.any(x !=0, dim=2)

            
            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                
                x = self.embeddings[feature_key](x)
                
                x = torch.sum(x, dim=2)
                
                mask = torch.any(x !=0, dim=2)

            
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

            _, x = self.rnn[feature_key](x, mask)       
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        
        logits = self.fc(patient_emb)
        
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        self.test_x = patient_emb
        self.test_y = y_true
        return results


class FineTuneRNN(BaseModel):
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
        super(FineTuneRNN, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.train_mode = None

        
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

        self.rnn = nn.ModuleDict()
        for feature_key in feature_keys:
            self.rnn[feature_key] = RNNLayer(
                input_size=embedding_dim, hidden_size=hidden_dim, **kwargs
            )

        self.total_rnn = RNNLayer(input_size=embedding_dim, hidden_size=hidden_dim, **kwargs)     

        output_size = self.get_output_size(self.label_tokenizer)
        self.old_fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)
        self.fc = nn.Linear(self.hidden_dim, output_size)
        self.MLM_fc = nn.Linear(self.hidden_dim, 100*self.hidden_dim)       

        
        self.synx_data = None
        self.synx_data_mask = None

        self.train_x_emb = []
        self.train_y = []

        self.gene_x_emb = []
        self.gene_y = []

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
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
                
                x = self.embeddings[feature_key](x)
                
                mask = torch.any(x !=0, dim=2)

            
            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                
                x = self.embeddings[feature_key](x)
                
                x = torch.sum(x, dim=2)
                
                mask = torch.any(x !=0, dim=2)

            
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
            patient_mask.append(mask)

        patient_emb = torch.cat(patient_emb, dim=1)
        patient_mask = torch.cat(patient_mask, dim=1)

        if self.train_mode == 'Pretrain':
            mask_prob = 0.1
            mask_visit_prob = torch.rand_like(patient_emb)
            mask_visit_prob = mask_visit_prob > mask_prob
            patient_emb = patient_emb * mask_visit_prob
            punish_oral = mask_visit_prob * 10 + ~mask_visit_prob * 1

        _, x = self.total_rnn(patient_emb, patient_mask)

        
        logits = self.fc(x)
        
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        y_prob = self.prepare_y_prob(logits)

        
        
        

        loss = 0
        if self.train_mode == 'Pretrain':
            for (synx_data, synx_data_mask) in zip(self.synx_data, self.synx_data_mask):        
                
                
                
                mask_visit_prob = torch.rand_like(synx_data)
                mask_visit_prob = mask_visit_prob > mask_prob
                synx_data = synx_data * mask_visit_prob
                punish_synx = mask_visit_prob * 10 + ~mask_visit_prob * 1

                _, synx_x_emb = self.total_rnn(synx_data, synx_data_mask)
                
                
                

                rec_x_synx = self.MLM_fc(synx_x_emb)
                rec_x_oral = self.MLM_fc(x)

                rec_x_synx = rec_x_synx.reshape(rec_x_synx.shape[0], 100, -1)
                rec_x_oral = rec_x_oral.reshape(rec_x_oral.shape[0], 100, -1)
                rec_x_synx = rec_x_synx[:, :synx_data.shape[1], :]
                rec_x_oral = rec_x_oral[:, :patient_emb.shape[1], :]

                MLM_loss = (((rec_x_oral - patient_emb) ** 2 * punish_oral).masked_select(patient_mask.unsqueeze(-1).repeat(1,1,punish_oral.shape[-1]))).mean() + \
                           (((rec_x_synx - synx_data) ** 2 * punish_synx).masked_select(synx_data_mask.unsqueeze(-1).repeat(1,1,punish_synx.shape[-1]))).mean()

                loss += MLM_loss
                
                loss += 0.33 * self.fake_contrastive(x, synx_x_emb)        

        elif self.train_mode == 'FineTune':
            self.test_x = x
            self.test_y = y_true
            loss += self.get_loss_function()(logits, y_true)

        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


    def fake_contrastive(self, fake_feature, real_feature):
        def get_loss(positive, negative):
            n = positive.size(0)

            all_feature = torch.cat([positive, negative], dim=0)
            similarity = torch.matmul(positive, all_feature.transpose(0, 1))

            label = torch.zeros_like(similarity)
            label[:, :n] = 1

            loss = F.binary_cross_entropy(similarity.softmax(dim=1), label)

            return loss

        loss = get_loss(fake_feature, real_feature) + get_loss(real_feature, fake_feature)

        return loss


class VAE_based_Transformer_Tech(BaseModel):
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
        super(VAE_based_Transformer_Tech, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.train_mode = None

        
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
            

        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim*3, nhead=3)       
        
        self.VAE_Encoder = nn.Sequential(           
            nn.TransformerEncoder(encoder_layer, num_layers=3),       
            nn.ReLU(),
            nn.Linear(int(hidden_dim*3), int(hidden_dim*6))
        )

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim*3, nhead=3)       
        self.VAE_Decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)       

        self.total_rnn = RNNLayer(input_size=embedding_dim*3, hidden_size=hidden_dim, **kwargs)     

        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(self.hidden_dim, output_size)

        
        self.synx_data = None
        self.synx_data_mask = None

        self.multi_head = 3
        
        self.Q_Linear = nn.Linear(int(hidden_dim*3), int(hidden_dim*3)*self.multi_head)
        self.K_Linear = nn.Linear(int(hidden_dim*3), int(hidden_dim*3)*self.multi_head)
        self.V_Linear = nn.Linear(int(hidden_dim*3), int(hidden_dim*3)*self.multi_head)

        self.Score_Linear = nn.Linear(int(hidden_dim*3), 1)

        self.generate_chain = []
        self.generate_mask = []
        self.generate_label = []

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
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
                
                x = self.embeddings[feature_key](x)
                
                mask = torch.any(x !=0, dim=2)

            
            elif (dim_ == 3) and (type_ == str):                        
                if self.train_mode == 'MixTrain':                       
                    with torch.no_grad():
                        x = self.feat_tokenizers[feature_key].batch_encode_3d(
                            kwargs[feature_key]
                        )
                        
                        x = torch.tensor(x, dtype=torch.long, device=self.device)
                        
                        x = self.embeddings[feature_key](x)
                        
                        x = torch.sum(x, dim=2)
                        
                        mask = torch.any(x !=0, dim=2)
                else:
                    x = self.feat_tokenizers[feature_key].batch_encode_3d(
                        kwargs[feature_key]
                    )
                    
                    x = torch.tensor(x, dtype=torch.long, device=self.device)
                    
                    x = self.embeddings[feature_key](x)
                    
                    x = torch.sum(x, dim=2)
                    
                    mask = torch.any(x != 0, dim=2)

            
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
            patient_mask.append(mask)

        if self.train_mode == 'Pretrain':       
            patient_emb = torch.cat(patient_emb, dim=1)  
            patient_mask = torch.cat(patient_mask, dim=1)  

            patient_mask = patient_mask.unsqueeze(-1).repeat([1, 1, patient_emb.shape[-1]])                
            patient_emb = patient_emb.reshape(patient_emb.shape[0], -1)                   
            patient_mask = patient_mask.reshape(patient_mask.shape[0], -1)                

            
            
            mu, logvar = torch.chunk(self.VAE_Encoder(patient_emb), 2, dim=1)
            std = torch.exp(0.5 * logvar)  
            eps = torch.randn_like(std)  
            z = mu + eps * std
            recon_patient_emb = self.VAE_Decoder(z, memory=z)          

            y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)

        elif self.train_mode == 'TrainChain':
            loss = 0
            patient_emb = [item.unsqueeze(2) for item in patient_emb]
            patient_mask = [item.unsqueeze(2) for item in patient_mask]

            patient_emb = torch.cat(patient_emb, dim=2)    
            patient_mask = torch.cat(patient_mask, dim=2)  
            patient_mask = patient_mask.unsqueeze(-1).repeat([1,1,1,patient_emb.shape[-1]])     

            
            actual_time_len = patient_emb.shape[1]
            assert self.padding_time_len >= 1, 'Padding Size Must Be Greater Than 1'
            total_time_len = actual_time_len + self.padding_time_len

            stop_flag_matrix = torch.zeros(patient_emb.shape[0]).to(self.device)

            loss = 0
            IG_pos_list = []
            IG_neg_list = []

            for time_idx in range(0, total_time_len):     
                if time_idx < actual_time_len:
                    if time_idx == 0:
                        idx_batch = patient_emb[:, time_idx, :, :]          
                        idx_batch = idx_batch.reshape(idx_batch.shape[0], -1)       
                    else:           
                        idx_batch = probablity_matrix.unsqueeze(-1) * patient_emb[:, time_idx, :, :].reshape(idx_batch.shape[0], -1) \
                                    + ~probablity_matrix.unsqueeze(-1) * recon_patient_emb
                    
                    stop_flag_matrix = torch.where(torch.any(patient_mask[:, time_idx, :, 0], 1)==True,
                                               torch.zeros_like(stop_flag_matrix), stop_flag_matrix+1)
                else:
                    idx_batch = recon_patient_emb
                    stop_flag_matrix = stop_flag_matrix + 1

                if stop_flag_matrix.min() > self.padding_time_len:      
                    break
                

                
                mu, logvar = torch.chunk(self.VAE_Encoder(idx_batch), 2, dim=1)
                std = torch.exp(0.5 * logvar)  
                eps = torch.randn_like(std)  

                z = mu + eps * std

                
                if time_idx == 0:       
                    memory_mu = mu
                    memory_std = std
                    memory_z = torch.sigmoid(z)


                
                query_multi = torch.chunk(self.Q_Linear(memory_z), self.multi_head, dim=1)  
                key_multi = torch.chunk(self.K_Linear(z), self.multi_head, dim=1)  
                value_multi = torch.chunk(self.V_Linear(z), self.multi_head, dim=1)  

                att_hidden_mean = 0
                for (query, key, value) in zip(query_multi, key_multi, value_multi):
                    
                    query = query / math.sqrt(self.hidden_dim)
                    attn = torch.mm(query, key.transpose(-2, -1))
                    attn = F.softmax(attn, dim=-1)
                    att_hidden = torch.mm(attn, value)
                    att_hidden_mean += att_hidden

                
                
                

                time_bn = att_hidden_mean / self.multi_head
                time_z = z + time_bn                          

                time_z = torch.clip(time_z, -5e1, 5e1)        

                loss += - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())       

                
                
                

                
                recon_patient_emb = self.VAE_Decoder(time_z, memory=time_z)                             

                
                probablity_matrix = torch.rand_like(stop_flag_matrix) > self.schdule_probablity         
                
                
                probablity_matrix = probablity_matrix * (stop_flag_matrix == 0)     

                
                IG = -torch.mean(torch.exp(time_z) * time_z, dim=1) + torch.mean(torch.exp(memory_z) * memory_z, dim=1)
                
                IG = self.Score_Linear(time_z-memory_z).sigmoid()
                IG = torch.clip(IG, -1e5, 1e5)  

                if time_idx < actual_time_len - 1:
                    
                    rec_loss_mask = (stop_flag_matrix == 0).unsqueeze(-1) * (
                        patient_mask[:, time_idx + 1, :, :].reshape(patient_mask.shape[0], -1))

                    loss += torch.mean(rec_loss_mask * (recon_patient_emb -
                                                        patient_emb[:, time_idx+1, :, :].reshape(idx_batch.shape[0], -1))**2)
                    
                    IG_pos = torch.masked_select(IG, patient_mask[:, time_idx + 1, :, 0].any(dim=1))     
                    IG_neg = torch.masked_select(IG, ~patient_mask[:, time_idx + 1, :, 0].any(dim=1))

                    IG_pos_list.append(IG_pos)
                    IG_neg_list.append(IG_neg)

                    
                    
                    
                    
                    
                    
                    
                    
                else:
                    IG_neg_list.append(IG)
                    
                    

                
                
                
                
                

                memory_z = time_z
           
            
            hinge_loss = 0
            pos_len = 0
            pos_prop = 0
            neg_len = 0
            neg_prop = 0
            
            criterion = nn.MSELoss()
            for item in IG_pos_list:
                hinge_loss += criterion(item, torch.ones_like(item))
                pos_len += len(item)
                pos_prop += torch.sum(item > 1 / 2)
            for jtem in IG_neg_list:
                hinge_loss += criterion(jtem, torch.zeros_like(jtem))
                neg_len += len(jtem)
                neg_prop += torch.sum(jtem < 1 / 2)
            loss += hinge_loss
            
            
            
            
            
            
            
            
            
            
            
            
            
            

            print("POS(%), NEG(%), TOTAL(%):", pos_prop/pos_len, neg_prop/neg_len, (pos_prop+neg_prop)/(pos_len+neg_len))

            y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
            pass

        elif self.train_mode == 'GenerateChain':
            
            patient_emb = torch.cat(patient_emb, dim=1)  
            patient_mask = torch.cat(patient_mask, dim=1)  

            stop_flag_matrix = torch.zeros(patient_emb.shape[0]).to(self.device)
            stop_flag_matrix = stop_flag_matrix < 1         
            recon_patient_emb = 0
            gen_patient_emb = []
            gen_patient_mask = []
            for time_idx in range(0, self.max_time_step):  
                if time_idx == 0:
                    idx_batch = patient_emb.reshape(patient_emb.shape[0], -1)  
                    gen_patient_emb.append(idx_batch)
                    gen_patient_mask.append(patient_mask)
                else:  
                    idx_batch = recon_patient_emb

                    gen_patient_emb.append(recon_patient_emb)
                    
                    gen_patient_mask.append(stop_flag_matrix[0].unsqueeze(-1).repeat(1, 3))            

                if not stop_flag_matrix.any():  
                    break
                

                
                mu, logvar = torch.chunk(self.VAE_Encoder(idx_batch), 2, dim=1)
                std = torch.exp(0.5 * logvar)  
                eps = torch.randn_like(std)  

                z = mu + eps * std

                
                if time_idx == 0:  
                    memory_mu = mu
                    memory_std = std
                    memory_z = torch.sigmoid(z)

                
                query_multi = torch.chunk(self.Q_Linear(memory_z), self.multi_head, dim=1)  
                key_multi = torch.chunk(self.K_Linear(z), self.multi_head, dim=1)  
                value_multi = torch.chunk(self.V_Linear(z), self.multi_head, dim=1)  

                att_hidden_mean = 0
                for (query, key, value) in zip(query_multi, key_multi, value_multi):
                    
                    query = query / math.sqrt(self.hidden_dim)
                    attn = torch.mm(query, key.transpose(-2, -1))
                    attn = F.softmax(attn, dim=-1)
                    att_hidden = torch.mm(attn, value)
                    att_hidden_mean += att_hidden

                
                time_bn = F.softmax(att_hidden_mean / self.multi_head, dim=-1)
                time_z = F.sigmoid(z + time_bn)       

                
                
                

                
                recon_patient_emb = self.VAE_Decoder(time_z, memory=time_z)  

                
                IG = -torch.mean(time_z * torch.log(time_z + 1e-2), dim=1) + torch.mean(memory_z * torch.log(memory_z + 1e-2), dim=1)
                IG = self.Score_Linear(time_z - memory_z).sigmoid()
                
                
                
                
                stop_flag_matrix = stop_flag_matrix * (torch.sigmoid(IG+1e-4+self.bound)>1/2)
                
                stop_flag_matrix = stop_flag_matrix * (IG>1/2)

                memory_z = time_z

            patient_emb = [item.unsqueeze(1) for item in gen_patient_emb]
            patient_mask = [item.unsqueeze(1) for item in gen_patient_mask]

            generate_chain = torch.cat(patient_emb, dim=1)  
            generate_mask = torch.cat(patient_mask, dim=1)  
            y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)  

            self.generate_chain.append(generate_chain)
            self.generate_mask.append(generate_mask)
            self.generate_label.append(y_true)

            

        elif self.train_mode == 'MixTrain':
            
            patient_emb = torch.cat(patient_emb, dim=-1)  

            
            loss = 0
            if self.pretrain_epoch > 0:
                self.pretrain_epoch -= 1
                
                for synx_bs_idx in range(0, len(self.generate_label)):
                    synx_bs_chain = self.generate_chain[synx_bs_idx]
                    synx_bs_mask = self.generate_mask[synx_bs_idx]
                    synx_bs_label = self.generate_label[synx_bs_idx]
                    
                    
                    

                    _, synx_x = self.total_rnn(torch.Tensor(synx_bs_chain).to(self.device), torch.Tensor(synx_bs_mask[:, :, 0]).to(self.device))
                    synx_logits = self.fc(synx_x)
                    loss += self.get_loss_function()(synx_logits, torch.Tensor(synx_bs_label).to(self.device))

            _, x = self.total_rnn(patient_emb, patient_mask[0])    
            
            logits = self.fc(x)
            
            y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
            y_prob = self.prepare_y_prob(logits)

        if self.train_mode == 'Pretrain':
            rec_loss = torch.mean(patient_mask * (recon_patient_emb - patient_emb) ** 2)
            loss = rec_loss - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            results = {"name": self.train_mode, "loss": loss, "y_prob": torch.rand_like(y_true), "y_true": y_true, "logit": torch.ones_like(y_true)}
        elif self.train_mode == 'TrainChain':
            results = {"name": self.train_mode, "loss": loss, "y_prob": torch.rand_like(y_true), "y_true": y_true, "logit": torch.ones_like(y_true)}
        elif self.train_mode == 'GenerateChain':
            loss = self.get_loss_function()(torch.ones_like(y_true).to(self.device), y_true)       
            results = {"name": self.train_mode, "loss": loss, "y_prob": torch.rand_like(y_true), "y_true": y_true, "logit": torch.ones_like(y_true),
                       "generate_chain": generate_chain, "generate_mask": generate_mask, "generate_label": y_true}
        elif self.train_mode == 'MixTrain':
            loss += self.get_loss_function()(logits, y_true)
            results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


if __name__ == "__main__":
    from pyhealth.datasets import SampleEHRDataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            
            "list_codes": ["505800458", "50580045810", "50580045811"],  
            "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
            "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  
            "list_list_vectors": [
                [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
                [[7.7, 8.5, 9.4]],
            ],
            "label": 1,
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            
            "list_codes": [
                "55154191800",
                "551541928",
                "55154192800",
                "705182798",
                "70518279800",
            ],
            "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
            "list_list_codes": [["A04A", "B035", "C129"]],
            "list_list_vectors": [
                [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
            ],
            "label": 0,
        },
    ]

    
    dataset = SampleEHRDataset(samples=samples, dataset_name="test")

    
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    
    model = RNN(
        dataset=dataset,
        feature_keys=[
            "list_codes",
            "list_vectors",
            "list_list_codes",
            "list_list_vectors",
        ],
        label_key="label",
        mode="binary",
    )

    
    data_batch = next(iter(train_loader))

    
    ret = model(**data_batch)
    print(ret)

    
    ret["loss"].backward()
