from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit

 


class StageNetLayer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        chunk_size: int = 128,
        conv_size: int = 10,
        levels: int = 3,
        dropconnect: int = 0.3,
        dropout: int = 0.3,
        dropres: int = 0.3,
    ):
        super(StageNetLayer, self).__init__()

        self.dropout = dropout
        self.dropconnect = dropconnect
        self.dropres = dropres
        self.input_dim = input_dim
        self.hidden_dim = chunk_size * levels
        self.conv_dim = self.hidden_dim
        self.conv_size = conv_size
         
        self.levels = levels
        self.chunk_size = chunk_size

        self.kernel = nn.Linear(
            int(input_dim + 1), int(self.hidden_dim * 4 + levels * 2)
        )
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.zeros_(self.kernel.bias)
        self.recurrent_kernel = nn.Linear(
            int(self.hidden_dim + 1), int(self.hidden_dim * 4 + levels * 2)
        )
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.recurrent_kernel.bias)

        self.nn_scale = nn.Linear(int(self.hidden_dim), int(self.hidden_dim // 6))
        self.nn_rescale = nn.Linear(int(self.hidden_dim // 6), int(self.hidden_dim))
        self.nn_conv = nn.Conv1d(
            int(self.hidden_dim), int(self.conv_dim), int(conv_size), 1
        )
         

        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=dropout)
            self.nn_dropres = nn.Dropout(p=dropres)

    def cumax(self, x, mode="l2r"):
        if mode == "l2r":
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return x
        elif mode == "r2l":
            x = torch.flip(x, [-1])
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return torch.flip(x, [-1])
        else:
            return x

    def step(self, inputs, c_last, h_last, interval, device):
        x_in = inputs.to(device=device)

         
        interval = interval.unsqueeze(-1).to(device=device)
        x_out1 = self.kernel(torch.cat((x_in, interval), dim=-1)).to(device)
        x_out2 = self.recurrent_kernel(
            torch.cat((h_last.to(device=device), interval), dim=-1)
        )

        if self.dropconnect:
            x_out1 = self.nn_dropconnect(x_out1)
            x_out2 = self.nn_dropconnect_r(x_out2)
        x_out = x_out1 + x_out2
        f_master_gate = self.cumax(x_out[:, : self.levels], "l2r")
        f_master_gate = f_master_gate.unsqueeze(2).to(device=device)
        i_master_gate = self.cumax(x_out[:, self.levels : self.levels * 2], "r2l")
        i_master_gate = i_master_gate.unsqueeze(2)
        x_out = x_out[:, self.levels * 2 :]
        x_out = x_out.reshape(-1, self.levels * 4, self.chunk_size)
        f_gate = torch.sigmoid(x_out[:, : self.levels]).to(device=device)
        i_gate = torch.sigmoid(x_out[:, self.levels : self.levels * 2]).to(
            device=device
        )
        o_gate = torch.sigmoid(x_out[:, self.levels * 2 : self.levels * 3])
        c_in = torch.tanh(x_out[:, self.levels * 3 :]).to(device=device)
        c_last = c_last.reshape(-1, self.levels, self.chunk_size).to(device=device)
        overlap = (f_master_gate * i_master_gate).to(device=device)
        c_out = (
            overlap * (f_gate * c_last + i_gate * c_in)
            + (f_master_gate - overlap) * c_last
            + (i_master_gate - overlap) * c_in
        )
        h_out = o_gate * torch.tanh(c_out)
        c_out = c_out.reshape(-1, self.hidden_dim)
        h_out = h_out.reshape(-1, self.hidden_dim)
        out = torch.cat([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1)
        return out, c_out, h_out

    def forward(
        self,
        x: torch.tensor,
        time: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor]:
         
        batch_size, time_step, feature_dim = x.size()
        device = x.device
        if time == None:
            time = torch.ones(batch_size, time_step)
        time = time.reshape(batch_size, time_step)
        c_out = torch.zeros(batch_size, self.hidden_dim)
        h_out = torch.zeros(batch_size, self.hidden_dim)

        tmp_h = (
            torch.zeros_like(h_out, dtype=torch.float32)
            .view(-1)
            .repeat(self.conv_size)
            .view(self.conv_size, batch_size, self.hidden_dim)
        )
        tmp_dis = torch.zeros((self.conv_size, batch_size))
        h = []
        origin_h = []
        distance = []
        for t in range(time_step):
            out, c_out, h_out = self.step(x[:, t, :], c_out, h_out, time[:, t], device)
            cur_distance = 1 - torch.mean(
                out[..., self.hidden_dim : self.hidden_dim + self.levels], -1
            )
            origin_h.append(out[..., : self.hidden_dim])
            tmp_h = torch.cat(
                (
                    tmp_h[1:].to(device=device),
                    out[..., : self.hidden_dim].unsqueeze(0).to(device=device),
                ),
                0,
            )
            tmp_dis = torch.cat(
                (
                    tmp_dis[1:].to(device=device),
                    cur_distance.unsqueeze(0).to(device=device),
                ),
                0,
            )
            distance.append(cur_distance)

             
            local_dis = tmp_dis.permute(1, 0)
            local_dis = torch.cumsum(local_dis, dim=1)
            local_dis = torch.softmax(local_dis, dim=1)
            local_h = tmp_h.permute(1, 2, 0)
            local_h = local_h * local_dis.unsqueeze(1)

             
            local_theme = torch.mean(local_h, dim=-1)
            local_theme = self.nn_scale(local_theme).to(device)
            local_theme = torch.relu(local_theme)
            local_theme = self.nn_rescale(local_theme).to(device)
            local_theme = torch.sigmoid(local_theme)

            local_h = self.nn_conv(local_h).squeeze(-1)
            local_h = local_theme * local_h
            h.append(local_h)

        origin_h = torch.stack(origin_h).permute(1, 0, 2)
        rnn_outputs = torch.stack(h).permute(1, 0, 2)
        if self.dropres > 0.0:
            origin_h = self.nn_dropres(origin_h)
        rnn_outputs = rnn_outputs + origin_h
        rnn_outputs = rnn_outputs.contiguous().view(-1, rnn_outputs.size(-1))
        if self.dropout > 0.0:
            rnn_outputs = self.nn_dropout(rnn_outputs)

        output = rnn_outputs.contiguous().view(batch_size, time_step, self.hidden_dim)
        last_output = get_last_visit(output, mask)

        return last_output, output, torch.stack(distance)


class StageNet(BaseModel):

    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        time_keys: List[str] = None,
        embedding_dim: int = 128,
        chunk_size: int = 128,
        levels: int = 3,
        **kwargs,
    ):
        super(StageNet, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.levels = levels

         
        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")
        if time_keys is not None:
            if len(time_keys) != len(feature_keys):
                raise ValueError(
                    "time_keys should have the same length as feature_keys"
                )

         
        self.feat_tokenizers = {}
        self.time_keys = time_keys
        self.label_tokenizer = self.get_label_tokenizer()
         
        self.embeddings = nn.ModuleDict()
         
        self.linear_layers = nn.ModuleDict()

        self.stagenet = nn.ModuleDict()
         
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
             
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "StageNet only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "StageNet only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "StageNet only supports 2-dim or 3-dim float and int as input types"
                )

             
             
            self.add_feature_transform_layer(feature_key, input_info)
            self.stagenet[feature_key] = StageNetLayer(
                input_dim=embedding_dim,
                chunk_size=self.chunk_size,
                levels=self.levels,
                **kwargs,
            )

        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(
            len(self.feature_keys) * self.chunk_size * self.levels, output_size
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_emb = []
        distance = []
        mask_dict = {}
        for idx, feature_key in enumerate(self.feature_keys):
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

             
            if (dim_ == 2) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                 
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                 
                x = self.embeddings[feature_key](x)
                 
                mask = torch.any(x !=0, dim=2)
                mask_dict[feature_key] = mask

             
            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                 
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                 
                x = self.embeddings[feature_key](x)
                 
                x = torch.sum(x, dim=2)
                 
                mask = torch.any(x !=0, dim=2)
                mask[:, 0] = True
                mask_dict[feature_key] = mask

             
            elif (dim_ == 2) and (type_ in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                 
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                 
                x = self.linear_layers[feature_key](x)
                 
                mask = mask.bool().to(self.device)
                mask_dict[feature_key] = mask

             
            elif (dim_ == 3) and (type_ in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                 
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                 
                x = torch.sum(x, dim=2)
                x = self.linear_layers[feature_key](x)
                 
                mask = mask[:, :, 0]
                mask = mask.bool().to(self.device)
                mask_dict[feature_key] = mask

            else:
                raise NotImplementedError

            time = None
            if self.time_keys is not None:
                input_info = self.dataset.input_info[self.time_keys[idx]]
                dim_, type_ = input_info["dim"], input_info["type"]
                if (dim_ != 2) or (type_ not in [float, int]):
                    raise ValueError("Time interval must be 2-dim float or int.")
                time, _ = self.padding2d(kwargs[self.time_keys[idx]])
                time = torch.tensor(time, dtype=torch.float, device=self.device)
            x, _, cur_dis = self.stagenet[feature_key](x, time=time, mask=mask)
            patient_emb.append(x)
            distance.append(cur_dis)

        patient_emb = torch.cat(patient_emb, dim=1)
         
        logits = self.fc(patient_emb)
         
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)

        y_prob = self.prepare_y_prob(logits)
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        self.test_x = patient_emb
        self.test_y = y_true
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
            "list_vectors_time": [[0.0], [1.3]],
            "list_codes_time": [[0.0], [2.0], [1.3]],
            "list_list_codes_time": [[0.0], [1.5]],
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
            "list_vectors_time": [[0.0], [2.0], [1.0]],
            "list_codes_time": [[0.0], [2.0], [1.3], [1.0], [2.0]],
            "list_list_codes_time": [[0.0]],
        },
    ]

     
    dataset = SampleEHRDataset(samples=samples, dataset_name="test")

     
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

     
    model = StageNet(
        dataset=dataset,
        feature_keys=[
            "list_codes",
            "list_vectors",
            "list_list_codes",
             
        ],
        time_keys=["list_codes_time", "list_vectors_time", "list_list_codes_time"],
        label_key="label",
        mode="binary",
    )

     
    data_batch = next(iter(train_loader))

     
    ret = model(**data_batch)
    print(ret)

     
    ret["loss"].backward()
