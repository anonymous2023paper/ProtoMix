import functools
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from pyhealth.datasets import BaseEHRDataset
from pyhealth.models import BaseModel


class DeeprLayer(nn.Module):

    def __init__(
        self,
        feature_size: int = 100,
        window: int = 1,
        hidden_size: int = 3,
    ):
        super(DeeprLayer, self).__init__()

        self.conv = torch.nn.Conv1d(
            feature_size, hidden_size, kernel_size=2 * window + 1
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = x.permute(0, 2, 1)   
        x = torch.relu(self.conv(x))
        x = x.max(-1)[0]
        return x


def _flatten_and_fill_gap(gap_embedding, batch, device):
    embed_dim = gap_embedding.shape[-1]
    batch = [
        [
            [torch.tensor(_, device=device, dtype=torch.float) for _ in _visit_x]
            for _visit_x in _pat_x
        ]
        for _pat_x in batch
    ]
    batch = [
        torch.stack(functools.reduce(lambda a, b: a + [gap_embedding] + b, _), 0)
        for _ in batch
    ]
    batch_max_length = max(map(len, batch))
    mask = torch.tensor(
        [[1] * len(x) + [0] * (batch_max_length - len(x)) for x in batch],
        dtype=torch.long,
        device=device,
    )
    out = torch.zeros(
        [len(batch), batch_max_length, embed_dim], device=device, dtype=torch.float
    )
    for i, x in enumerate(batch):
        out[i, : len(x)] = x
    return out, mask


class Deepr(BaseModel):

    def __init__(
        self,
        dataset: BaseEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super(Deepr, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

         
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
         
        self.embeddings = nn.ModuleDict()
         
        self.linear_layers = nn.ModuleDict()

         
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
             
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "Deepr only supports str code, float and int as input types"
                )
            if (input_info["type"] == str) and (input_info["dim"] != 3):
                raise ValueError("Deepr only supports 2-level str code as input types")
            if (input_info["type"] in [float, int]) and (input_info["dim"] != 3):
                raise ValueError(
                    "Deepr only supports 3-level float and int as input types"
                )
             
             
            self.add_feature_transform_layer(
                feature_key, input_info, special_tokens=["<pad>", "<unk>", "<gap>"]
            )
            if input_info["type"] != str:
                self.embeddings[feature_key] = torch.nn.Embedding(1, input_info["len"])

        self.cnn = nn.ModuleDict()
        for feature_key in feature_keys:
            self.cnn[feature_key] = DeeprLayer(
                feature_size=embedding_dim, hidden_size=hidden_dim, **kwargs
            )
        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_emb = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

             
            if (dim_ == 3) and (type_ == str):
                feature_vals = [
                    functools.reduce(lambda a, b: a + ["<gap>"] + b, _)
                    for _ in kwargs[feature_key]
                ]
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    feature_vals, padding=True, truncation=False
                )
                pad_idx = self.feat_tokenizers[feature_key].vocabulary("<pad>")
                mask = torch.tensor(
                    [[_code != pad_idx for _code in _pat] for _pat in x],
                    dtype=torch.long,
                    device=self.device,
                )
                 
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                 
                x = self.embeddings[feature_key](x)
             
            elif (dim_ == 3) and (type_ in [float, int]):
                gap_embedding = self.embeddings[feature_key](
                    torch.zeros(1, dtype=torch.long, device=self.device)
                )[0]
                x, mask = _flatten_and_fill_gap(
                    gap_embedding, kwargs[feature_key], self.device
                )
                 
                x = self.linear_layers[feature_key](x)
            else:
                raise NotImplementedError(
                    f"Deepr does not support this input format (dim={dim_}, type={type_})."
                )
             
            x = self.cnn[feature_key](x, mask)
            patient_emb.append(x)

         
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
            "single_vector": [1, 2, 3],
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
            "single_vector": [1, 5, 8],
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

     
    model = Deepr(
        dataset=dataset,
         
        feature_keys=["list_list_codes", "list_list_vectors"],
        label_key="label",
        mode="binary",
    ).to("cuda:0")

     
    data_batch = next(iter(train_loader))

     
    ret = model(**data_batch)
    print(ret)

     
    ret["loss"].backward()
