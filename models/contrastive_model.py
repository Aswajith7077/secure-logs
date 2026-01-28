# models/contrastive_model.py
import torch
import torch.nn as nn


class LogContrastiveModel(nn.Module):
    def __init__(self, encoder, hidden_dim=768):
        super().__init__()
        self.encoder = encoder

        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, 2))

    def forward(self, batch_a, batch_b):
        """
        batch_a, batch_b: (input_ids, attention_mask)
        """
        emb_a = self.encoder(*batch_a)
        emb_b = self.encoder(*batch_b)

        diff = torch.abs(emb_a - emb_b)
        concat = torch.cat([emb_a, diff], dim=1)

        logits = self.classifier(concat)
        return logits, emb_a, emb_b
