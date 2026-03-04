# models/classifier.py
import torch.nn as nn


class LogClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim=768):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_ids, attention_mask, output_attentions=False):
        if output_attentions:
            emb, attentions = self.encoder(
                input_ids, attention_mask, output_attentions=True
            )
            logits = self.fc(emb).squeeze(-1)
            return logits, attentions

        emb = self.encoder(input_ids, attention_mask)
        return self.fc(emb).squeeze(-1)
