# models/bert_encoder.py
import torch
import torch.nn as nn
from transformers import BertModel


class LogBERTEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.freeze_bert = freeze_bert

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Returns mean-pooled log sequence embedding [B, 768].
        When BERT is frozen we skip autograd graph construction entirely
        with no_grad, which gives a significant speedup.
        """
        ctx = torch.no_grad() if self.freeze_bert else torch.enable_grad()
        with ctx:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * mask).sum(1) / mask.sum(1)
        return pooled
