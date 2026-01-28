# models/bert_encoder.py
import torch.nn as nn
from transformers import BertModel


class LogBERTEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        """
        Returns pooled log sequence embedding
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling (as in paper)
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * mask).sum(1) / mask.sum(1)
        return pooled  # shape: [B, 768]
