# models/bert_encoder.py
import torch
import torch.nn as nn
from transformers import BertModel


class LogBERTEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", freeze_bert=False, attn_implementation=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, attn_implementation=attn_implementation)
        self.freeze_bert = freeze_bert

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, output_attentions=False):
        """
        Returns mean-pooled log sequence embedding [B, 768].
        Optionally returns attention weights from the last layer [B, heads, seq, seq].
        """
        # We MUST ensure the graph is kept open for XAI/Captum to work, 
        # even if the weights themselves are frozen.
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        token_embeddings = outputs.last_hidden_state
        attentions = outputs.attentions if output_attentions else None

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * mask).sum(1) / mask.sum(1)

        if output_attentions:
            return pooled, attentions
        return pooled
