# models/bert_encoder.py
import torch.nn as nn
from transformers import BertModel


class LogBERTEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

        # Freeze BERT backbone so only the small task heads are trained.
        # This is standard feature-extraction transfer learning and makes
        # CPU training finish in seconds. Set freeze_bert=False on GPU
        # to fine-tune the full model.
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Returns mean-pooled log sequence embedding [B, 768].
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * mask).sum(1) / mask.sum(1)
        return pooled
