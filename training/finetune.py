# training/finetune.py
import torch.nn.functional as F


def finetune(model, dataloader, optimizer, device):
    model.train()

    for input_ids, mask, labels in dataloader:
        input_ids, mask, labels = (
            input_ids.to(device),
            mask.to(device),
            labels.to(device),
        )

        logits = model(input_ids, mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
