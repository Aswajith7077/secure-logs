# training/pretrain.py
from utils.loss import joint_loss


def pretrain(model, dataloader, optimizer, device="cpu"):
    model.train()

    for batch in dataloader:
        batch_a, batch_b, labels = batch
        batch_a = [x.to(device) for x in batch_a]
        batch_b = [x.to(device) for x in batch_b]
        labels = labels.to(device)

        logits, z1, z2 = model(batch_a, batch_b)
        loss = joint_loss(logits, labels, z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
