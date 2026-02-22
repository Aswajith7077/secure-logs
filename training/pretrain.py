# training/pretrain.py
from utils.loss import joint_loss


def pretrain(model, dataloader, optimizer, device="cpu", num_epochs=2):
    model.train()
    final_loss = 0.0

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
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

            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"[Pretrain] Epoch {epoch}/{num_epochs} — Loss: {avg_loss:.4f}")
        final_loss = avg_loss

    return final_loss
