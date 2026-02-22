# training/finetune.py
import torch.nn.functional as F


def finetune(model, dataloader, optimizer, device="cpu", num_epochs=2):
    model.train()
    final_loss = 0.0

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
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

            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"[Finetune] Epoch {epoch}/{num_epochs} — Loss: {avg_loss:.4f}")
        final_loss = avg_loss

    return final_loss
