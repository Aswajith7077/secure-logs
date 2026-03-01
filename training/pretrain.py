# training/pretrain.py
from tqdm import tqdm
from utils.loss import joint_loss
from services.logger import get_logger

log = get_logger(__name__)


def pretrain(model, dataloader, optimizer, device="cpu", num_epochs=2):
    model.train()
    final_loss = 0.0
    total_batches = len(dataloader)

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        pbar = tqdm(
            dataloader,
            desc=f"[Pretrain] Epoch {epoch}/{num_epochs}",
            unit="batch",
            leave=True,
        )
        for batch in pbar:
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
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(total_batches, 1)
        log.info("[Pretrain] Epoch %d/%d — Avg Loss: %.4f", epoch, num_epochs, avg_loss)
        final_loss = avg_loss

    return final_loss
