import torch
import torch.nn.functional as F
from tqdm import tqdm
from services.logger import get_logger

log = get_logger(__name__)


def finetune(model, dataloader, optimizer, device="cpu", num_epochs=2):
    model.train()
    final_loss = 0.0
    total_batches = len(dataloader)
    loss_weights = torch.tensor([dataloader.dataset.NORMAL_COUNT/dataloader.dataset.ANOMALY_COUNT]).to(device)

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        pbar = tqdm(
            dataloader,
            desc=f"[Finetune] Epoch {epoch}/{num_epochs}",
            unit="batch",
            leave=True,
        )
        for input_ids, mask, labels in pbar:
            input_ids, mask, labels = (
                input_ids.to(device),
                mask.to(device),
                labels.to(device),
            )

            logits = model(input_ids, mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels.float(),pos_weight=loss_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(total_batches, 1)
        log.info("[Finetune] Epoch %d/%d — Avg Loss: %.4f", epoch, num_epochs, avg_loss)
        final_loss = avg_loss

    return final_loss
