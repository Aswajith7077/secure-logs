# main.py
import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import config_service as cfg
from data.dataset import HDFSPretrainDataset, HDFSFinetuneDataset
from models.bert_encoder import LogBERTEncoder
from models.contrastive_model import LogContrastiveModel
from models.classifier import LogClassifier
from training.pretrain import pretrain
from training.finetune import finetune
from retrieval.knn_index import build_index
from services import get_logger
from services import hugging_face_service

log = get_logger(__name__)


def main():
    log.info("=" * 60)
    log.info("  LogSentry — Training Pipeline")
    log.info("=" * 60)
    log.info("Device: %s", cfg.DEVICE)

    # ── 1. Data Loading ───────────────────────────────────────────
    log.info("[Data] Loading pre-training dataset...")
    pretrain_ds = HDFSPretrainDataset(
        csv_path=cfg.DATA_PATH,
        tokenizer_name=cfg.MODEL_NAME,
        max_len=cfg.MAX_LEN,
        num_pairs=cfg.PRETRAIN_PAIRS,
    )
    pretrain_loader = DataLoader(
        pretrain_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(cfg.DEVICE == "cuda"),
    )
    log.info("[Data] Pre-train pairs: %d", len(pretrain_ds))

    log.info("[Data] Loading fine-tuning dataset...")
    finetune_ds = HDFSFinetuneDataset(
        csv_path=cfg.DATA_PATH,
        tokenizer_name=cfg.MODEL_NAME,
        max_len=cfg.MAX_LEN,
        label_path=cfg.LABEL_PATH,
    )
    finetune_loader = DataLoader(
        finetune_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(cfg.DEVICE == "cuda"),
    )
    log.info("[Data] Fine-tune samples: %d", len(finetune_ds))

    # ── 2. Pre-training ───────────────────────────────────────────
    log.info("[Pretrain] Starting contrastive pre-training...")
    encoder = LogBERTEncoder(model_name=cfg.MODEL_NAME, freeze_bert=True).to(cfg.DEVICE)
    contrastive_model = LogContrastiveModel(encoder=encoder).to(cfg.DEVICE)
    trainable = [p for p in contrastive_model.parameters() if p.requires_grad]
    log.info(
        "[Pretrain] Trainable params: %s (BERT frozen)",
        f"{sum(p.numel() for p in trainable):,}",
    )
    pretrain_optimizer = AdamW(trainable, lr=cfg.LR)

    pretrain(
        model=contrastive_model,
        dataloader=pretrain_loader,
        optimizer=pretrain_optimizer,
        device=cfg.DEVICE,
        num_epochs=cfg.PRETRAIN_EPOCHS,
    )

    # ── 3. Fine-tuning ────────────────────────────────────────────
    log.info("[Finetune] Starting supervised fine-tuning...")
    classifier = LogClassifier(encoder=encoder).to(cfg.DEVICE)
    trainable_ft = [p for p in classifier.parameters() if p.requires_grad]
    log.info(
        "[Finetune] Trainable params: %s (BERT frozen)",
        f"{sum(p.numel() for p in trainable_ft):,}",
    )
    finetune_optimizer = AdamW(trainable_ft, lr=cfg.LR)

    finetune(
        model=classifier,
        dataloader=finetune_loader,
        optimizer=finetune_optimizer,
        device=cfg.DEVICE,
        num_epochs=cfg.FINETUNE_EPOCHS,
    )

    # ── 4. Save models ────────────────────────────────────────────
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    encoder_path = os.path.join(cfg.SAVE_DIR, "bert_encoder.pt")
    classifier_path = os.path.join(cfg.SAVE_DIR, "log_classifier.pt")

    torch.save(encoder.state_dict(), encoder_path)
    torch.save(classifier.state_dict(), classifier_path)

    log.info("Saved encoder    → %s", encoder_path)
    log.info("Saved classifier → %s", classifier_path)

    # ── 5. Build FAISS KNN vector store ───────────────────────────
    build_index(
        encoder=encoder,
        dataloader=finetune_loader,
        device=cfg.DEVICE,
        save_dir=cfg.SAVE_DIR,
    )

    # ── 6. Push models to Hugging Face ────────────────────────────
    # hugging_face_service.push_model()

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
