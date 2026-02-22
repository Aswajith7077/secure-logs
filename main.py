# main.py
import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config.config as cfg
from data.dataset import HDFSPretrainDataset, HDFSFinetuneDataset
from models.bert_encoder import LogBERTEncoder
from models.contrastive_model import LogContrastiveModel
from models.classifier import LogClassifier
from training.pretrain import pretrain
from training.finetune import finetune
from retrieval.knn_index import build_index


def main():
    print(f"Device: {cfg.DEVICE}")

    # ── 1. Data Loading ───────────────────────────────────────────
    print("\n[Data] Loading pre-training dataset...")
    pretrain_ds = HDFSPretrainDataset(
        csv_path=cfg.DATA_PATH,
        tokenizer_name=cfg.MODEL_NAME,
        max_len=cfg.MAX_LEN,
        num_pairs=cfg.PRETRAIN_PAIRS,
    )
    pretrain_loader = DataLoader(pretrain_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    print(f"[Data] Pre-train pairs: {len(pretrain_ds)}")

    print("\n[Data] Loading fine-tuning dataset...")
    finetune_ds = HDFSFinetuneDataset(
        csv_path=cfg.DATA_PATH,
        tokenizer_name=cfg.MODEL_NAME,
        max_len=cfg.MAX_LEN,
        label_path=cfg.LABEL_PATH,
    )
    finetune_loader = DataLoader(finetune_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    print(f"[Data] Fine-tune samples: {len(finetune_ds)}")

    # ── 2. Pre-training ───────────────────────────────────────────
    print("\n[Pretrain] Starting contrastive pre-training...")
    encoder = LogBERTEncoder(model_name=cfg.MODEL_NAME, freeze_bert=True).to(cfg.DEVICE)
    contrastive_model = LogContrastiveModel(encoder=encoder).to(cfg.DEVICE)
    trainable = [p for p in contrastive_model.parameters() if p.requires_grad]
    print(
        f"[Pretrain] Trainable params: {sum(p.numel() for p in trainable):,} (BERT frozen)"
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
    print("\n[Finetune] Starting supervised fine-tuning...")
    classifier = LogClassifier(encoder=encoder).to(cfg.DEVICE)
    trainable_ft = [p for p in classifier.parameters() if p.requires_grad]
    print(
        f"[Finetune] Trainable params: {sum(p.numel() for p in trainable_ft):,} (BERT frozen)"
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

    print(f"\n✓ Saved encoder    → {encoder_path}")
    print(f"✓ Saved classifier → {classifier_path}")

    # ── 5. Build FAISS KNN vector store ───────────────────────────
    # Encode every fine-tune training session with the frozen encoder
    # and persist the resulting embedding vectors + labels to disk.
    # This index is used at inference time: encode a new log → find
    # k nearest stored embeddings → weighted vote on their labels.
    build_index(
        encoder=encoder,
        dataloader=finetune_loader,
        device=cfg.DEVICE,
        save_dir=cfg.SAVE_DIR,
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
