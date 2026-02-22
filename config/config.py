# config/config.py
import torch

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64
LR = 2e-5

DATA_PATH = "data/HDFS/2k-structured.csv"
LABEL_PATH = "data/HDFS/anomaly_label.csv"  # ground-truth block labels
SAVE_DIR = "ai-models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Scale down for CPU; increase these when running on GPU
BATCH_SIZE = 8
PRETRAIN_EPOCHS = 1
FINETUNE_EPOCHS = 1
PRETRAIN_PAIRS = 200  # total contrastive pairs (keep small for CPU)
