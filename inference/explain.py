# inference/explain.py
import os
import sys
import torch
from transformers import BertTokenizer

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config_service as cfg
from models.bert_encoder import LogBERTEncoder
from models.classifier import LogClassifier
from utils.explain import explain_main

def main():
    device = cfg.DEVICE
    save_dir = cfg.SAVE_DIR # "ai-models"
    
    # 1. Load trained items
    encoder_path = os.path.join(save_dir, "bert_encoder.pt")
    classifier_path = os.path.join(save_dir, "log_classifier.pt")
    
    if not os.path.exists(encoder_path):
        print(f"Error: Model not found at {encoder_path}. Please run main.py first.")
        return

    print(f"Loading models from {save_dir}...")
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL_NAME)
    encoder = LogBERTEncoder(
        model_name=cfg.MODEL_NAME, 
        freeze_bert=True, 
        attn_implementation="eager"
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    
    classifier = LogClassifier(encoder=encoder).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))

    # 2. Pick a sample (this could be a known anomaly)
    # For demonstration, we'll use a sample anomalous-looking log sequence
    sample_log = (
        "Received block blk_-1608999688391216206 of size 67108864 from /10.250.19.102 [SEP] "
        "Verification succeeded for blk_-1608999688391216206 [SEP] "
        "java.io.IOException: Connection reset by peer [SEP] "
        "Failed to write block blk_-1608999688391216206 to mirror 10.251.126.5:50010"
    )
    
    print("\nProcessing sample log for explanation...")
    print(f"Log: {sample_log[:100]}...")
    
    # 3. Generate explanations
    explain_main(
        model=classifier,
        tokenizer=tokenizer,
        text=sample_log,
        device=device,
        save_dir="result/xai_explanations"
    )
    
    print("\nExplanation complete! Results saved in result/xai_explanations/")

if __name__ == "__main__":
    main()
