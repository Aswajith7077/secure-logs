from data.dataset import HDFSPretrainDataset
from data.dataset import HDFSFinetuneDataset

pretrain_dataset = HDFSPretrainDataset(
    csv_path="data/HDFS/100k-structured.csv",
    tokenizer_name="bert-base-uncased",
    max_len=64,
    num_pairs=2000,
)

finetune_dataset = HDFSFinetuneDataset(
    csv_path="data/HDFS/100k-structured.csv",
    tokenizer_name="bert-base-uncased",
    max_len=64,
    label_path="data/HDFS/anomaly_label.csv",
)

