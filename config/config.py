# config/config.py
import pandas as pd
import torch
from dotenv import dotenv_values


class ConfigService:
    CATEGORIES = ["2K", "100K", "FULL"]

    # Keys that have CPU_/GPU_ variants in the .env files
    _DEVICE_KEYS = [
        "BATCH_SIZE",
        "PRETRAIN_EPOCHS",
        "FINETUNE_EPOCHS",
        "PRETRAIN_PAIRS",
    ]

    _ENV_FILES = {
        "2K": ".env.2k",
        "100K": ".env.100k",
        "FULL": ".env.full",
    }

    def __init__(self, category: str = "2K"):
        category = category.upper()
        if category not in self._ENV_FILES:
            raise ValueError(
                f"Invalid category '{category}'. Choose from: {self.CATEGORIES}"
            )

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self._prefix = "GPU" if self.DEVICE == "cuda" else "CPU"
        self._envs = dotenv_values(self._ENV_FILES[category])

        # Eagerly resolve device-specific keys so they're accessible directly
        for key in self._DEVICE_KEYS:
            device_key = f"{self._prefix}_{key}"
            if device_key in self._envs:
                # Cast numeric values to the appropriate type
                raw = self._envs[device_key]
                try:
                    value = int(raw)
                except ValueError:
                    try:
                        value = float(raw)
                    except ValueError:
                        value = raw
                object.__setattr__(self, key, value)

    #     self.__label_distribution()

    # def __label_distribution(self):
    #     label_df = pd.read_csv(self._envs["LABEL_PATH"])
    #     train_normal = (label_df["Label"] == "Normal").sum()
    #     train_anomaly = (label_df["Label"] == "Anomaly").sum()

    #     print("The number of normal logs is ", train_normal)
    #     print("The number of anomaly logs is ", train_anomaly)

    #     self.NORMAL_COUNT = train_normal
    #     self.ANOMALY_COUNT = train_anomaly
    #     self.POS_WEIGHT = train_normal / train_anomaly


    def __getattr__(self, name: str):
        # Called only when normal attribute lookup fails
        envs = object.__getattribute__(self, "_envs")
        if name in envs:
            raw = envs[name]
            try:
                return int(raw)
            except ValueError:
                try:
                    return float(raw)
                except ValueError:
                    return raw
        raise AttributeError(f"ConfigService has no attribute '{name}'")

    def __repr__(self):
        lines = [f"ConfigService(device={self.DEVICE})"]
        for key in self._DEVICE_KEYS:
            lines.append(f"  {key} = {getattr(self, key, 'N/A')}")
        for k, v in self._envs.items():
            if not any(k.startswith(p) for p in ("CPU_", "GPU_")):
                lines.append(f"  {k} = {v}")
        return "\n".join(lines)
