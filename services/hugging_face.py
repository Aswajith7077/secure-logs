from huggingface_hub import login, upload_folder, snapshot_download
from config import config_service as cfg
from services.logger import get_logger

log = get_logger(__name__)

# (optional) Login with your Hugging Face credentials


# Push your model files

class HuggingFaceService:
    def __init__(self):
        self.folder_path = cfg.MODELS_DIR
        self.repo_id = cfg.HUGGING_FACE_REPO_ID
        self.repo_type = "model"
        login()
    
    def push_model(self):
        try:
            log.info("Initialize Pushing model to Hugging Face")
            upload_folder(folder_path=self.folder_path, repo_id=self.repo_id, repo_type=self.repo_type)
            log.info("Model pushed to Hugging Face")
        except Exception as e:
            log.error("Failed to push model to Hugging Face: %s", e)

    def pull_model(self):
        try:
            log.info("Initialize Pulling model from Hugging Face")
            local_dir = snapshot_download(
                repo_id=cfg.HUGGING_FACE_REPO_ID,
                repo_type=self.repo_type,
                local_dir=self.folder_path,
                local_dir_use_symlinks=False
            )
            log.info("Model pulled from Hugging Face")
            return local_dir
        except Exception as e:
            log.error("Failed to pull model from Hugging Face: %s", e)
