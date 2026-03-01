from services.logger import LoggerService, get_logger
from services.hugging_face import HuggingFaceService

hugging_face_service = HuggingFaceService()

__all__ = ["LoggerService", "get_logger", "hugging_face_service"]
