from .config import ConfigService


config_service = ConfigService(ConfigService.CATEGORIES[0])

__all__ = ["config_service"]