from .config import ConfigService


config_service = ConfigService(ConfigService.CATEGORIES[1])

__all__ = ["config_service"]