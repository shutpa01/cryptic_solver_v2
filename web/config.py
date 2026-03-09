"""Flask configuration."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"

PUZZLES_PER_PAGE = 30


class Config:
    SECRET_KEY = "dev-secret-change-in-prod"
    ADMIN_KEY = os.environ.get("ADMIN_KEY", "dev-admin-key")
    CLUES_DB = str(CLUES_DB)
    PUZZLES_PER_PAGE = PUZZLES_PER_PAGE


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
