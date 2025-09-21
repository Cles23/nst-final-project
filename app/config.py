
import os

class AppConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    STATIC_OUTPUT = os.getenv("STATIC_OUTPUT", "static/output")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
