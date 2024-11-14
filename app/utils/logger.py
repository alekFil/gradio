import logging
import os
from datetime import datetime, timedelta, timezone


# Класс для форматирования времени в заданной временной зоне
class UTC6Formatter(logging.Formatter):
    converter = datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created, timezone(timedelta(hours=6)))  # UTC+6
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()  # стандартный ISO формат


# Функция для настройки логирования с тайм-зоной UTC+6
def setup_logger(name: str, log_file: str, level=logging.DEBUG):
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(level)

    # Указываем нужный формат для логов и добавляем UTC+6 к времени
    formatter = UTC6Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
