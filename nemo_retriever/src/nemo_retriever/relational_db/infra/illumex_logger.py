import logging
from pythonjsonlogger import jsonlogger
import os

fmt = "%(created)f %(levelname)s %(name)s %(message)s"
uvicorn_access_fmt = fmt + " %(method)s %(url)s %(statusCode)s"
rename_fields = {
    "created": "time",
    "levelname": "level",
    "name": "context",
    "message": "msg",
}


def init_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if os.environ["LMX_ENV"] == "development":
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s]: %(levelname)s : %(message)s"
        )
    else:
        formatter = jsonlogger.JsonFormatter(
            fmt=fmt,
            rename_fields=rename_fields,
        )
    handler.setFormatter(formatter)

    logger.addHandler(handler)


# https://github.com/encode/uvicorn/discussions/2027#discussioncomment-7628361
class UvicornJSONAccessFormatter(jsonlogger.JsonFormatter):
    def format(self, record: logging.LogRecord) -> str:
        client_addr, method, full_path, http_version, status_code = record.args  # type: ignore[misc]
        record.args = None
        record.__dict__.update(
            {
                "method": method,
                "url": full_path,
                "statusCode": status_code,
                "msg": "request completed",
            }
        )
        return super().format(record=record)


class UvicornJSONDefaultFormatter(jsonlogger.JsonFormatter):
    def format(self, record: logging.LogRecord) -> str:
        record.__dict__.pop("color_message", None)
        return super().format(record=record)


UVICORN_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": UvicornJSONDefaultFormatter,
            "fmt": fmt,
            "rename_fields": rename_fields,
        },
        "access": {
            "()": UvicornJSONAccessFormatter,
            "fmt": uvicorn_access_fmt,
            "rename_fields": rename_fields,
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {
            "handlers": ["access"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
