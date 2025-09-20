import logging

import structlog
from colorama import Fore, Style

COLOR_MAP = {
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "cyan": Fore.CYAN,
}


def custom_colorizer(logger, method_name, event_dict):
    """
    A processor that adds color to a log message based on a `color` keyword.

    Example:
        logger.info("This is a yellow message", color="yellow")
    """
    color_name = event_dict.pop("color", None)
    if color_name:
        # Get the colorama code from our map
        color_code = COLOR_MAP.get(color_name.lower())
        if color_code:
            event_dict["event"] = (
                f"{color_code}{Style.BRIGHT}{event_dict['event']}{Style.RESET_ALL}"
            )

    return event_dict


def bind(**kwargs):
    structlog.contextvars.bind_contextvars(
        **kwargs,
    )


def setup_logging(
    log_level: str = "INFO",
    structured: bool = True,
) -> structlog.BoundLogger:
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        custom_colorizer,
    ]

    if structured:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger()
    logger.info("Finished logging setup")
    return logger


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
