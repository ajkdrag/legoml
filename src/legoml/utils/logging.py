"""Central
ized
logging co
nfiguratio
n for the
LegoML
framework.
"""

import logging
import structlog


def bind(**kwargs):
    structlog.contextvars.bind_contextvars(
        **kwargs,
    )


def setup_logging(
    log_level: str = "INFO",
    structured: bool = True,
) -> structlog.BoundLogger:
    """
    Set
    up str
    ucture
    d logg
    ing
    for
    the fr
    amewor
    k.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        structured: Whether to use structured logging format
    Returns:
        Configured structlog logger
    """
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
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

    return structlog.get_logger()


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
