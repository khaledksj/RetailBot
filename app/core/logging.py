"""
Structured logging configuration for the Shop Manual Chatbot RAG system.
"""

import logging
import sys
from typing import Dict, Any
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present, avoiding conflicts with built-in fields
        if hasattr(record, "extra") and record.extra:
            for key, value in record.extra.items():
                # Avoid overwriting built-in LogRecord fields
                if key not in {"filename", "lineno", "module", "funcName", "pathname"}:
                    log_entry[key] = value
                else:
                    # Use a prefixed version for conflicting keys
                    log_entry[f"custom_{key}"] = value
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

def setup_logging(log_level: str = "INFO") -> None:
    """Setup structured logging configuration."""
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with structured formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    
    # Configure root logger
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with structured logging support."""
    return logging.getLogger(name)

class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding structured context to log messages."""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message and add extra context."""
        extra = kwargs.get("extra", {})
        if self.extra:
            extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs

def get_structured_logger(name: str, **context) -> LoggerAdapter:
    """Get a structured logger with default context."""
    logger = get_logger(name)
    return LoggerAdapter(logger, context)
