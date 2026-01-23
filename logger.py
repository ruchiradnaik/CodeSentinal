"""
CodeSentinel Logging Framework
Production-ready logging with structured output, rotation, and security features.
"""

import logging
import logging.handlers
import sys
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict
from functools import wraps
import time
import traceback


class SecretFilter(logging.Filter):
    """Filter to redact sensitive information from logs"""
    
    SECRET_PATTERNS = [
        (r'ghp_[a-zA-Z0-9]{36,}', '[GITHUB_TOKEN]'),
        (r'gho_[a-zA-Z0-9]{36,}', '[GITHUB_OAUTH]'),
        (r'sk-[a-zA-Z0-9]{48,}', '[OPENAI_KEY]'),
        (r'sk-proj-[a-zA-Z0-9-_]{48,}', '[OPENAI_PROJECT_KEY]'),
        (r'(password[=:\s]+)[^\s]+', r'\1[REDACTED]'),
        (r'(token[=:\s]+)[^\s]+', r'\1[REDACTED]'),
        (r'(api[_-]?key[=:\s]+)[^\s]+', r'\1[REDACTED]'),
        (r'(secret[=:\s]+)[^\s]+', r'\1[REDACTED]'),
        (r'(authorization[=:\s]+)[^\s]+', r'\1[REDACTED]'),
        (r'(bearer\s+)[^\s]+', r'\1[REDACTED]'),
    ]
    
    def __init__(self):
        super().__init__()
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in self.SECRET_PATTERNS
        ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Redact secrets from log message"""
        if hasattr(record, 'msg') and record.msg:
            message = str(record.msg)
            for pattern, replacement in self._compiled_patterns:
                message = pattern.sub(replacement, message)
            record.msg = message
        
        # Also check args
        if record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern, replacement in self._compiled_patterns:
                        arg = pattern.sub(replacement, arg)
                new_args.append(arg)
            record.args = tuple(new_args)
        
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info[0] else None,
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['msg', 'args', 'created', 'filename', 'funcName', 'levelname',
                          'levelno', 'lineno', 'module', 'msecs', 'message', 'name',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName']:
                log_data[key] = value
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{self.BOLD}{record.levelname}{self.RESET}"
        
        return super().format(record)


class LoggerManager:
    """Centralized logger management"""
    
    _instance: Optional['LoggerManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._loggers: Dict[str, logging.Logger] = {}
        self._root_logger: Optional[logging.Logger] = None
    
    def setup(
        self,
        level: str = "INFO",
        log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S",
        log_file: Optional[str] = None,
        max_size_mb: int = 10,
        backup_count: int = 5,
        json_output: bool = False,
        colored_output: bool = True,
    ) -> logging.Logger:
        """
        Setup the logging system.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Format string for log messages
            date_format: Format string for timestamps
            log_file: Path to log file (optional)
            max_size_mb: Maximum log file size before rotation
            backup_count: Number of backup files to keep
            json_output: Use JSON format for file logs
            colored_output: Use colored output for console
            
        Returns:
            Root logger instance
        """
        # Get numeric level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # Create root logger
        root_logger = logging.getLogger("codesentinel")
        root_logger.setLevel(numeric_level)
        
        # Remove existing handlers
        root_logger.handlers.clear()
        
        # Add secret filter
        secret_filter = SecretFilter()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.addFilter(secret_filter)
        
        if colored_output and sys.stdout.isatty():
            console_formatter = ColoredFormatter(log_format, datefmt=date_format)
        else:
            console_formatter = logging.Formatter(log_format, datefmt=date_format)
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_size_mb * 1024 * 1024,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(numeric_level)
            file_handler.addFilter(secret_filter)
            
            if json_output:
                file_formatter = JSONFormatter()
            else:
                file_formatter = logging.Formatter(log_format, datefmt=date_format)
            
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        self._root_logger = root_logger
        
        # Log startup
        root_logger.info(f"Logging initialized: level={level}, file={log_file}")
        
        return root_logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a named logger"""
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(f"codesentinel.{name}")
        self._loggers[name] = logger
        return logger


# Global logger manager
_logger_manager = LoggerManager()


def setup_logging(**kwargs) -> logging.Logger:
    """Setup the logging system"""
    return _logger_manager.setup(**kwargs)


def get_logger(name: str = "main") -> logging.Logger:
    """Get a named logger"""
    return _logger_manager.get_logger(name)


# Convenience decorators
def log_execution(logger: logging.Logger = None, level: int = logging.DEBUG):
    """Decorator to log function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            func_name = func.__name__
            logger.log(level, f"Entering {func_name}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.log(level, f"Exiting {func_name} (took {elapsed:.3f}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.exception(f"Error in {func_name} after {elapsed:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


def log_errors(logger: logging.Logger = None, reraise: bool = True):
    """Decorator to log exceptions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Error in {func.__name__}: {e}")
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator


# Auto-setup with defaults if not already configured
def _auto_setup():
    """Auto-setup logging with environment variables"""
    import os
    
    level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE")
    json_output = os.getenv("LOG_JSON", "").lower() in ("true", "1", "yes")
    
    setup_logging(
        level=level,
        log_file=log_file,
        json_output=json_output,
    )


# Setup logging on import
_auto_setup()
