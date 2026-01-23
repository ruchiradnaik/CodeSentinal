"""
CodeSentinel Configuration Management
Centralized configuration with environment variable support and validation.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_env(key: str, default: str = None, required: bool = False) -> Optional[str]:
    """Get environment variable with validation"""
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(f"Required environment variable '{key}' is not set")
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: str = field(default_factory=lambda: get_env("OPENAI_API_KEY") or get_env("OPEN_API_KEY", required=True))
    model: str = field(default_factory=lambda: get_env("OPENAI_MODEL", "gpt-4o-mini"))
    temperature: float = 0.0
    max_tokens: int = field(default_factory=lambda: get_env_int("OPENAI_MAX_TOKENS", 4096))
    timeout: int = field(default_factory=lambda: get_env_int("OPENAI_TIMEOUT", 60))


@dataclass
class GitHubConfig:
    """GitHub API configuration"""
    token: Optional[str] = field(default_factory=lambda: get_env("GITHUB_TOKEN"))
    api_url: str = field(default_factory=lambda: get_env("GITHUB_API_URL", "https://api.github.com"))
    
    @property
    def is_configured(self) -> bool:
        return bool(self.token)


@dataclass
class DockerConfig:
    """Docker sandbox configuration"""
    enabled: bool = field(default_factory=lambda: get_env_bool("DOCKER_ENABLED", True))
    image_name: str = field(default_factory=lambda: get_env("DOCKER_IMAGE", "codesentinel-sandbox"))
    sandbox_image: str = field(default_factory=lambda: get_env("DOCKER_SANDBOX_IMAGE", "codesentinel-sandbox:latest"))
    
    # Resource limits
    memory_limit: str = field(default_factory=lambda: get_env("DOCKER_MEMORY_LIMIT", "256m"))
    cpu_period: int = field(default_factory=lambda: get_env_int("DOCKER_CPU_PERIOD", 100000))
    cpu_quota: int = field(default_factory=lambda: get_env_int("DOCKER_CPU_QUOTA", 50000))  # 50% CPU
    
    # Timeouts
    execution_timeout: int = field(default_factory=lambda: get_env_int("DOCKER_TIMEOUT", 30))
    build_timeout: int = field(default_factory=lambda: get_env_int("DOCKER_BUILD_TIMEOUT", 120))
    
    # Security settings
    network_disabled: bool = field(default_factory=lambda: get_env_bool("DOCKER_NETWORK_DISABLED", True))
    read_only: bool = field(default_factory=lambda: get_env_bool("DOCKER_READ_ONLY", False))
    privileged: bool = False  # Never allow privileged mode
    
    # Paths
    temp_dir: str = field(default_factory=lambda: get_env("DOCKER_TEMP_DIR", "/tmp/codesentinel"))


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = field(default_factory=lambda: get_env("LOG_LEVEL", "INFO"))
    format: str = field(default_factory=lambda: get_env(
        "LOG_FORMAT", 
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    ))
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_path: Optional[str] = field(default_factory=lambda: get_env("LOG_FILE"))
    max_size_mb: int = field(default_factory=lambda: get_env_int("LOG_MAX_SIZE_MB", 10))
    backup_count: int = field(default_factory=lambda: get_env_int("LOG_BACKUP_COUNT", 5))
    
    # Structured logging
    json_format: bool = field(default_factory=lambda: get_env_bool("LOG_JSON", False))


@dataclass
class AgentConfig:
    """Agent workflow configuration"""
    max_retries: int = field(default_factory=lambda: get_env_int("AGENT_MAX_RETRIES", 3))
    max_research_iterations: int = field(default_factory=lambda: get_env_int("AGENT_MAX_RESEARCH", 3))
    recursion_limit: int = field(default_factory=lambda: get_env_int("AGENT_RECURSION_LIMIT", 100))
    
    # File processing
    max_file_size_kb: int = field(default_factory=lambda: get_env_int("AGENT_MAX_FILE_SIZE_KB", 500))
    supported_extensions: List[str] = field(default_factory=lambda: [".py"])
    
    # Skip patterns
    skip_directories: List[str] = field(default_factory=lambda: [
        "__pycache__", ".git", "node_modules", "venv", ".venv", 
        "env", ".env", "dist", "build", ".pytest_cache"
    ])


@dataclass
class SecurityConfig:
    """Security configuration"""
    # Rate limiting
    rate_limit_enabled: bool = field(default_factory=lambda: get_env_bool("RATE_LIMIT_ENABLED", True))
    rate_limit_requests: int = field(default_factory=lambda: get_env_int("RATE_LIMIT_REQUESTS", 100))
    rate_limit_window: int = field(default_factory=lambda: get_env_int("RATE_LIMIT_WINDOW", 3600))  # 1 hour
    
    # Input validation
    max_code_length: int = field(default_factory=lambda: get_env_int("MAX_CODE_LENGTH", 100000))
    max_files_per_request: int = field(default_factory=lambda: get_env_int("MAX_FILES_PER_REQUEST", 50))
    
    # Secrets (redact these in logs)
    secret_patterns: List[str] = field(default_factory=lambda: [
        r"ghp_[a-zA-Z0-9]+",  # GitHub tokens
        r"sk-[a-zA-Z0-9]+",   # OpenAI keys
        r"password[=:\s]+\S+",
        r"token[=:\s]+\S+",
        r"api[_-]?key[=:\s]+\S+",
    ])


@dataclass
class AppConfig:
    """Application configuration"""
    name: str = "CodeSentinel"
    version: str = "1.0.0"
    environment: str = field(default_factory=lambda: get_env("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: get_env_bool("DEBUG", False))
    
    # Server
    host: str = field(default_factory=lambda: get_env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: get_env_int("PORT", 8501))
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = field(default_factory=lambda: Path(get_env("DATA_DIR", "./data")))
    logs_dir: Path = field(default_factory=lambda: Path(get_env("LOGS_DIR", "./logs")))
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"


@dataclass 
class Settings:
    """Main settings container"""
    app: AppConfig = field(default_factory=AppConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check required API key
        if not self.openai.api_key:
            errors.append("OpenAI API key is required (OPENAI_API_KEY or OPEN_API_KEY)")
        
        # Warn about missing GitHub token
        if not self.github.is_configured:
            errors.append("GitHub token not configured - some features will be limited")
        
        # Validate Docker settings
        if self.docker.enabled:
            if self.docker.privileged:
                errors.append("Docker privileged mode must be disabled for security")
        
        return errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary (for logging, excluding secrets)"""
        return {
            "app": {
                "name": self.app.name,
                "version": self.app.version,
                "environment": self.app.environment,
                "debug": self.app.debug,
            },
            "docker": {
                "enabled": self.docker.enabled,
                "memory_limit": self.docker.memory_limit,
                "network_disabled": self.docker.network_disabled,
            },
            "agent": {
                "max_retries": self.agent.max_retries,
                "recursion_limit": self.agent.recursion_limit,
            },
            "logging": {
                "level": self.logging.level,
            }
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment"""
    global settings
    load_dotenv(override=True)
    settings = Settings()
    return settings
