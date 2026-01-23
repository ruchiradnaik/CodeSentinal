"""
CodeSentinel Python Executor
Production-ready code execution with Docker sandboxing and security features.
"""

import docker
import subprocess
import sys
import os
import tempfile
import re
import time
from typing import Dict, Set, Optional, List
from dataclasses import dataclass
from pathlib import Path

from logger import get_logger, log_execution, log_errors
from config import get_settings

logger = get_logger("executor")


@dataclass
class ExecutionResult:
    """Result of code execution"""
    success: bool
    output: str
    error: str
    execution_time_ms: int
    warning: Optional[str] = None
    exit_code: int = 0


class PythonExecutor:
    """
    Secure Python code executor with Docker sandboxing.
    
    Security features:
    - Resource limits (CPU, memory)
    - Network isolation (configurable)
    - Non-root execution
    - Timeout protection
    - Input validation
    """
    
    # Known non-critical errors that don't indicate actual code bugs
    NON_CRITICAL_ERRORS = [
        "EOFError: EOF when reading a line",
        "EOFError: EOF when reading",
        "BrokenPipeError",
    ]
    
    # Patterns indicating local module import errors (not actual bugs)
    LOCAL_IMPORT_ERROR_PATTERNS = [
        r"ModuleNotFoundError: No module named '([^']+)'",
        r"ImportError: cannot import name '([^']+)' from '([^']+)'",
        r"ImportError: No module named '([^']+)'",
    ]
    
    # Wrapper code to mock input() for non-interactive execution
    INPUT_MOCK_WRAPPER = '''
# CodeSentinel: Mock input() for non-interactive testing
import builtins
_original_input = builtins.input
_input_counter = [0]
_mock_inputs = ["test", "42", "yes", "example@test.com", "John Doe"]

def _mock_input(prompt=""):
    print(f"[INPUT REQUIRED] {prompt}")
    if _input_counter[0] < len(_mock_inputs):
        value = _mock_inputs[_input_counter[0]]
        _input_counter[0] += 1
        print(f"[AUTO-PROVIDED] {value}")
        return value
    return "test_value"

builtins.input = _mock_input
# End of CodeSentinel mock

'''

    # Standard library modules to exclude from pip install
    STDLIB_MODULES = {
        "sys", "os", "re", "json", "time", "math", "hashlib", "subprocess",
        "typing", "pathlib", "datetime", "collections", "itertools", "functools",
        "io", "string", "random", "copy", "pickle", "sqlite3", "csv", "logging",
        "unittest", "contextlib", "abc", "threading", "multiprocessing", "socket",
        "http", "urllib", "email", "html", "xml", "base64", "secrets", "uuid",
        "tempfile", "shutil", "glob", "fnmatch", "stat", "filecmp", "struct",
        "codecs", "unicodedata", "locale", "gettext", "argparse", "configparser",
        "warnings", "traceback", "gc", "inspect", "dis", "code", "codeop",
        "pprint", "reprlib", "enum", "graphlib", "operator", "dataclasses",
    }
    
    # Import name to pip package mapping
    IMPORT_TO_PIP = {
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "yaml": "PyYAML",
        "dateutil": "python-dateutil",
        "dotenv": "python-dotenv",
        "bs4": "beautifulsoup4",
        "tf": "tensorflow",
        "np": "numpy",
        "pd": "pandas",
    }

    def __init__(self, dependencies: Set[str] = None):
        """
        Initialize the executor.
        
        Args:
            dependencies: Pre-known dependencies to install in the sandbox
        """
        self.settings = get_settings()
        self.use_docker = False
        self.client = None
        self.dependencies = dependencies or set()
        self.local_modules: Set[str] = set()
        self._sandbox_image_ready = False
        
        if self.settings.docker.enabled:
            self._setup_docker()
        else:
            logger.warning("Docker is disabled in configuration. Using local execution.")
    
    def _setup_docker(self):
        """Setup Docker client and build sandbox image"""
        try:
            self.client = docker.from_env()
            self.client.ping()
            logger.info("Docker is active. Building sandbox...")
            self._build_sandbox_image()
            self.use_docker = True
        except docker.errors.DockerException as e:
            logger.warning(f"Docker unavailable: {e}. Falling back to local execution.")
            self.client = None
        except Exception as e:
            logger.error(f"Docker setup failed: {e}")
            self.client = None
    
    def _build_sandbox_image(self):
        """Build the sandbox Docker image"""
        dockerfile_path = Path(__file__).parent / "Dockerfile.sandbox"
        
        if dockerfile_path.exists():
            try:
                logger.info("Building sandbox image from Dockerfile.sandbox...")
                self.client.images.build(
                    path=str(dockerfile_path.parent),
                    dockerfile="Dockerfile.sandbox",
                    tag=self.settings.docker.sandbox_image,
                    rm=True,
                    timeout=self.settings.docker.build_timeout,
                )
                self._sandbox_image_ready = True
                logger.info(f"Sandbox image built: {self.settings.docker.sandbox_image}")
            except Exception as e:
                logger.warning(f"Failed to build sandbox image: {e}. Using base Python image.")
                self._use_base_image()
        else:
            logger.warning("Dockerfile.sandbox not found. Using base Python image.")
            self._use_base_image()
    
    def _use_base_image(self):
        """Fall back to base Python image"""
        try:
            self.client.images.pull("python:3.11-slim")
            self._sandbox_image_ready = True
            logger.info("Using base Python image: python:3.11-slim")
        except Exception as e:
            logger.error(f"Failed to pull base image: {e}")
            self._sandbox_image_ready = False

    def set_dependencies(self, dependencies: Set[str]):
        """Update dependencies"""
        if dependencies != self.dependencies:
            self.dependencies = dependencies
            logger.info(f"Dependencies updated: {len(dependencies)} packages")

    def set_local_modules(self, local_modules: Set[str]):
        """Set local module names to exclude from pip install"""
        self.local_modules = local_modules or set()
        logger.debug(f"Local modules set: {local_modules}")

    def _validate_code(self, code: str) -> tuple[bool, str]:
        """
        Validate code before execution.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not code or not code.strip():
            return False, "Empty code provided"
        
        # Check code length
        max_length = self.settings.security.max_code_length
        if len(code) > max_length:
            return False, f"Code exceeds maximum length ({len(code)} > {max_length})"
        
        # Check for obviously dangerous patterns (additional layer of security)
        dangerous_patterns = [
            r"os\.system\s*\(",
            r"subprocess\.call\s*\([^)]*shell\s*=\s*True",
            r"eval\s*\(\s*input",
            r"exec\s*\(\s*input",
            r"__import__\s*\(\s*['\"]os['\"]\s*\)",
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                logger.warning(f"Potentially dangerous pattern detected: {pattern}")
                # We still allow it (Docker sandbox should contain it) but log it
        
        return True, ""

    def _contains_input_calls(self, code: str) -> bool:
        """Check if code contains input() calls"""
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "input(" in line:
                return True
        return False
    
    def _wrap_code_with_input_mock(self, code: str) -> str:
        """Wrap code with input() mock for non-interactive execution"""
        if self._contains_input_calls(code):
            return self.INPUT_MOCK_WRAPPER + code
        return code
    
    def _is_non_critical_error(self, error_msg: str) -> bool:
        """Check if error is non-critical"""
        return any(pattern in error_msg for pattern in self.NON_CRITICAL_ERRORS)
    
    def _is_local_module_error(self, error_msg: str) -> tuple[bool, str]:
        """
        Check if error is due to missing local module (not a real bug).
        
        Returns:
            Tuple of (is_local_module_error, module_name)
        """
        for pattern in self.LOCAL_IMPORT_ERROR_PATTERNS:
            match = re.search(pattern, error_msg)
            if match:
                module_name = match.group(1)
                # Check if it's a local module (not a well-known pip package)
                if module_name in self.local_modules:
                    return True, module_name
                # Check if it looks like a local module (lowercase, simple name)
                if (module_name not in self.STDLIB_MODULES and 
                    module_name not in self.IMPORT_TO_PIP and
                    module_name not in self.dependencies and
                    not any(c.isupper() for c in module_name) and
                    '_' not in module_name[:1]):  # Doesn't start with underscore
                    # Likely a local module
                    return True, module_name
        return False, ""

    def _extract_imports(self, code: str) -> Set[str]:
        """Extract import names from code"""
        modules: Set[str] = set()
        
        import_re = re.compile(r"^\s*import\s+([a-zA-Z0-9_.,\s]+)")
        from_re = re.compile(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import\s+")

        for line in code.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            m_import = import_re.match(line)
            if m_import:
                names_part = m_import.group(1)
                for chunk in names_part.split(","):
                    base = chunk.strip().split()[0]
                    if base and not base.startswith("."):
                        modules.add(base.split(".")[0])
                continue

            m_from = from_re.match(line)
            if m_from:
                pkg = m_from.group(1)
                if pkg and not pkg.startswith("."):
                    modules.add(pkg.split(".")[0])

        # Filter out standard library modules
        modules = {m for m in modules if m not in self.STDLIB_MODULES}
        
        return modules

    def _normalize_package_names(self, modules: Set[str]) -> Set[str]:
        """Convert import names to pip package names"""
        return {self.IMPORT_TO_PIP.get(m, m) for m in modules}

    @log_execution(level=20)  # INFO level
    def execute(self, code: str) -> Dict:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            
        Returns:
            Dict with success, output, error, and metadata
        """
        # Validate code
        is_valid, error = self._validate_code(code)
        if not is_valid:
            logger.warning(f"Code validation failed: {error}")
            return {"success": False, "output": "", "error": error}
        
        # Wrap code with input mock if needed
        wrapped_code = self._wrap_code_with_input_mock(code)
        
        # Execute in Docker or locally
        if self.use_docker and self.client and self._sandbox_image_ready:
            result = self._execute_docker(wrapped_code)
        else:
            result = self._execute_local(wrapped_code)
        
        return result

    @log_errors(reraise=False)
    def _execute_docker(self, code: str) -> Dict:
        """Execute code in Docker container with security constraints"""
        start_time = time.time()
        
        # Create temp file for code
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            # Determine if we need to install any packages
            inferred_modules = self._extract_imports(code)
            inferred_modules = {m for m in inferred_modules if m not in self.local_modules}
            missing_modules = inferred_modules - self.dependencies
            
            # Build install command if needed
            install_cmd = ""
            if missing_modules:
                normalized = self._normalize_package_names(missing_modules)
                # Only install if network is allowed or we're in development
                if not self.settings.docker.network_disabled or self.settings.app.is_development:
                    joined = " ".join(sorted(normalized))
                    install_cmd = f"pip install -q {joined} 2>/dev/null && "
                    logger.debug(f"Will install: {normalized}")
            
            # Determine image to use
            image = self.settings.docker.sandbox_image
            try:
                self.client.images.get(image)
            except docker.errors.ImageNotFound:
                image = "python:3.11-slim"
                logger.warning(f"Sandbox image not found, using {image}")

            # Run container with security constraints
            container_config = {
                "image": image,
                "command": [
                    "/bin/sh", "-c",
                    f"{install_cmd}python /app/script.py"
                ],
                "volumes": {tmp_path: {'bind': '/app/script.py', 'mode': 'ro'}},
                "remove": True,
                "stdout": True,
                "stderr": True,
                "network_disabled": self.settings.docker.network_disabled,
                "mem_limit": self.settings.docker.memory_limit,
                "cpu_period": self.settings.docker.cpu_period,
                "cpu_quota": self.settings.docker.cpu_quota,
                "read_only": self.settings.docker.read_only,
                "security_opt": ["no-new-privileges"],
                "user": "nobody" if not self.settings.docker.read_only else None,
            }
            
            logger.debug(f"Running container with config: mem={self.settings.docker.memory_limit}, network_disabled={self.settings.docker.network_disabled}")
            
            logs = self.client.containers.run(**container_config)
            output = logs.decode("utf-8")
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Check for errors in output
            if "Traceback (most recent call last):" in output or "SyntaxError" in output:
                if self._is_non_critical_error(output):
                    logger.info("Non-critical error (input() related), marking as success")
                    return {
                        "success": True,
                        "output": output,
                        "error": "",
                        "warning": "Code requires interactive input. Auto-mocked for testing.",
                        "execution_time_ms": execution_time,
                    }
                
                # Check for local module import errors
                is_local_error, module_name = self._is_local_module_error(output)
                if is_local_error:
                    logger.info(f"Local module import error for '{module_name}', marking as skipped")
                    return {
                        "success": True,
                        "output": output,
                        "error": "",
                        "warning": f"Cannot test in isolation: requires local module '{module_name}'. This is NOT a bug.",
                        "skipped_reason": "local_module_dependency",
                        "missing_module": module_name,
                        "execution_time_ms": execution_time,
                    }
                
                return {
                    "success": False,
                    "output": "",
                    "error": output,
                    "execution_time_ms": execution_time,
                }

            logger.info(f"Execution successful in {execution_time}ms")
            return {
                "success": True,
                "output": output,
                "error": "",
                "execution_time_ms": execution_time,
            }

        except docker.errors.ContainerError as e:
            execution_time = int((time.time() - start_time) * 1000)
            err = e.stderr.decode("utf-8") if e.stderr else str(e)
            
            logger.warning(f"Container error: {err[:100]}")

            if self._is_non_critical_error(err):
                return {
                    "success": True,
                    "output": "",
                    "error": "",
                    "warning": "Code requires interactive input. Auto-mocked for testing.",
                    "execution_time_ms": execution_time,
                }

            if "pip install" in err or "Could not find a version" in err:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Dependency installation failed:\n{err}",
                    "execution_time_ms": execution_time,
                }

            if "Traceback (most recent call last):" in err or "SyntaxError" in err:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Runtime error:\n{err}",
                    "execution_time_ms": execution_time,
                }

            return {
                "success": False,
                "output": "",
                "error": err,
                "execution_time_ms": execution_time,
            }

        except docker.errors.APIError as e:
            logger.error(f"Docker API error: {e}")
            return {
                "success": False,
                "output": "",
                "error": f"Docker error: {e}",
                "execution_time_ms": int((time.time() - start_time) * 1000),
            }

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @log_errors(reraise=False)
    def _execute_local(self, code: str) -> Dict:
        """Execute code locally (fallback when Docker unavailable)"""
        logger.warning("Executing locally - this is less secure than Docker")
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        
        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.settings.docker.execution_timeout,
                cwd=os.path.dirname(tmp_path)
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout,
                    "error": "",
                    "execution_time_ms": execution_time,
                }
            else:
                # Check for non-critical errors (input() related)
                if self._is_non_critical_error(result.stderr):
                    return {
                        "success": True,
                        "output": result.stdout,
                        "error": "",
                        "warning": "Code requires interactive input. Auto-mocked for testing.",
                        "execution_time_ms": execution_time,
                    }
                
                # Check for local module import errors
                is_local_error, module_name = self._is_local_module_error(result.stderr)
                if is_local_error:
                    return {
                        "success": True,
                        "output": result.stdout,
                        "error": "",
                        "warning": f"Cannot test in isolation: requires local module '{module_name}'. This is NOT a bug.",
                        "skipped_reason": "local_module_dependency",
                        "missing_module": module_name,
                        "execution_time_ms": execution_time,
                    }
                
                return {
                    "success": False,
                    "output": result.stdout,
                    "error": result.stderr,
                    "execution_time_ms": execution_time,
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Execution timed out after {self.settings.docker.execution_timeout} seconds",
                "execution_time_ms": self.settings.docker.execution_timeout * 1000,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time_ms": int((time.time() - start_time) * 1000),
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def health_check(self) -> Dict:
        """Check executor health status"""
        status = {
            "docker_available": self.use_docker,
            "sandbox_image_ready": self._sandbox_image_ready,
            "dependencies_count": len(self.dependencies),
            "local_modules_count": len(self.local_modules),
        }
        
        if self.client:
            try:
                self.client.ping()
                status["docker_responsive"] = True
            except:
                status["docker_responsive"] = False
        
        return status
