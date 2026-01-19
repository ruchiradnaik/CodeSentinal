import docker
import subprocess
import sys
import os
import tempfile
import re

class PythonExecutor:
    def __init__(self, dependencies: set = None):
        self.use_docker = False
        self.client = None
        self.dependencies = dependencies or set()
        self.local_modules: set[str] = set()
        try:
            self.client = docker.from_env()
            self.client.ping() # Check if Docker is actually running
            print("🐳 Docker is active. Building sandbox...")
            # Build image with dependencies if provided
            self._build_docker_image()
            self.use_docker = True
        except Exception as e:
            print(f"⚠️ Docker Engine is stopped or unavailable ({str(e)[:50]}). Falling back to Local Execution.")
            self.client = None
    
    def _build_docker_image(self):
        """Build Docker image with dependencies pre-installed"""
        if self.dependencies:
            # Create Dockerfile content with dependencies
            deps_list = " ".join(sorted(self.dependencies))
            dockerfile_content = f"""FROM python:3.9-slim

# Install dependencies
RUN pip install --no-cache-dir {deps_list}

# Create a non-privileged user
RUN useradd -m botuser

# Set working directory
WORKDIR /home/botuser/app

# Switch to non-privileged user
USER botuser
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as df:
                df.write(dockerfile_content)
                dockerfile_path = df.name
            
            try:
                self.client.images.build(
                    path=".",
                    dockerfile=dockerfile_path,
                    tag="codesentinel-sandbox",
                    rm=True
                )
                print(f"✅ Docker image built with {len(self.dependencies)} dependencies pre-installed")
            except Exception as e:
                print(f"⚠️ Error building Docker image: {e}")
                # Fallback to standard image (still allows runtime installs)
                self.client.images.build(path=".", tag="codesentinel-sandbox")
            finally:
                if os.path.exists(dockerfile_path):
                    os.remove(dockerfile_path)
        else:
            # Build standard image
            self.client.images.build(path=".", tag="codesentinel-sandbox")

    def set_dependencies(self, dependencies: set):
        """Update dependencies and rebuild image if needed"""
        if dependencies != self.dependencies:
            self.dependencies = dependencies
            if self.use_docker and self.client:
                print("🔄 Rebuilding Docker image with new dependencies...")
                self._build_docker_image()

    def set_local_modules(self, local_modules: set):
        """Provide local module names so we never try to pip-install them."""
        self.local_modules = local_modules or set()

    def execute(self, code_string: str) -> dict:
        # Use Docker if available, otherwise fall back to local execution
        if self.use_docker and self.client is not None:
            return self.execute_docker(code_string)
        else:
            return self.execute_local(code_string)
    
    def execute_docker(self, code_string: str) -> dict:
        """Execute code in Docker container"""
        # Use a system temp directory instead of the Desktop folder
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as tmp:
            tmp.write(code_string)
            tmp_path = tmp.name  # This will be something like /tmp/tmp_xyz.py

        # Dependencies should already be pre-installed in the image
        # Only install missing ones at runtime if needed (with network access)
        inferred_modules = self._extract_imports(code_string)
        # Never try to install local project modules as pip packages
        inferred_modules = {m for m in inferred_modules if m not in self.local_modules}
        missing_modules = inferred_modules - self.dependencies
        
        install_cmd = ""
        if missing_modules:
            # Normalize common aliases at runtime just in case (defensive)
            alias_map = {"cv2": "opencv-python", "PIL": "Pillow", "sklearn": "scikit-learn", "yaml": "PyYAML", "dateutil": "python-dateutil", "dotenv": "python-dotenv"}
            normalized = {alias_map.get(m, m) for m in missing_modules}
            joined = " ".join(sorted(normalized))
            install_cmd = f"pip install -q {joined} && "

        try:
            logs = self.client.containers.run(
            image="codesentinel-sandbox",
            command=[
            "/bin/sh", "-c",
            f"""
            set -e
            {install_cmd}
            python /app/script.py
            """
            ],
            volumes={tmp_path: {'bind': '/app/script.py', 'mode': 'ro'}},
            remove=True,
            stdout=True,
            stderr=True,
            network_disabled=False
        )

            output = logs.decode("utf-8")

            # Real Python failure
            if "Traceback (most recent call last):" in output or "SyntaxError" in output:
                return {"success": False, "output": "", "error": output}

            # Otherwise success (warnings, pip notices are fine)
            return {"success": True, "output": output, "error": ""}

        except docker.errors.ContainerError as e:
            err = e.stderr.decode("utf-8") if e.stderr else str(e)

        # Dependency / pip failure
            if "pip install" in err or "Could not find a version" in err:
                return {
            "success": False,
            "output": "",
            "error": f"Dependency installation failed:\n{err}"
        }

    # Python runtime failure
            if "Traceback (most recent call last):" in err or "SyntaxError" in err:
                return {
            "success": False,
            "output": "",
            "error": f"Runtime error:\n{err}"
        }

    # Fallback: container-level failure
            return {
        "success": False,
        "output": "",
        "error": err
        }

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    
    def _is_only_pip_noise(self, text: str) -> bool:
        pip_noise_patterns = [
            "A new release of pip is available",
            "You should consider upgrading",
            "WARNING: Retrying",
        ]
        return all(any(p in line for p in pip_noise_patterns)
               for line in text.splitlines() if line.strip())


    def _extract_imports(self, code_string: str) -> set[str]:
        """
        Very lightweight import parser that looks for top-level `import x` and
        `from x import y` statements and returns the root module names. This is
        intentionally simple and conservative; it's fine if it misses some
        modules, but it should avoid obviously-invalid names.
        """
        modules: set[str] = set()

        # Regexes for `import x` and `from x import y`
        import_re = re.compile(r"^\s*import\s+([a-zA-Z0-9_.,\s]+)")
        from_re = re.compile(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import\s+")

        for line in code_string.splitlines():
            # Skip comments and empty lines early
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            m_import = import_re.match(line)
            if m_import:
                names_part = m_import.group(1)
                # Handle cases like: import numpy as np, pandas as pd
                for chunk in names_part.split(","):
                    base = chunk.strip().split()[0]  # drop "as alias"
                    if base and not base.startswith("."):
                        modules.add(base.split(".")[0])
                continue

            m_from = from_re.match(line)
            if m_from:
                pkg = m_from.group(1)
                if pkg and not pkg.startswith("."):
                    modules.add(pkg.split(".")[0])

        # 🔽 ADD THIS BLOCK RIGHT HERE 🔽
        stdlib_like = {
            "sys", "os", "re", "json", "time", "math", "hashlib",
            "subprocess", "typing", "pathlib"
        }

        modules = {m for m in modules if m not in stdlib_like}
        # 🔼 END ADDITION 🔼

        return modules


    def execute_local(self, code_string: str) -> dict:
        """Execute code locally using subprocess"""
        # Use a temporary file that will be cleaned up
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as tmp:
            tmp.write(code_string)
            tmp_path = tmp.name
        
        try:
            result = subprocess.run(
                [sys.executable, tmp_path], 
                capture_output=True, 
                text=True, 
                timeout=10,  # Increased timeout for complex code
                cwd=os.path.dirname(tmp_path)  # Run in temp directory
            )
            
            # Check if execution was successful (return code 0)
            if result.returncode == 0:
                return {"success": True, "output": result.stdout, "error": ""}
            else:
                return {"success": False, "output": result.stdout, "error": result.stderr}
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "", "error": "Execution timed out after 10 seconds"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}
        finally:
            # Cleanup the temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
