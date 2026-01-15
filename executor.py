import docker
import subprocess
import sys
import os
import tempfile
import re

class PythonExecutor:
    def __init__(self):
        self.use_docker = False
        self.client = None
        try:
            self.client = docker.from_env()
            self.client.ping() # Check if Docker is actually running
            print("🐳 Docker is active. Building sandbox...")
            self.client.images.build(path=".", tag="codesentinel-sandbox")
            self.use_docker = True
        except Exception as e:
            print(f"⚠️ Docker Engine is stopped or unavailable ({str(e)[:50]}). Falling back to Local Execution.")
            self.client = None

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

        # Try to infer required third-party modules from the code and install them
        # inside the ephemeral container before execution. This avoids treating
        # missing libraries (e.g., pandas, numpy, cv2) as code bugs.
        inferred_modules = self._extract_imports(code_string)
        install_cmd = ""
        if inferred_modules:
            # Use `pip` inside the container; installation will be local to the
            # container filesystem and discarded once the container is removed.
            # We keep it quiet to reduce log noise.
            joined = " ".join(sorted(inferred_modules))
            install_cmd = f"pip install -q {joined} && "

        try:
            # Docker usually has full access to /tmp or /private/tmp on Mac
            container = self.client.containers.run(
                image="codesentinel-sandbox",
                command=["/bin/sh", "-c", f"{install_cmd}python /app/script.py"],
                volumes={tmp_path: {'bind': '/app/script.py', 'mode': 'ro'}},
                remove=True,
                stdout=True,
                stderr=True,
                network_disabled=True
            )
            return {"success": True, "output": container.decode('utf-8'), "error": ""}
        except docker.errors.ContainerError as e:
            error_msg = e.stderr.decode('utf-8') if hasattr(e, 'stderr') and e.stderr else str(e)
            return {"success": False, "output": "", "error": error_msg}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}
        finally:
            # Cleanup the temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

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

        # We could optionally filter out a small known subset of stdlib modules
        # to reduce unnecessary installs, but it's generally harmless if pip
        # is asked to install an already-available module.
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