import docker
import subprocess
import sys
import os
import tempfile

class PythonExecutor:
    def __init__(self):
        self.use_docker = False
        try:
            self.client = docker.from_env()
            self.client.ping() # Check if Docker is actually running
            print("🐳 Docker is active. Building sandbox...")
            self.client.images.build(path=".", tag="codesentinel-sandbox")
            self.use_docker = True
        except Exception:
            print("⚠️ Docker Engine is stopped. Falling back to Local Execution.")

    def execute(self, code_string: str) -> dict:
    # Use a system temp directory instead of the Desktop folder
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as tmp:
            tmp.write(code_string)
            tmp_path = tmp.name  # This will be something like /tmp/tmp_xyz.py

        try:
        # Docker usually has full access to /tmp or /private/tmp on Mac
            container = self.client.containers.run(
            image="codesentinel-sandbox",
            command=["python", "/app/script.py"],
            volumes={tmp_path: {'bind': '/app/script.py', 'mode': 'ro'}},
            remove=True,
            stdout=True,
            stderr=True,
            network_disabled=True
        )
            return {"success": True, "output": container.decode('utf-8'), "error": ""}
        except docker.errors.ContainerError as e:
            return {"success": False, "output": "", "error": e.stderr.decode('utf-8')}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}
        finally:
        # Cleanup the temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def execute_local(self, code_string: str) -> dict:
        filename = "temp_agent_code.py"
        with open(filename, "w") as f:
            f.write(code_string)
        try:
            result = subprocess.run(
                [sys.executable, filename], 
                capture_output=True, text=True, timeout=5
            )
            return {"success": True, "output": result.stdout, "error": result.stderr}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def execute_docker(self, code_string: str) -> dict:
        # ... (keep your existing execute_docker code here)
        pass