import docker
import subprocess
import sys
import os
import tempfile

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
            error_msg = e.stderr.decode('utf-8') if hasattr(e, 'stderr') and e.stderr else str(e)
            return {"success": False, "output": "", "error": error_msg}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}
        finally:
            # Cleanup the temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

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