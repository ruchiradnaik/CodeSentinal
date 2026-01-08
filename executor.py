import subprocess
import sys

class PythonExecutor:
    def execute(self, code_string: str) -> dict:
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