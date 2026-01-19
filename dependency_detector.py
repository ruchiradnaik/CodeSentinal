"""
Dependency Detector: Automatically detects Python dependencies from codebase
"""
import os
import re
import ast
import sys
from typing import Set, List, Dict
from collections import defaultdict


class DependencyDetector:
    """Detects Python dependencies from codebase"""

    # Explicit import-name → pip-name mapping
    IMPORT_TO_PIP: Dict[str, str] = {
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "yaml": "PyYAML",
        "dateutil": "python-dateutil",
        "dotenv": "python-dotenv",
    }

    def __init__(self):
        self.all_imports: Set[str] = set()
        self.file_imports: Dict[str, Set[str]] = defaultdict(set)
        self.local_modules: Set[str] = set()

    # ---------- Core classification helpers ----------

    def _is_stdlib(self, module_name: str) -> bool:
        """Check if a module is part of Python standard library"""
        base = module_name.split(".")[0]
        return base in sys.stdlib_module_names or base.startswith("_")

    def _is_local_module(self, module_name: str) -> bool:
        """Check if a module belongs to the current repo"""
        base = module_name.split(".")[0]
        return base in self.local_modules

    # ---------- Import extraction ----------

    def _extract_imports_from_code(self, code: str, file_path: str = "") -> Set[str]:
        """Extract import statements from Python code using AST"""
        imports = set()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._extract_imports_regex(code)
        except Exception:
            return self._extract_imports_regex(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])

        return imports

    def _extract_imports_regex(self, code: str) -> Set[str]:
        """Fallback regex-based import extraction (conservative)"""
        imports = set()

        import_pattern = re.compile(r'^\s*import\s+([a-zA-Z0-9_.,\s]+)')
        from_pattern = re.compile(r'^\s*from\s+([a-zA-Z0-9_\.]+)\s+import')

        for line in code.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            m = import_pattern.match(line)
            if m:
                for part in m.group(1).split(","):
                    base = part.strip().split()[0]
                    if base and not base.startswith("."):
                        imports.add(base.split(".")[0])
                continue

            m = from_pattern.match(line)
            if m:
                base = m.group(1).split(".")[0]
                if base and not base.startswith("."):
                    imports.add(base)

        return imports

    # ---------- Scanning ----------

    def _detect_local_modules(self, file_paths: List[str]):
        """Detect local modules from repo file structure"""
        local: Set[str] = set()
        for p in file_paths:
            if not p.endswith(".py"):
                continue
            base = os.path.splitext(os.path.basename(p))[0]
            local.add(base)

            # Treat packages as local modules too (dir/__init__.py => dir)
            if p.endswith("/__init__.py"):
                pkg = os.path.basename(os.path.dirname(p))
                if pkg:
                    local.add(pkg)

        self.local_modules = local

    def scan_file(self, file_path: str, content: str):
        """Scan a single file for imports"""
        raw_imports = self._extract_imports_from_code(content, file_path)

        filtered = set()
        for imp in raw_imports:
            if self._is_stdlib(imp):
                continue
            if self._is_local_module(imp):
                continue
            filtered.add(imp)

        self.file_imports[file_path] = filtered
        self.all_imports.update(filtered)

    def scan_codebase(self, repo, file_paths: List[str]) -> Set[str]:
        """Scan entire codebase for dependencies"""
        print(f"🔍 Scanning {len(file_paths)} files for dependencies...")

        self._detect_local_modules(file_paths)

        for file_path in file_paths:
            try:
                content = repo.get_contents(file_path).decoded_content.decode()
                self.scan_file(file_path, content)
            except Exception as e:
                print(f"⚠️ Error scanning {file_path}: {e}")

        print(f"📦 Found {len(self.all_imports)} third-party dependencies")
        if self.all_imports:
            print(f"   Dependencies: {', '.join(sorted(self.all_imports))}")

        return self.all_imports

    # ---------- Output helpers ----------

    def generate_requirements_txt(self, dependencies: Set[str] = None) -> str:
        """Generate a requirements.txt content from dependencies"""
        deps = dependencies or self.all_imports

        resolved = []
        for dep in sorted(deps):
            pkg = self.IMPORT_TO_PIP.get(dep, dep)
            if pkg not in resolved:
                resolved.append(pkg)

        return "\n".join(resolved) + "\n"

    def get_dependencies_for_file(self, file_path: str) -> Set[str]:
        """Get dependencies for a specific file"""
        return self.file_imports.get(file_path, set())

    def get_all_dependencies(self) -> Set[str]:
        """Get all detected dependencies"""
        return self.all_imports.copy()
