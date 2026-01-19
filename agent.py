import os
import operator
import uuid
import difflib
from typing import Annotated, TypedDict
from github import Github, Auth
from github.GithubException import UnknownObjectException
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import re
from executor import PythonExecutor 
from codebase_indexer import CodebaseIndexer
from dependency_detector import DependencyDetector
from dotenv import load_dotenv

load_dotenv()

def update_last(old, new):
    return new

# --- 1. Define State ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    file_path: Annotated[str, update_last]
    original_code: Annotated[str, update_last]
    docs: Annotated[str, update_last]
    extra_context: Annotated[str, update_last] # <--- NEW: Stores extra file contents
    code: Annotated[str, update_last]
    error: Annotated[str, update_last]
    iterations: Annotated[int, update_last]
    # Multi-file support
    all_files: Annotated[list[str], update_last]  # List of all Python files in repo
    failing_files: Annotated[list[str], update_last]  # List of files that failed tests
    fixed_files: Annotated[dict[str, str], update_last]  # Dict mapping file_path -> fixed_code
    original_codes: Annotated[dict[str, str], update_last]  # Dict mapping file_path -> original_code
    file_errors: Annotated[dict[str, str], update_last]  # Dict mapping file_path -> error_message
    current_file_index: Annotated[int, update_last]  # Index of current file being processed
    # Codebase understanding
    codebase_index: Annotated[dict, update_last]  # Codebase indexer instance (stored as dict for state)
    related_context: Annotated[str, update_last]  # Related file context for current file
    research_iterations: Annotated[int, update_last]  # Counter to prevent infinite research loops

class CodeSentinel:
    def __init__(self, repo_name):
        # Try both OPEN_API_KEY and OPENAI_API_KEY for compatibility
        api_key = os.getenv("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key not found! Please set OPEN_API_KEY or OPENAI_API_KEY environment variable.")
        # Ensure API key is a string, not a callable
        if callable(api_key):
            api_key = api_key()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=str(api_key))
        self.executor = PythonExecutor()  # Will be updated with dependencies later
        self.memory = MemorySaver()
        # Initialize codebase indexer (will be populated during indexing)
        self.indexer = None
        # Initialize dependency detector
        self.dependency_detector = DependencyDetector()
        
        # Fixed the DeprecationWarning you saw
        auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
        self.gh = Github(auth=auth)
        self.repo = self.gh.get_repo(repo_name)

        # --- Handle Permissions & Forking ---
        try:
            self.repo = self.gh.get_repo(repo_name)
            # Test write access by checking permissions
            perms = self.repo.permissions
            if not perms.push:
                raise Exception("403: No push access")
            print(f"✅ Connected to: {repo_name}")
        except Exception as e:
            if "403" in str(e) or "Resource not accessible" in str(e) or "No push access" in str(e):
                print(f"🔒 Access denied to {repo_name}. Using fork with upstream sync...")
                original_repo = self.gh.get_repo(repo_name)
                user = self.gh.get_user()

                # Try to reuse an existing fork if it already exists for this user
                fork_full_name = f"{user.login}/{original_repo.name}"
                fork_repo = None
                try:
                    fork_repo = self.gh.get_repo(fork_full_name)
                    print(f"🍴 Existing fork found at: {fork_repo.full_name}")
                except UnknownObjectException:
                    # No existing fork; create a new one
                    print("🍴 No existing fork found. Creating a new fork...")
                    fork_repo = user.create_fork(original_repo)
                    print(f"🍴 Fork created at: {fork_repo.full_name}")

                # Sync the fork's default branch with the latest upstream commit
                try:
                    upstream_default = original_repo.default_branch
                    fork_default = fork_repo.default_branch
                    upstream_branch = original_repo.get_branch(upstream_default)
                    fork_ref = fork_repo.get_git_ref(f"heads/{fork_default}")
                    fork_ref.edit(sha=upstream_branch.commit.sha, force=True)
                    print(f"🔄 Synced fork '{fork_repo.full_name}' ({fork_default}) with upstream '{original_repo.full_name}' ({upstream_default})")
                except Exception as sync_err:
                    # If sync fails, still proceed with the fork but log it clearly
                    print(f"⚠️ Failed to fully sync fork with upstream: {sync_err}")

                self.repo = fork_repo
            else:
                raise e
        
        # Build Graph
        workflow = StateGraph(AgentState)
        
        # Define Nodes
        workflow.add_node("fetch_docs", self.fetch_docs)
        workflow.add_node("scan_files", self.scan_files)  # NEW: Scan all Python files
        workflow.add_node("detect_dependencies", self.detect_dependencies)  # NEW: Detect dependencies
        workflow.add_node("index_codebase", self.index_codebase)  # NEW: Index codebase with embeddings
        workflow.add_node("test_all_files", self.test_all_files)  # NEW: Test all files
        workflow.add_node("fetch_file", self.fetch_file)
        workflow.add_node("analyze_impact", self.analyze_impact)  # NEW: Analyze impact before fixing
        workflow.add_node("generate_fix", self.generate_fix)
        workflow.add_node("fetch_extra", self.fetch_extra_context)
        workflow.add_node("execute_test", self.execute_test)
        workflow.add_node("create_pr", self.create_pr)

        # Connect Nodes - Multi-file flow
        workflow.add_edge(START, "fetch_docs")
        workflow.add_edge("fetch_docs", "scan_files")
        workflow.add_edge("scan_files", "detect_dependencies")  # Detect dependencies first
        workflow.add_edge("detect_dependencies", "index_codebase")  # Index after detecting deps
        workflow.add_edge("index_codebase", "test_all_files")  # Test after indexing
        
        # After testing, check if there are failing files
        workflow.add_conditional_edges(
            "test_all_files",
            self.decide_has_failures,
            {"fix_files": "fetch_file", "create_pr": "create_pr"}
        )
        
        # Single file fix flow (for each failing file)
        workflow.add_edge("fetch_file", "analyze_impact")  # Analyze impact first
        workflow.add_edge("analyze_impact", "generate_fix")  # Then generate fix with context
        workflow.add_conditional_edges(
            "generate_fix",
            self.decide_research_or_test,
            {"research": "fetch_extra", "test": "execute_test"}
        )
        workflow.add_conditional_edges(
            "fetch_extra",
            self.decide_after_research,
            {"continue_research": "generate_fix", "test": "execute_test"}
        )
        
        workflow.add_conditional_edges(
            "execute_test",
            self.decide_next_file,
            {"retry": "generate_fix", "next_file": "process_next_file", "create_pr": "create_pr"}
        )
        
        # Node to process next file
        workflow.add_node("process_next_file", self.process_next_file)
        workflow.add_edge("process_next_file", "fetch_file")
        
        workflow.add_edge("create_pr", END)
        self.app = workflow.compile(checkpointer=self.memory)

    # --- NEW NODE: The Researcher ---
    def fetch_docs(self, state: AgentState):
        print("📖 Reading project structure...")
    # List all files in the repo to give the bot context
        contents = self.repo.get_contents("")
        repo_files = [content.path for content in contents]
        file_list_str = "\n".join(repo_files)
    
        try:
            readme = self.repo.get_contents("README.md").decoded_content.decode()
        except:
            readme = "No README found."
        
        context = f"File Structure:\n{file_list_str}\n\nDocumentation:\n{readme}"
        return {"docs": context}
    
    # --- NEW NODE: Scan all Python files in repo ---
    def scan_files(self, state: AgentState):
        print("🔍 Scanning repository for Python files...")
        python_files = []
        
        def get_python_files(path=""):
            """Recursively get all Python files from the repo"""
            try:
                contents = self.repo.get_contents(path)
                # Handle both list and single ContentFile responses
                if not isinstance(contents, list):
                    contents = [contents]
                
                for content in contents:
                    if content.type == "file" and content.path.endswith(".py"):
                        # Skip common non-testable files
                        if not any(skip in content.path for skip in ["__pycache__", ".pyc", "venv/", "env/", ".git/"]):
                            python_files.append(content.path)
                    elif content.type == "dir":
                        # Skip common directories
                        if not any(skip in content.path for skip in ["__pycache__", "venv/", "env/", ".git/", "node_modules/"]):
                            get_python_files(content.path)
            except Exception as e:
                print(f"⚠️ Error scanning {path}: {e}")
        
        get_python_files()
        print(f"📋 Found {len(python_files)} Python file(s)")
        
        return {
            "all_files": python_files,
            "failing_files": [],
            "fixed_files": {},
            "original_codes": {},
            "file_errors": {},
            "current_file_index": 0,
            "codebase_index": {},
            "related_context": ""
        }
    
    # --- NEW NODE: Index codebase with embeddings ---
    def index_codebase(self, state: AgentState):
        print("📚 Indexing codebase for semantic understanding...")
        all_files = state.get("all_files", [])
        
        if not all_files:
            print("⚠️ No files to index")
            return {"codebase_index": {}}
        
        try:
            # Initialize indexer if not already done
            if self.indexer is None:
                self.indexer = CodebaseIndexer(self.repo)
            
            # Index all files
            index_info = self.indexer.index_codebase(all_files)
            print(f"✅ Codebase indexed: {index_info['indexed_files']} files")
            print(f"   Dependencies mapped: {len(index_info.get('dependencies', {}))} files have dependencies")
            
            return {"codebase_index": {"indexed": True, "file_count": index_info['indexed_files']}}
        except Exception as e:
            print(f"⚠️ Error indexing codebase: {e}")
            # Continue without indexing if it fails
            return {"codebase_index": {"indexed": False, "error": str(e)}}
    
    # --- NEW NODE: Detect dependencies from codebase ---
    def detect_dependencies(self, state: AgentState):
        """Detect all dependencies from the codebase"""
        print("📦 Detecting dependencies from codebase...")
        all_files = state.get("all_files", [])
        
        if not all_files:
            return {}
        
        # Check if repo has requirements.txt
        repo_requirements = set()
        try:
            reqs_file = self.repo.get_contents("requirements.txt")
            reqs_content = reqs_file.decoded_content.decode()
            # Parse requirements.txt
            for line in reqs_content.splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before ==, >=, etc.)
                    pkg_name = re.split(r'[>=<!=]', line)[0].strip()
                    if pkg_name:
                        repo_requirements.add(pkg_name)
            print(f"📋 Found requirements.txt with {len(repo_requirements)} packages")
        except Exception:
            print("📋 No requirements.txt found in repo")
        
        # Scan codebase for imports
        detected_imports = self.dependency_detector.scan_codebase(self.repo, all_files)

        # Convert import names -> pip package names (cv2 -> opencv-python, etc.)
        detected_pkgs_txt = self.dependency_detector.generate_requirements_txt(detected_imports)
        detected_packages = {
            line.strip()
            for line in detected_pkgs_txt.splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        
        # Combine repo requirements with detected dependencies
        # Map repo requirements through the same import→pip mapping to normalize (e.g., cv2->opencv-python)
        repo_req_txt = self.dependency_detector.generate_requirements_txt(repo_requirements)
        repo_pkgs = {
            line.strip()
            for line in repo_req_txt.splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

        all_dependencies = repo_pkgs.union(detected_packages)
        
        # Update executor with dependencies
        self.executor.set_dependencies(all_dependencies)
        # Also set local modules so runtime pip never tries to install them
        self.executor.set_local_modules(self.dependency_detector.local_modules)
        
        print(f"✅ Total dependencies: {len(all_dependencies)}")
        if all_dependencies:
            deps_list = sorted(all_dependencies)
            print(f"   Packages: {', '.join(deps_list[:10])}{'...' if len(deps_list) > 10 else ''}")
        
        return {
            "dependencies": list(all_dependencies),
            "local_modules": list(self.dependency_detector.local_modules),
        }
    
    # --- NEW NODE: Test all files and identify failures ---
    def test_all_files(self, state: AgentState):
        print(f"🧪 Testing {len(state.get('all_files', []))} file(s)...")
        all_files = state.get("all_files", [])
        failing_files = []
        original_codes = {}
        file_errors = {}
        
        if not all_files:
            print("⚠️ No Python files found to test")
            return {
                "failing_files": [],
                "original_codes": {},
                "file_errors": {},
                "messages": [HumanMessage(content="No Python files found in the repository.")]
            }
        
        for file_path in all_files:
            try:
                print(f"  Testing: {file_path}")
                content = self.repo.get_contents(file_path).decoded_content.decode()
                
                # Store original code for all files (needed for PR description)
                original_codes[file_path] = content
                
                # Skip files that are likely not executable (e.g., empty files, only comments)
                if not content.strip() or len(content.strip()) < 10:
                    print(f"  ⏭️  {file_path} skipped (too small or empty)")
                    continue
                
                # Test the file
                result = self.executor.execute(content)
                
                if result.get("error") or not result.get("success", False):
                    error_msg = result.get("error", "Unknown error")
                    print(f"  ❌ {file_path} failed: {error_msg[:100]}")
                    failing_files.append(file_path)
                    file_errors[file_path] = error_msg
                else:
                    print(f"  ✅ {file_path} passed")
            except Exception as e:
                print(f"  ⚠️ Error testing {file_path}: {e}")
                # Only add to failing if it's a real error, not just import issues
                error_str = str(e)
                if "No such file" not in error_str and "ModuleNotFoundError" not in error_str:
                    failing_files.append(file_path)
                    file_errors[file_path] = str(e)
                    # Ensure we have the original code stored
                    if file_path not in original_codes:
                        try:
                            original_codes[file_path] = self.repo.get_contents(file_path).decoded_content.decode()
                        except:
                            original_codes[file_path] = ""
        
        print(f"📊 Test Results: {len(failing_files)} file(s) need fixing out of {len(all_files)} tested")
        
        return {
            "failing_files": failing_files,
            "original_codes": original_codes,
            "file_errors": file_errors,
            "messages": [HumanMessage(content=f"Tested {len(all_files)} file(s). Found {len(failing_files)} file(s) that need fixing: {', '.join(failing_files) if failing_files else 'None'}")]
        }

    # --- Node: Process next file ---
    def process_next_file(self, state: AgentState):
        failing_files = state.get("failing_files", [])
        current_index = state.get("current_file_index", 0)
        fixed_files = state.get("fixed_files", {})
        current_file = state.get("file_path", "")
        
        # Save the fixed code for current file ONLY if it actually changed
        code = state.get("code", "")
        if code and current_file:
            original_codes = state.get("original_codes", {})
            original = original_codes.get(current_file, state.get("original_code", ""))
            if original.strip() != code.strip():
                fixed_files[current_file] = code
                print(f"💾 Saved fix for {current_file}")
            else:
                print(f"⏭️  No changes for {current_file}; skipping save")
        
        # Move to next file
        if current_index + 1 < len(failing_files):
            next_file = failing_files[current_index + 1]
            print(f"➡️  Moving to next file: {next_file}")
            return {
                "fixed_files": fixed_files,
                "file_path": next_file,
                "current_file_index": current_index + 1,
                "iterations": 0,
                "error": "",
                "code": "",  # Reset code for next file
                "research_iterations": 0  # Reset research counter for next file
            }
        else:
            # All files processed
            print(f"✅ Finished processing all {len(failing_files)} file(s)")
            return {"fixed_files": fixed_files}
    
    # --- Node: Fetch Code ---
    def fetch_file(self, state: AgentState):
        path = state.get("file_path", "")
        failing_files = state.get("failing_files", [])
        current_index = state.get("current_file_index", 0)
        
        # If path not set, get from failing_files list
        if not path and failing_files:
            path = failing_files[current_index]
        
        if not path:
            # Fallback: use file_path from state or return empty
            return {"original_code": "", "file_path": ""}
        
        print(f"📂 Fetching {path} from GitHub...")
        content = self.repo.get_contents(path).decoded_content.decode()
        
        # Get related context from codebase indexer
        related_context = ""
        if self.indexer:
            try:
                related_context = self.indexer.get_context_for_file(path, max_files=5)
                if related_context:
                    print(f"🔗 Found {len(related_context.split('---')) - 1} related file(s) for context")
            except Exception as e:
                print(f"⚠️ Error getting context: {e}")
        
        return {
            "original_code": content, 
            "file_path": path,
            "current_file_index": current_index,
            "iterations": 0,
            "error": "",
            "related_context": related_context,
            "research_iterations": 0  # Reset research counter for new file
        }
    
    # --- NEW NODE: Analyze impact before fixing ---
    def analyze_impact(self, state: AgentState):
        file_path = state.get("file_path", "")
        if not file_path or not self.indexer:
            return {}
        
        print(f"🔍 Analyzing impact of changes to {file_path}...")
        try:
            impact = self.indexer.analyze_impact(file_path, state.get("code", ""))
            
            if impact.get("files_that_import_this"):
                print(f"⚠️ {len(impact['files_that_import_this'])} file(s) depend on this file")
                for dep_file in impact["files_that_import_this"][:3]:
                    print(f"   - {dep_file}")
            else:
                print("✅ No other files depend on this file")
            
            # Store impact info in state for use in generate_fix
            impact_msg = impact.get("recommendation", "")
            return {"extra_context": state.get("extra_context", "") + f"\n\n{impact_msg}"}
        except Exception as e:
            print(f"⚠️ Error analyzing impact: {e}")
            return {}
    
    # --- Node: Fetch Extra Context (The Multi-File Key) ---
    def fetch_extra_context(self, state: AgentState):
        last_msg = state["messages"][-1].content
        match = re.search(r"FETCH_FILE:\s*(\S+)", last_msg)
        
        # Increment research iterations counter
        research_iterations = state.get("research_iterations", 0) + 1
        
        if match:
            extra_path = match.group(1)
            print(f"🧐 Agent needs to see: {extra_path} (research iteration {research_iterations})")
            try:
                content = self.repo.get_contents(extra_path).decoded_content.decode()
                new_context = state.get("extra_context", "") + f"\n\n--- Content of {extra_path} ---\n{content}"
                return {
                    "extra_context": new_context, 
                    "messages": [HumanMessage(content=f"Fetched {extra_path}")],
                    "research_iterations": research_iterations
                }
            except:
                return {
                    "messages": [HumanMessage(content=f"Error: {extra_path} not found.")],
                    "research_iterations": research_iterations
                }
        return {"research_iterations": research_iterations}
        
    # --- Node: Generate (Modified to use Docs + Codebase Context) ---
    def generate_fix(self, state: AgentState):
        signature = "\n\n# CodeSentinal: created for you by RuchirAdnaik."

        # Build comprehensive context
        related_context = state.get("related_context", "")
        extra_context = state.get("extra_context", "")
        
        # Combine all context
        context_section = ""
        if related_context:
            context_section += f"\n\nRELATED FILES (for understanding dependencies and patterns):\n{related_context}"
        if extra_context:
            context_section += f"\n\nADDITIONAL CONTEXT:\n{extra_context}"

        # We now pass the documentation + codebase context into the prompt
        prompt = f"""
        PROJECT DOCUMENTATION:
        {state['docs']}
        
        {context_section}

        CODE TO FIX:
        {state['original_code']}

        INSTRUCTIONS:
        1. Analyze the code in the context of the entire codebase. Consider how this file relates to other files shown above.
        2. Identify the runtime error in the code.
        3. Fix the error while ensuring your changes don't break other files that depend on this one.
        4. If you see related files above, ensure your fix maintains compatibility with their expectations (function signatures, imports, etc.).
        5. Ensure your fix follows any coding standards or rules mentioned in the PROJECT DOCUMENTATION above.
        6. Return ONLY the full corrected python code in a markdown block.
        7. MANDATORY: You must add the following comment as the very last line of the code:
           # CodeSentinal: created for you by RuchirAdnaik.
        """
        
        if state.get("error"):
            prompt += f"\n\nYour last attempt failed with this error: {state['error']}. Please try again."
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Improved Extraction
        content = response.content
        if "```python" in content:
            code = content.split("```python")[1].split("```")[0].strip()
        else:
            code = content.strip()

        # Only append signature if this is a real change vs original
        original_code = state.get("original_code", "")
        if original_code.strip() != code.strip():
            if "CodeSentinal: created for you by RuchirAdnaik." not in code:
                code += signature
            
        return {"messages": [response], "code": code, "iterations": state.get("iterations", 0) + 1}

    # --- Node: Execute ---
    def execute_test(self, state: AgentState):
        print(f"⚙️ Testing fix in Docker (Attempt {state['iterations']})...")
        result = self.executor.execute(state["code"])
    
        # If there's no error, we MUST return an empty string, not None
        error_value = result["error"] if result["error"] else ""
        
        # Update file_errors dict with the current error (if any)
        file_errors = state.get("file_errors", {}).copy()
        current_file = state.get("file_path", "")
        if error_value and current_file:
            file_errors[current_file] = error_value
        
        return {"error": error_value, "file_errors": file_errors}
    
    # --- Router: Research vs Test ---
    def decide_research_or_test(self, state: AgentState):
        # Prevent infinite research loops - limit to 3 research iterations per fix attempt
        research_iterations = state.get("research_iterations", 0)
        if research_iterations >= 3:
            print("⚠️ Research limit reached, proceeding to test")
            return "test"
        
        if "FETCH_FILE:" in state["messages"][-1].content:
            return "research"
        return "test"
    
    # --- Router: After Research ---
    def decide_after_research(self, state: AgentState):
        """Decide whether to continue research or proceed to testing"""
        research_iterations = state.get("research_iterations", 0)
        
        # If still requesting files and under limit, continue research
        if "FETCH_FILE:" in state["messages"][-1].content and research_iterations < 3:
            return "continue_research"
        
        # Otherwise, proceed to test (research limit reached or no more file requests)
        return "test"
    
    # --- Router: Check if there are failing files ---
    def decide_has_failures(self, state: AgentState):
        failing_files = state.get("failing_files", [])
        if failing_files:
            return "fix_files"
        return "create_pr"
    
    # --- Router: Retry vs Next File vs Submit ---
    def decide_next_file(self, state: AgentState):
        failing_files = state.get("failing_files", [])
        current_index = state.get("current_file_index", 0)
        current_file = state.get("file_path", "")
        
        # If current file has no error, it's fixed
        if not state.get("error"):
            # Move to next file
            if current_index + 1 < len(failing_files):
                return "next_file"
            else:
                # All files fixed, create PR
                return "create_pr"
        
        # If there's an error, retry or give up
        iterations = state.get("iterations", 0)
        if iterations < 3:
            return "retry"
        else:
            # Max retries reached, move to next file
            if current_index + 1 < len(failing_files):
                return "next_file"
            else:
                return "create_pr"
    
    def _create_unified_diff(self, original: str, fixed: str, file_path: str) -> str:
        """Create a unified diff between original and fixed code"""
        original_lines = original.splitlines(keepends=True)
        fixed_lines = fixed.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            fixed_lines,
            fromfile=f"original/{file_path}",
            tofile=f"fixed/{file_path}",
            lineterm=''
        )
        return ''.join(diff)
    
    def generate_pr_summary(self, state: AgentState):
        fixed_files = state.get("fixed_files", {})
        failing_files = state.get("failing_files", [])
        original_codes = state.get("original_codes", {})
        file_errors = state.get("file_errors", {})
        
        # Filter out files where no actual changes were made
        files_with_changes = {}
        for file_path, fixed_code in fixed_files.items():
            original_code = original_codes.get(file_path, "")
            # Only include files where code actually changed
            if original_code.strip() != fixed_code.strip():
                files_with_changes[file_path] = fixed_code
        
        if len(files_with_changes) == 0:
            # No actual changes made
            return "## Summary\nNo code changes were required. All files are already correct or errors were external (e.g., network issues)."
        
        if len(files_with_changes) > 1:
            # Multi-file PR description
            file_details = []
            for file_path in files_with_changes.keys():
                original_code = original_codes.get(file_path, "")
                fixed_code = fixed_files.get(file_path, "")
                error_msg = file_errors.get(file_path, "Runtime error")
                
                # Create actual diff
                diff = self._create_unified_diff(original_code, fixed_code, file_path)
                
                file_details.append(f"""
=== FILE: {file_path} ===

ORIGINAL ERROR:
{error_msg[:1000]}

CODE DIFF (showing actual changes):
```diff
{diff[:3000] if len(diff) > 3000 else diff}
```

ORIGINAL CODE (full):
```python
{original_code[:2000] if len(original_code) > 2000 else original_code}
```

FIXED CODE (full):
```python
{fixed_code[:2000] if len(fixed_code) > 2000 else fixed_code}
```

""")
            
            prompt = f"""
You are a code review expert. Analyze the ACTUAL CODE CHANGES made to multiple files and write a comprehensive, professional PR description.

IMPORTANT: Only describe files where ACTUAL CODE CHANGES were made. If a file shows no diff (no changes), do NOT include it in the description.

FILES WITH ACTUAL CHANGES ({len(files_with_changes)} files):
{''.join(file_details)}

TOTAL FILES TESTED: {len(state.get('all_files', []))}
FILES WITH CHANGES: {len(files_with_changes)}
FILES WITHOUT CHANGES: {len(fixed_files) - len(files_with_changes)}

Write a detailed PR description with the following structure:

## Summary
Brief overview of what was actually fixed (2-3 sentences). Only mention files that had code changes.

## Files Fixed
For EACH file that has actual code changes (check the diff!), provide:
1. **File Name**: [filename]
2. **Error Description**: What was the error? Explain in detail what went wrong.
3. **Root Cause**: Why was this error occurring? Explain the underlying issue.
4. **Solution**: What SPECIFIC code changes were made? Reference the diff - what lines were added, removed, or modified? Be very specific.
5. **Impact**: How does this fix improve the code?

IMPORTANT: 
- Only describe files that have actual code changes (non-empty diff)
- Be specific about what changed - reference line numbers, function names, etc.
- If a file has no changes, do NOT include it in the description

## Verification
- All fixed files have been tested and verified to execute without errors
- Code follows project standards and best practices

Make the description detailed, professional, and accurate. Focus on the ACTUAL changes made, not assumptions.
"""
        else:
            # Single file PR description
            file_path = list(fixed_files.keys())[0] if fixed_files else state.get("file_path", "")
            original_code = original_codes.get(file_path, state.get("original_code", ""))
            fixed_code = fixed_files.get(file_path, state.get("code", ""))
            error_msg = file_errors.get(file_path, state.get("error", "Runtime error"))
            
            # Check if there are actual changes
            if original_code.strip() == fixed_code.strip():
                return f"## Summary\nNo code changes were required for {file_path}. The error was external (e.g., network issues) or the file is already correct."
            
            # Create diff
            diff = self._create_unified_diff(original_code, fixed_code, file_path)
            
            prompt = f"""
You are a code review expert. Analyze the ACTUAL CODE CHANGES made to a file and write a comprehensive, professional PR description.

FILE: {file_path}

ORIGINAL ERROR:
{error_msg}

CODE DIFF (showing actual changes):
```diff
{diff}
```

ORIGINAL CODE:
```python
{original_code}
```

FIXED CODE:
```python
{fixed_code}
```

Write a detailed PR description with the following structure:

## Summary
Brief overview of what was actually fixed (2-3 sentences). Be specific about what changed.

## Error Analysis
1. **Error Description**: What was the error? Explain in detail what went wrong, including the exact error message and when it occurs.
2. **Root Cause**: Why was this error occurring? Explain the underlying issue, what caused it, and why the original code was problematic.

## Solution
1. **Changes Made**: What SPECIFIC code changes were made? Reference the diff above - what lines were added, removed, or modified? Be very specific with line numbers and code snippets.
2. **How It Works**: Explain how the fix resolves the issue. Walk through the corrected code logic.
3. **Code Quality**: Does the fix follow best practices? Are there any improvements beyond just fixing the error?

## Verification
- The fixed code has been tested and verified to execute without errors
- Code follows project standards and best practices

Make the description detailed, professional, and accurate. Focus on the ACTUAL changes shown in the diff, not assumptions.
"""
        summary = self.llm.invoke([HumanMessage(content=prompt)])
        return summary.content

    # --- Node: Create PR (Updated for multi-file) ---
    def create_pr(self, state: AgentState):
        branch_name = f"fix-{uuid.uuid4().hex[:6]}"
        fixed_files = state.get("fixed_files", {}).copy()  # Make a copy to modify
        original_codes = state.get("original_codes", {})
        
        # Save current file's code if it exists and hasn't been saved yet
        current_file = state.get("file_path", "")
        current_code = state.get("code", "")
        if current_file and current_code and current_file not in fixed_files:
            original = original_codes.get(current_file, state.get("original_code", ""))
            if original.strip() != current_code.strip():
                fixed_files[current_file] = current_code
                print(f"💾 Saved fix for {current_file} before creating PR")
            else:
                print(f"⏭️  No changes for {current_file}; skipping save before PR")
        
        # If no fixed files but we have code, use single file mode
        if not fixed_files and state.get("code"):
            fixed_files[state.get("file_path", "")] = state.get("code", "")
        
        # If still no files, create empty PR message
        if not fixed_files:
            msg = "No files needed fixing. All tests passed!"
            return {"messages": [HumanMessage(content=msg)], "fixed_files": {}}

        # Filter to only files with actual changes (avoid committing tagline-only/no-op updates)
        fixed_files = {
            path: code
            for path, code in fixed_files.items()
            if original_codes.get(path, "").strip() != code.strip()
        }
        if not fixed_files:
            msg = "No code changes were required. All failures were external (e.g., dependency/network)."
            return {"messages": [HumanMessage(content=msg)], "fixed_files": {}}
        
        # Update state with fixed_files for summary generation
        pr_body = self.generate_pr_summary({**state, "fixed_files": fixed_files})
        
        # Check if this is a fork or our own repo
        is_fork = self.repo.fork 
        
        # 1. Create the fix branch
        main_ref = self.repo.get_branch(self.repo.default_branch)
        self.repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=main_ref.commit.sha)
        
        # 2. Update all fixed files on that branch
        files_fixed = []
        for path, code in fixed_files.items():
            try:
                file_git = self.repo.get_contents(path, ref=branch_name)
                self.repo.update_file(
                    path, 
                    f"🤖 CodeSentinal Auto-fix: {path}", 
                    code, 
                    file_git.sha, 
                    branch=branch_name
                )
                files_fixed.append(path)
                print(f"✅ Committed fix for {path}")
            except Exception as e:
                # File might not exist in branch yet, try creating it
                try:
                    self.repo.create_file(
                        path,
                        f"🤖 CodeSentinal Auto-fix: {path}",
                        code,
                        branch=branch_name
                    )
                    files_fixed.append(path)
                    print(f"✅ Created fix for {path}")
                except Exception as e2:
                    print(f"⚠️ Error updating {path}: {e2}")
        
        if is_fork:
            print("🍴 This is a fork. Auto-merging fix into the fork's main branch...")
            # AUTO-MERGE into the fork's default branch
            self.repo.merge(self.repo.default_branch, branch_name, "Auto-merging verified fixes")
            
            # Optional: Still create a PR to the ORIGINAL owner so they see your work
            parent_repo = self.repo.parent # This is the original repo you forked from
            try:
                pr = parent_repo.create_pull(
                    title=f"Fix for {len(files_fixed)} file(s)",
                    body=f"Sent with ❤️ from CodeSentinal.\n\n{pr_body}",
                    head=f"{self.gh.get_user().login}:{branch_name}",
                    base=parent_repo.default_branch
                )
                msg = f"Fixed {len(files_fixed)} file(s). Merged into fork and PR sent to original owner: {pr.html_url}"
            except:
                msg = f"Fixed {len(files_fixed)} file(s). Merged into fork. (Original owner doesn't allow PRs)"
            
            return {"messages": [HumanMessage(content=msg)], "fixed_files": fixed_files}
        else:
            print("👤 You own this repo. Creating a PR for your review...")
            pr = self.repo.create_pull(
                title=f"Proposed Fix: {len(files_fixed)} file(s)",
                body=pr_body,
                head=branch_name,
                base=self.repo.default_branch
            )
            msg = f"PR Created for {len(files_fixed)} file(s): {pr.html_url}"
        
        # Return updated fixed_files so UI can display correct count
        return {"messages": [HumanMessage(content=msg)], "fixed_files": fixed_files}

    def decide_next(self, state: AgentState):
        if not state.get("error"): return "submit"
        return "retry" if state["iterations"] < 3 else "submit"

    def run(self, file_to_fix=None):
        # 1. Create a unique thread ID for this specific run
        # 2. Increase recursion limit to handle multi-file processing
        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": 100  # Increased from default 25 to handle multiple files
        }
        
        # 2. Initialize EVERY key to a default empty value
        # This prevents the 'NoneType' error
        initial_state = {
            "messages": [],
            "file_path": file_to_fix or "",
            "original_code": "",
            "docs": "",
            "extra_context": "",
            "code": "",
            "error": "",
            "iterations": 0,
            "all_files": [],
            "failing_files": [],
            "fixed_files": {},
            "original_codes": {},
            "file_errors": {},
            "current_file_index": 0,
            "codebase_index": {},
            "related_context": "",
            "research_iterations": 0
        }
        
        # 3. Use the initialized state
        return self.app.invoke(initial_state, config)
    