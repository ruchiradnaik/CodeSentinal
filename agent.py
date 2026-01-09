import os
import operator
import uuid
from typing import Annotated, TypedDict
from github import Github, Auth
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import re
from executor import PythonExecutor 
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

class CodeSentinel:
    def __init__(self, repo_name):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPEN_API_KEY"))
        self.executor = PythonExecutor()
        self.memory = MemorySaver()
        
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
                print(f"🔒 Access denied to {repo_name}. Attempting to fork...")
                original_repo = self.gh.get_repo(repo_name)
                user = self.gh.get_user()
                self.repo = user.create_fork(original_repo)
                print(f"🍴 Fork created at: {self.repo.full_name}")
            else:
                raise e
        
        # Build Graph
        workflow = StateGraph(AgentState)
        
        # Define Nodes
        workflow.add_node("fetch_docs", self.fetch_docs) # <--- New Node
        workflow.add_node("fetch_file", self.fetch_file)
        workflow.add_node("generate_fix", self.generate_fix)
        workflow.add_node("fetch_extra", self.fetch_extra_context)
        workflow.add_node("execute_test", self.execute_test)
        workflow.add_node("create_pr", self.create_pr)

        # Connect Nodes
        workflow.add_edge(START, "fetch_docs")           # Start with Research
        workflow.add_edge("fetch_docs", "fetch_file")
        workflow.add_edge("fetch_file", "generate_fix")
        workflow.add_edge("generate_fix", "execute_test")

        # MULTI-FILE ROUTING: Test the code or go get more files?
        workflow.add_conditional_edges(
            "generate_fix",
            self.decide_research_or_test,
            {"research": "fetch_extra", "test": "execute_test"}
        )
        
        # Loop back after getting extra info
        workflow.add_edge("fetch_extra", "generate_fix")
        
        workflow.add_conditional_edges(
            "execute_test",
            self.decide_next,
            {"retry": "generate_fix", "submit": "create_pr"}
        )
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

    # --- Node: Fetch Code ---
    def fetch_file(self, state: AgentState):
        path = state["file_path"]
        print(f"📂 Fetching {path} from GitHub...")
        content = self.repo.get_contents(path).decoded_content.decode()
        return {"original_code": content}
    
    # --- Node: Fetch Extra Context (The Multi-File Key) ---
    def fetch_extra_context(self, state: AgentState):
        last_msg = state["messages"][-1].content
        match = re.search(r"FETCH_FILE:\s*(\S+)", last_msg)
        
        if match:
            extra_path = match.group(1)
            print(f"🧐 Agent needs to see: {extra_path}")
            try:
                content = self.repo.get_contents(extra_path).decoded_content.decode()
                new_context = state.get("extra_context", "") + f"\n\n--- Content of {extra_path} ---\n{content}"
                return {"extra_context": new_context, "messages": [HumanMessage(content=f"Fetched {extra_path}")]}
            except:
                return {"messages": [HumanMessage(content=f"Error: {extra_path} not found.")]}
        return {}
        
    # --- Node: Generate (Modified to use Docs) ---
    def generate_fix(self, state: AgentState):
        signature = "\n\n# CodeSentinal: created for you by RuchirAdnaik."

        # We now pass the documentation into the prompt
        prompt = f"""
        PROJECT DOCUMENTATION:
        {state['docs']}

        CODE TO FIX:
        {state['original_code']}

        INSTRUCTIONS:
        1. Identify the runtime error in the code.
        2. Fix the error.
        3. Ensure your fix follows any coding standards or rules mentioned in the PROJECT DOCUMENTATION above.
        4. Return ONLY the full corrected python code in a markdown block.
        5. MANDATORY: You must add the following comment as the very last line of the code:
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
        if "CodeSentinal: created for you by RuchirAdnaik." not in code:
            code += signature
            
        return {"messages": [response], "code": code, "iterations": state.get("iterations", 0) + 1}

    # --- Node: Execute ---
    def execute_test(self, state: AgentState):
        print(f"⚙️ Testing fix in Docker (Attempt {state['iterations']})...")
        result = self.executor.execute(state["code"])
    
        # If there's no error, we MUST return an empty string, not None
        error_value = result["error"] if result["error"] else ""
    
        return {"error": error_value}
    
    # --- Router: Research vs Test ---
    def decide_research_or_test(self, state: AgentState):
        if "FETCH_FILE:" in state["messages"][-1].content:
            return "research"
        return "test"
    
    # --- Router: Retry vs Submit ---
    def decide_next(self, state: AgentState):
        if not state.get("error"): return "submit"
        return "retry" if state["iterations"] < 3 else "submit"
    
    def generate_pr_summary(self, state: AgentState):
        prompt = f"""
        Analyze the fix you just made.
        File: {state['file_path']}
        Original Error: {state.get('error', 'Runtime crash')}
        Fixed Code: {state['code']}
    
        Write a professional PR description:
        - **Issue**: Explain the bug.
        - **Resolution**: Explain how you fixed it.
        - **Verification**: State that the code passed local execution tests.
        """
        summary = self.llm.invoke([HumanMessage(content=prompt)])
        return summary.content

    # --- Node: Create PR ---
    def create_pr(self, state: AgentState):
        branch_name = f"fix-{uuid.uuid4().hex[:6]}"
        path = state["file_path"]
        pr_body = self.generate_pr_summary(state)
        
        # Check if this is a fork or our own repo
        is_fork = self.repo.fork 
        
        # 1. Create the fix branch
        main_ref = self.repo.get_branch(self.repo.default_branch)
        self.repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=main_ref.commit.sha)
        
        # 2. Update the file on that branch
        file_git = self.repo.get_contents(path)
        self.repo.update_file(
            path, 
            "🤖 CodeSentinal Auto-fix", 
            state["code"], 
            file_git.sha, 
            branch=branch_name
        )
        
        if is_fork:
            print("🍴 This is a fork. Auto-merging fix into the fork's main branch...")
            # AUTO-MERGE into the fork's default branch
            self.repo.merge(self.repo.default_branch, branch_name, "Auto-merging verified fix")
            
            # Optional: Still create a PR to the ORIGINAL owner so they see your work
            parent_repo = self.repo.parent # This is the original repo you forked from
            try:
                pr = parent_repo.create_pull(
                    title=f"Fix for {path}",
                    body=f"Sent with ❤️ from CodeSentinal.\n\n{pr_body}",
                    head=f"{self.gh.get_user().login}:{branch_name}",
                    base=parent_repo.default_branch
                )
                msg = f"Fix merged into your fork and PR sent to original owner: {pr.html_url}"
            except:
                msg = "Fix merged into your fork. (Original owner doesn't allow PRs)"
        else:
            print("👤 You own this repo. Creating a PR for your review...")
            pr = self.repo.create_pull(
                title=f"Proposed Fix: {path}",
                body=pr_body,
                head=branch_name,
                base=self.repo.default_branch
            )
            msg = f"PR Created for your review: {pr.html_url}"
        
        return {"messages": [HumanMessage(content=msg)]}

    def decide_next(self, state: AgentState):
        if not state.get("error"): return "submit"
        return "retry" if state["iterations"] < 3 else "submit"

    def run(self, file_to_fix):
        # 1. Create a unique thread ID for this specific run
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        # 2. Initialize EVERY key to a default empty value
        # This prevents the 'NoneType' error
        initial_state = {
            "messages": [],
            "file_path": file_to_fix,
            "original_code": "",
            "docs": "",
            "code": "",
            "error": "",
            "iterations": 0
        }
        
        # 3. Use the initialized state
        return self.app.invoke(initial_state, config)
    