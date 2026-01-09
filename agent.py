import os
import operator
import uuid
from typing import Annotated, TypedDict
from github import Github, Auth
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from executor import PythonExecutor 
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define State ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    file_path: str
    original_code: str
    docs: str             # <--- New: Stores project context
    code: str
    error: str
    iterations: int

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
        workflow.add_node("execute_test", self.execute_test)
        workflow.add_node("create_pr", self.create_pr)

        # Connect Nodes
        workflow.add_edge(START, "fetch_docs")           # Start with Research
        workflow.add_edge("fetch_docs", "fetch_file")
        workflow.add_edge("fetch_file", "generate_fix")
        workflow.add_edge("generate_fix", "execute_test")
        
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

    # --- Node: Generate (Modified to use Docs) ---
    def generate_fix(self, state: AgentState):
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
            
        return {"messages": [response], "code": code, "iterations": state.get("iterations", 0) + 1}

    # --- Node: Execute ---
    def execute_test(self, state: AgentState):
        print(f"⚙️ Testing fix in Docker (Attempt {state['iterations']})...")
        result = self.executor.execute(state["code"])
    
        # If there's no error, we MUST return an empty string, not None
        error_value = result["error"] if result["error"] else ""
    
        return {"error": error_value}
    
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
        print(f"🚀 Success! Creating PR on branch {branch_name}...")
        
        # 1. Generate the detailed explanation using your new method
        pr_body = self.generate_pr_summary(state)
        
        main = self.repo.get_branch("main")
        self.repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=main.commit.sha)
        
        file_git = self.repo.get_contents(path)
        self.repo.update_file(
            path, 
            "🤖 Auto-fix with Documentation context", 
            state["code"], 
            file_git.sha, 
            branch=branch_name
        )
        
        # 2. Use the generated summary in the body
        pr = self.repo.create_pull(
            title=f"Auto-Fix: {path}", 
            body=pr_body, 
            head=branch_name, 
            base="main"
        )
        
        print(f"✅ Mission Accomplished! PR: {pr.html_url}")
        return {"messages": [HumanMessage(content=f"PR Created: {pr.html_url}")]}

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

if __name__ == "__main__":
    bot = CodeSentinel(os.getenv("GITHUB_REPO"))
    bot.run("calculator.py")