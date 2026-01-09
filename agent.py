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
        print("📖 Reading project documentation (README.md)...")
        try:
            content = self.repo.get_contents("README.md").decoded_content.decode()
            return {"docs": content}
        except:
            print("⚠️ No README.md found, proceeding without extra context.")
            return {"docs": "No documentation available."}

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
        print(f"⚙️ Testing fix locally (Attempt {state['iterations']})...")
        result = self.executor.execute(state["code"])
        return {"error": result["error"] if not result["success"] else None}

    # --- Node: Create PR ---
    def create_pr(self, state: AgentState):
        branch_name = f"fix-{uuid.uuid4().hex[:6]}"
        path = state["file_path"]
        print(f"🚀 Success! Creating PR on branch {branch_name}...")
        
        main = self.repo.get_branch("main")
        self.repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=main.commit.sha)
        
        file_git = self.repo.get_contents(path)
        self.repo.update_file(path, "🤖 Auto-fix with Documentation context", state["code"], file_git.sha, branch=branch_name)
        
        pr = self.repo.create_pull(title=f"Auto-Fix: {path}", body="Verified fix based on internal documentation.", head=branch_name, base="main")
        print(f"✅ Mission Accomplished! PR: {pr.html_url}")
        return {"messages": [HumanMessage(content=f"PR Created: {pr.html_url}")]}

    def decide_next(self, state: AgentState):
        if not state.get("error"): return "submit"
        return "retry" if state["iterations"] < 3 else "submit"

    def run(self, file_to_fix):
        config = {"configurable": {"thread_id": "gh-doc-1"}}
        self.app.invoke({"file_path": file_to_fix, "iterations": 0, "messages": []}, config)

if __name__ == "__main__":
    bot = CodeSentinel(os.getenv("GITHUB_REPO"))
    bot.run("calculator.py")