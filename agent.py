import os
import operator
from typing import Annotated, TypedDict, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from executor import PythonExecutor
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define the State ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    code: str              # The code currently being tested
    error: str             # The error from the last execution
    iterations: int        # To prevent infinite loops

class CodeSentinel:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.executor = PythonExecutor()
        self.memory = MemorySaver()
        self.max_iterations = 3

        # --- 2. Build the Graph ---
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("generate", self.generate_solution)
        workflow.add_node("execute", self.run_code)
        
        # Edges
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "execute")
        
        # Conditional Edge: Decide next step based on execution result
        workflow.add_conditional_edges(
            "execute",
            self.decide_next_step,
            {
                "retry": "generate",
                "finish": END
            }
        )

        self.app = workflow.compile(checkpointer=self.memory)

    # --- Node: Generate Solution ---
    def generate_solution(self, state: AgentState):
        messages = state["messages"]
        error = state.get("error", "")
        
        # If there is an error from a previous run, add it to context
        if error:
            messages.append(HumanMessage(content=f"The previous code failed with this error: {error}. Please fix it."))

        response = self.llm.invoke(messages)
        
        # Extract code from markdown blocks
        content = response.content
        code = ""
        if "```python" in content:
            code = content.split("```python")[1].split("```")[0].strip()
        
        return {
            "messages": [response], 
            "code": code, 
            "iterations": state.get("iterations", 0) + 1
        }

    # --- Node: Execute Code ---
    def run_code(self, state: AgentState):
        code = state["code"]
        if not code:
            return {"error": "No code generated"}
            
        print(f"⚙️ Executing code... (Iteration {state['iterations']})")
        result = self.executor.execute(code)
        
        return {"error": result["error"] if not result["success"] else None}

    # --- Conditional Logic ---
    def decide_next_step(self, state: AgentState):
        error = state.get("error")
        iterations = state.get("iterations")

        if not error:
            print("✅ Success! Code works.")
            return "finish"
        
        if iterations >= self.max_iterations:
            print("❌ Max retries reached. Stopping.")
            return "finish"
            
        print(f"⚠️ Error found: {error}. Retrying...")
        return "retry"

    def chat(self, user_input: str, thread_id: str = "1"):
        config = {"configurable": {"thread_id": thread_id}}
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "iterations": 0,
            "error": None,
            "code": None
        }
        
        self.app.invoke(initial_state, config)

if __name__ == "__main__":
    bot = CodeSentinel()
    bot.chat("Write a python function to calculate fibonacci sequence up to n, but make a deliberate syntax error so I can see you fix it.")