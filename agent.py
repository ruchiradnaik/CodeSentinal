import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver 
from langgraph.graph import START, MessagesState, StateGraph
from executor import PythonExecutor # Matches the class above
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
print(f"API Key loaded: {OPENAI_API_KEY[:10]}..." if OPENAI_API_KEY else "API Key is None!")

class CodeSentinel:
    def __init__(self):
        # Use your actual key here
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key= OPENAI_API_KEY)
        self.executor = PythonExecutor()
        self.memory = MemorySaver()
        
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_edge(START, "agent")
        
        self.app = workflow.compile(checkpointer=self.memory)

    def call_model(self, state: MessagesState):
        response = self.llm.invoke(state["messages"])
        return {"messages": [response]}

    def chat(self, user_input: str, thread_id: str = "1"):
        config = {"configurable": {"thread_id": thread_id}}
        print(f"\n👤 User: {user_input}")
        
        # Start the conversation
        current_input = user_input
        max_tries = 3
        
        for attempt in range(max_tries):
            # 1. Ask the Brain
            output = self.app.invoke(
                {"messages": [HumanMessage(content=current_input)]}, 
                config
            )
            
            response_text = output["messages"][-1].content
            
            # 2. Check if the Brain wrote code
            if "```python" in response_text:
                code = response_text.split("```python")[1].split("```")[0].strip()
                print(f"⚙️ Attempt {attempt + 1}: Running code...")
                
                # 3. Use the Hands to run it
                result = self.executor.execute(code)
                
                if result['success'] and not result['error']:
                    print(f"✅ Success! Output:\n{result['output']}")
                    break # It worked! Exit the loop.
                else:
                    # 4. IT FAILED! Feed the error back to the Brain
                    print(f"❌ Error found: {result['error'].strip()}")
                    print("🔄 Sending error to Brain for fixing...")
                    
                    # We update the 'current_input' to be the error message
                    current_input = f"That code gave an error: {result['error']}. Please fix the code and try again."
            else:
                print(f"🤖 Agent: {response_text}")
                break

if __name__ == "__main__":
    bot = CodeSentinel()
    # TEST 1: Introduction
    bot.chat("Hi, my name is Ruchir Adnaik.")
    # TEST 2: Memory Test
    bot.chat("What is my name?")
    # TEST 3: Coding Test
    bot.chat("""correct this code for me please: 

""")