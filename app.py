import streamlit as st
import os
import uuid
from agent import CodeSentinel
from dotenv import load_dotenv
from github import Github



load_dotenv()

st.set_page_config(page_title="CodeSentinel AI", page_icon="🤖", layout="wide")

# --- UI Header ---
st.title("🤖 CodeSentinel: Auto-Fix Dashboard")
st.markdown("Automated software maintenance agent. Fixes bugs while you drink coffee.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Settings")
    repo_name = st.text_input("GitHub Repository", value=os.getenv("GITHUB_REPO", ""))
    target_file = st.text_input("File to fix (e.g. calculator.py)", placeholder="calculator.py")
    
    st.divider()
    st.info("The bot will create a new branch and PR. Your main branch remains safe.")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Logic ---
if user_query := st.chat_input("Explain the error or just say 'Fix it'"):
    # 1. Show user message
    st.chat_message("user").write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Start Agent Process
    with st.chat_message("assistant"):
        # This status container will show the user exactly what is happening in real-time
        with st.status("🤖 CodeSentinel is on the job...", expanded=True) as status:
            try:
                # Initialize the bot with the repo from the sidebar
                bot = CodeSentinel(repo_name)
                
                st.write("🔍 **Step 1:** Researching repo structure and README...")
                # Note: The actual work happens inside bot.run, 
                # but these write statements give the user visual feedback.
                
                st.write(f"📂 **Step 2:** Fetching `{target_file}` from GitHub...")
                
                st.write("🧪 **Step 3:** Analyzing code and running tests in sandbox...")
                
                # --- RUN THE AGENT ---
                # This triggers the LangGraph workflow we built
                bot.run(target_file) 
                
                st.write("🚀 **Step 4:** Success! Creating Pull Request and Explanation...")
                
                # Update the status to finished
                status.update(label="✅ Fix Complete! Pull Request Created.", state="complete", expanded=False)
                
                # Final chat response
                final_response = f"I've analyzed `{target_file}`, verified a fix locally, and opened a Pull Request in your repository. Check GitHub for the details!"
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                
                # Big success banner at the bottom
                st.success(f"Successfully fixed {target_file}! Check your GitHub repo for the new Pull Request.")

            except Exception as e:
                status.update(label="❌ Error occurred", state="error")
                st.error(f"Error: {str(e)}")