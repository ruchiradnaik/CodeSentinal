import streamlit as st
import os
import difflib
from agent import CodeSentinel
from dotenv import load_dotenv

load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="CodeSentinel AI", page_icon="🤖", layout="wide")

# Custom CSS to make the chat look sleek
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stStatusWidget { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 CodeSentinel: Auto-Fix Agent")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("📍 Project Context")
    # Users can either type it here or the bot will extract it from chat
    repo_input = st.text_input("GitHub Repository", value=os.getenv("GITHUB_REPO", ""), placeholder="user/repo")
    file_input = st.text_input("Target File", placeholder="e.g., app.py")
    
    st.divider()
    st.markdown("### How it works")
    st.info("""
    1. Chat with the bot about your bug.
    2. Provide the Repo and File name.
    3. Watch the agent research, test in Docker, and create a PR!
    """)

# --- Chat Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! I'm CodeSentinel. Facing a bug? Tell me about the issue, or just share your GitHub repo and the file name you want me to fix."}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Logic ---
if user_query := st.chat_input("Explain your error or share repo details..."):
    # 1. Display User Message
    st.chat_message("user").write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Assistant Response Logic
    with st.chat_message("assistant"):
        # Check if we have the required info to start the "Auto-Fix"
        # We check both sidebar and user input
        repo = repo_input if repo_input else ""
        file_to_fix = file_input if file_input else ""
        
        # If we have both, start the engine!
        if "/" in repo and "." in file_to_fix:
            with st.status("🛠️ CodeSentinel Engine Starting...", expanded=True) as status:
                try:
                    # Initialize Agent
                    bot = CodeSentinel(repo)
                    
                    st.write("🔍 **Step 1:** Researching repo structure and README...")
                    # bot.run internally handles the LangGraph flow
                    
                    st.write(f"📂 **Step 2:** Fetching `{file_to_fix}` from GitHub...")
                    
                    st.write("🧪 **Step 3:** Analyzing code and running tests in Docker sandbox...")
                    
                    # RUN THE AGENT
                    # We store the result to extract codes and PR link
                    final_state = bot.run(file_to_fix)
                    
                    st.write("🚀 **Step 4:** Success! Creating Pull Request and Explanation...")
                    
                    status.update(label="✅ Mission Accomplished!", state="complete", expanded=False)
                    
                    # --- SUCCESS UI ---
                    # 1. Explanation from the bot
                    explanation = final_state.get("messages", [])[-1].content
                    st.markdown(f"### 🧠 Bot's Analysis & Fix\n{explanation}")
                    
                    # 2. Side-by-Side Diff Viewer
                    st.divider()
                    st.subheader("📊 Code Comparison")
                    old_code = final_state.get("original_code", "")
                    new_code = final_state.get("code", "")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Original (Broken)")
                        st.code(old_code, language="python")
                    with col2:
                        st.caption("Fixed (Verified)")
                        st.code(new_code, language="python")

                    # 3. Final Call to Action
                    # Find PR link in messages
                    pr_link = "Check your GitHub repository!"
                    for msg in reversed(final_state.get("messages", [])):
                        if "PR Created:" in msg.content:
                            pr_link = msg.content
                            break
                            
                    st.balloons()
                    st.success(f"**{pr_link}**")
                    st.session_state.messages.append({"role": "assistant", "content": f"I've fixed `{file_to_fix}`! {pr_link}"})

                except Exception as e:
                    status.update(label="❌ Process Failed", state="error")
                    st.error(f"Error: {str(e)}")
        
        else:
            # Normal ChatGPT-style conversation if info is missing
            response = "I'm ready to help! Please make sure you've entered the **GitHub Repo** (user/repo) and the **File Name** in the sidebar so I can start the auto-fix process."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})