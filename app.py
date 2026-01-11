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
    
    st.divider()
    st.markdown("### How it works")
    st.info("""
    1. Provide the GitHub Repository (user/repo).
    2. The agent will scan all Python files.
    3. Test each file and identify failures.
    4. Fix all failing files automatically.
    5. Create a PR with all fixes!
    """)

# --- Chat Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! I'm CodeSentinel. Just provide your GitHub repository (user/repo) and I'll scan all files, test them, fix any failures, and create a PR!"}
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
        
        # Extract repo from user query if not in sidebar
        if not repo and "/" in user_query:
            # Try to extract repo from user query
            parts = user_query.split()
            for part in parts:
                if "/" in part and len(part.split("/")) == 2:
                    repo = part
                    break
        
        # If we have repo, start the engine!
        if "/" in repo:
            with st.status("🛠️ CodeSentinel Engine Starting...", expanded=True) as status:
                try:
                    # Initialize Agent
                    bot = CodeSentinel(repo)
                    
                    st.write("🔍 **Step 1:** Researching repo structure and README...")
                    
                    st.write("📋 **Step 2:** Scanning repository for all Python files...")
                    
                    st.write("🧪 **Step 3:** Testing all files to identify failures...")
                    
                    # RUN THE AGENT (no file_to_fix needed - it will scan all files)
                    final_state = bot.run()
                    
                    st.write("🔧 **Step 4:** Fixing all failing files...")
                    
                    st.write("🚀 **Step 5:** Creating Pull Request with all fixes...")
                    
                    status.update(label="✅ Mission Accomplished!", state="complete", expanded=False)
                    
                    # --- SUCCESS UI ---
                    # 1. Explanation from the bot
                    explanation = final_state.get("messages", [])[-1].content if final_state.get("messages") else "Process completed!"
                    st.markdown(f"### 🧠 Bot's Summary\n{explanation}")
                    
                    # 2. Show fixed files
                    fixed_files = final_state.get("fixed_files", {})
                    failing_files = final_state.get("failing_files", [])
                    all_files = final_state.get("all_files", [])
                    
                    st.divider()
                    st.subheader("📊 Processing Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Files", len(all_files))
                    with col2:
                        st.metric("Files Tested", len(all_files))
                    with col3:
                        st.metric("Files Fixed", len(fixed_files))
                    
                    if fixed_files:
                        st.subheader("📝 Fixed Files")
                        for file_path in fixed_files.keys():
                            st.success(f"✅ {file_path}")
                    
                    if failing_files and len(fixed_files) < len(failing_files):
                        st.warning(f"⚠️ Some files couldn't be fixed after max retries: {len(failing_files) - len(fixed_files)}")

                    # 3. Final Call to Action
                    # Find PR link in messages
                    pr_link = "Check your GitHub repository!"
                    for msg in reversed(final_state.get("messages", [])):
                        if "PR Created" in msg.content or "PR sent" in msg.content or "html_url" in str(msg.content):
                            pr_link = msg.content
                            break
                            
                    st.balloons()
                    st.success(f"**{pr_link}**")
                    st.session_state.messages.append({"role": "assistant", "content": f"I've processed {len(all_files)} file(s) and fixed {len(fixed_files)} file(s)! {pr_link}"})

                except Exception as e:
                    status.update(label="❌ Process Failed", state="error")
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        else:
            # Normal ChatGPT-style conversation if info is missing
            response = "I'm ready to help! Please provide your **GitHub Repository** (user/repo) in the sidebar or in your message, and I'll scan all files, test them, fix any failures, and create a PR!"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})