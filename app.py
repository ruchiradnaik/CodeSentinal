import streamlit as st
import os
import difflib
import json
import zipfile
import io
import tempfile
from datetime import datetime
from agent import CodeSentinel
from project_builder import ProjectBuilder
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Production imports
from config import get_settings
from logger import get_logger, setup_logging

load_dotenv()

# --- Initialize logging and configuration ---
settings = get_settings()
logger = get_logger("app")

# Log startup
logger.info(f"CodeSentinel starting - Environment: {settings.app.environment}")

# --- LLM for Coding Chat ---
coding_llm = None
try:
    if settings.openai.api_key:
        coding_llm = ChatOpenAI(
            model=settings.openai.model, 
            temperature=settings.openai.temperature, 
            api_key=settings.openai.api_key
        )
        logger.info(f"LLM initialized: {settings.openai.model}")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")

# --- Page Config ---
st.set_page_config(
    page_title="CodeSentinel AI", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Fix chat input to bottom of viewport */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important;
        left: var(--sidebar-width, 21rem);
        right: 0;
        padding: 1rem 2rem 1rem 2rem;
        background: #0e1117;
        z-index: 999;
        border-top: 1px solid #333;
    }
    
    /* When sidebar is collapsed */
    [data-testid="stSidebar"][aria-expanded="false"] ~ .main [data-testid="stChatInput"] {
        left: 0;
    }
    
    /* Add padding to main content so messages don't hide behind input */
    .main .block-container {
        padding-bottom: 120px !important;
    }
    
    /* Chat messages styling */
    [data-testid="stChatMessage"] { 
        border-radius: 15px; 
        margin-bottom: 10px; 
    }
    
    .stStatusWidget { border-radius: 10px; }
    .diff-added { background-color: #1e4620; padding: 2px 5px; border-radius: 3px; }
    .diff-removed { background-color: #5c1e1e; padding: 2px 5px; border-radius: 3px; }
    .file-tree { font-family: monospace; font-size: 14px; }
    .metric-card { 
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .log-output {
        background-color: #0d1117;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 12px;
        max-height: 300px;
        overflow-y: auto;
    }
    .template-card {
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        cursor: pointer;
    }
    .template-card:hover {
        border-color: #4CAF50;
        background-color: #1a1a2e;
    }
</style>
<script>
    // Auto-scroll to bottom of page when new content is added
    const scrollToBottom = () => {
        window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
    };
    // Run on load and observe for changes
    setTimeout(scrollToBottom, 100);
    const observer = new MutationObserver(scrollToBottom);
    observer.observe(document.body, {childList: true, subtree: true});
</script>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "🔧 Fix Code"
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "repo_structure" not in st.session_state:
    st.session_state.repo_structure = None
if "settings" not in st.session_state:
    st.session_state.settings = {
        "max_retries": 3,
        "auto_commit": True,
        "show_diffs": True,
        "language": "Python"
    }

# --- Project Templates ---
PROJECT_TEMPLATES = {
    "🌐 REST API (FastAPI)": {
        "description": "A REST API with FastAPI, SQLAlchemy ORM, JWT authentication, and Swagger docs",
        "tech_stack": "Python, FastAPI, SQLAlchemy, SQLite, Pydantic"
    },
    "🤖 Discord Bot": {
        "description": "A Discord bot with slash commands, event handlers, and database integration",
        "tech_stack": "Python, discord.py, SQLite"
    },
    "📊 Data Dashboard": {
        "description": "An interactive data dashboard with charts, filters, and data upload",
        "tech_stack": "Python, Streamlit, Pandas, Plotly"
    },
    "🛒 E-commerce API": {
        "description": "E-commerce backend with products, cart, orders, and payment integration",
        "tech_stack": "Python, FastAPI, SQLAlchemy, Stripe"
    },
    "📝 Blog Platform": {
        "description": "A blog platform with posts, comments, user auth, and markdown support",
        "tech_stack": "Python, Flask, SQLAlchemy, Markdown"
    },
    "🔐 Auth Service": {
        "description": "Authentication microservice with OAuth2, JWT, and role-based access",
        "tech_stack": "Python, FastAPI, PyJWT, bcrypt"
    },
    "📁 File Manager API": {
        "description": "File upload/download API with cloud storage integration",
        "tech_stack": "Python, FastAPI, boto3, SQLAlchemy"
    },
    "🧪 Testing Framework": {
        "description": "A testing framework setup with pytest, coverage, and CI/CD config",
        "tech_stack": "Python, pytest, coverage, GitHub Actions"
    }
}

# --- Input Validation Functions ---
def validate_repo_url(repo: str) -> tuple[bool, str]:
    """Validate GitHub repository URL/path"""
    if not repo:
        return False, "Repository cannot be empty. Enter as 'username/repo'"
    
    repo = repo.strip()
    
    # Handle full URLs
    if repo.startswith("https://github.com/"):
        repo = repo.replace("https://github.com/", "")
    if repo.startswith("http://github.com/"):
        repo = repo.replace("http://github.com/", "")
    if repo.startswith("github.com/"):
        repo = repo.replace("github.com/", "")
    
    # Remove trailing slashes and .git
    repo = repo.rstrip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]
    
    # Validate format
    parts = repo.split("/")
    if len(parts) != 2:
        return False, f"Invalid format '{repo}'. Use 'username/repo' format"
    
    username, reponame = parts
    
    # Validate username
    if not username or len(username) < 1:
        return False, "Username cannot be empty"
    if username.startswith("-") or username.endswith("-"):
        return False, "Username cannot start or end with hyphen"
    
    # Validate repo name
    if not reponame or len(reponame) < 1:
        return False, "Repository name cannot be empty"
    
    # Check for invalid characters
    import re
    if not re.match(r'^[a-zA-Z0-9._-]+$', reponame):
        return False, "Repository name contains invalid characters"
    
    return True, repo  # Return cleaned repo string

def validate_project_description(description: str) -> tuple[bool, str]:
    """Validate project description"""
    if not description:
        return False, "Project description cannot be empty"
    
    description = description.strip()
    
    if len(description) < 10:
        return False, "Please provide a more detailed description (at least 10 characters)"
    
    if len(description) > 5000:
        return False, "Description is too long (max 5000 characters)"
    
    # Check if it's just gibberish/random characters
    words = description.split()
    if len(words) < 3:
        return False, "Please describe your project in at least a few words"
    
    return True, description

def validate_github_token_for_action(action: str, github_token: str = None) -> tuple[bool, str]:
    """Validate GitHub token is available for specific actions"""
    token = github_token or os.getenv("GITHUB_TOKEN")
    
    if not token:
        if action == "fix":
            return False, "GitHub token required to fix repositories. Set GITHUB_TOKEN in .env file"
        elif action == "create":
            return False, "GitHub token required to create repositories. Set GITHUB_TOKEN in .env file"
        return False, "GitHub token not configured"
    
    # Basic token format check
    if len(token) < 20:
        return False, "GitHub token appears to be invalid (too short)"
    
    return True, ""

def validate_uploaded_files(files) -> tuple[bool, str, list]:
    """Validate uploaded files"""
    if not files:
        return False, "No files uploaded", []
    
    valid_files = []
    errors = []
    
    for file in files:
        # Check file extension
        if not file.name.endswith(".py"):
            errors.append(f"'{file.name}' is not a Python file")
            continue
        
        # Check file size (max 1MB)
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if size > 1024 * 1024:  # 1MB
            errors.append(f"'{file.name}' is too large (max 1MB)")
            continue
        
        if size == 0:
            errors.append(f"'{file.name}' is empty")
            continue
        
        # Try to read content
        try:
            content = file.read().decode("utf-8")
            file.seek(0)
            
            # Basic Python syntax check
            if not content.strip():
                errors.append(f"'{file.name}' has no code content")
                continue
            
            valid_files.append(file)
        except UnicodeDecodeError:
            errors.append(f"'{file.name}' is not a valid text file")
            continue
    
    if not valid_files:
        return False, "No valid Python files: " + "; ".join(errors), []
    
    warning = "; ".join(errors) if errors else ""
    return True, warning, valid_files

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent issues"""
    if not text:
        return ""
    # Remove potentially dangerous characters
    sanitized = text.strip()
    # Limit length
    if len(sanitized) > 10000:
        sanitized = sanitized[:10000]
    return sanitized

# --- Helper Functions ---
def create_diff_html(original: str, modified: str, filename: str = "file") -> str:
    """Create a side-by-side diff view"""
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"original/{filename}",
        tofile=f"fixed/{filename}",
        lineterm=""
    )
    
    diff_lines = []
    for line in diff:
        if line.startswith('+') and not line.startswith('+++'):
            diff_lines.append(f'<span class="diff-added">{line}</span>')
        elif line.startswith('-') and not line.startswith('---'):
            diff_lines.append(f'<span class="diff-removed">{line}</span>')
        else:
            diff_lines.append(line)
    
    return '\n'.join(diff_lines)

def get_repo_structure(repo) -> dict:
    """Get repository file structure"""
    structure = {"folders": [], "files": []}
    
    def scan_contents(path=""):
        try:
            contents = repo.get_contents(path)
            if not isinstance(contents, list):
                contents = [contents]
            
            for item in contents:
                if item.type == "dir":
                    if not any(skip in item.path for skip in ["__pycache__", ".git", "node_modules", "venv", ".env"]):
                        structure["folders"].append(item.path)
                        scan_contents(item.path)
                else:
                    structure["files"].append({
                        "path": item.path,
                        "size": item.size,
                        "type": item.path.split(".")[-1] if "." in item.path else "unknown"
                    })
        except Exception as e:
            pass
    
    scan_contents()
    return structure

def create_zip_download(files: dict, project_name: str) -> bytes:
    """Create a zip file from generated files"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filepath, content in files.items():
            zip_file.writestr(f"{project_name}/{filepath}", content)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def render_file_tree(structure: dict) -> str:
    """Render a file tree view"""
    tree = []
    
    # Sort folders and files
    folders = sorted(set(structure.get("folders", [])))
    files = sorted(structure.get("files", []), key=lambda x: x["path"])
    
    # Create tree
    for folder in folders:
        depth = folder.count("/")
        indent = "  " * depth
        name = folder.split("/")[-1]
        tree.append(f"{indent}📁 {name}/")
    
    for file in files:
        path = file["path"]
        depth = path.count("/")
        indent = "  " * depth
        name = path.split("/")[-1]
        icon = get_file_icon(file["type"])
        tree.append(f"{indent}{icon} {name}")
    
    return "\n".join(tree)

def get_file_icon(file_type: str) -> str:
    """Get icon for file type"""
    icons = {
        "py": "🐍",
        "js": "📜",
        "ts": "📘",
        "json": "📋",
        "md": "📝",
        "txt": "📄",
        "yaml": "⚙️",
        "yml": "⚙️",
        "html": "🌐",
        "css": "🎨",
        "sql": "🗄️",
        "sh": "💻",
        "dockerfile": "🐳",
        "gitignore": "🙈"
    }
    return icons.get(file_type.lower(), "📄")

def add_to_history(action: str, details: dict):
    """Add action to history"""
    st.session_state.history.append({
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details
    })

# --- Main Title ---
st.title("🤖 CodeSentinel AI")
st.caption("AI-Powered Development Assistant • Fix Code • Create Projects • Get Help")

# --- Sidebar ---
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Mode Selection with icons
    mode = st.radio(
        "Select Mode",
        ["🔧 Fix Code", "🏗️ Create Project", "📤 Local Files", "💬 Chat Assistant", "⚙️ Settings"],
        help="Choose what you want CodeSentinel to do"
    )
    
    st.divider()
    
    # GitHub Status
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        try:
            from github import Github, Auth
            auth = Auth.Token(github_token)
            gh = Github(auth=auth)
            user = gh.get_user()
            
            st.success(f"✅ Connected as **{user.login}**")
            
            with st.expander("📊 Account Info"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Public Repos", user.public_repos)
                with col2:
                    st.metric("Private Repos", user.total_private_repos or "N/A")
                
                st.caption(f"Plan: {user.plan.name if user.plan else 'Free'}")
        except Exception as e:
            st.warning(f"⚠️ Token issue: {str(e)[:30]}...")
    else:
        st.error("❌ GitHub Token Missing")
        with st.expander("🔑 Setup Instructions"):
            st.markdown("""
            1. Go to [GitHub Tokens](https://github.com/settings/tokens/new)
            2. Select scopes: `repo` (full access)
            3. Copy token to `.env` file:
            ```
            GITHUB_TOKEN=your_token_here
            ```
            """)
    
    st.divider()
    
    # Mode-specific sidebar content
    if "Fix Code" in mode:
        st.subheader("📍 Repository")
        repo_input = st.text_input(
            "Enter repository",
            value=os.getenv("GITHUB_REPO", ""),
            placeholder="user/repo",
            help="Supports public and private repos"
        )
        
        branch_input = st.text_input(
            "Branch (optional)",
            placeholder="main",
            help="Leave empty for default branch"
        )
        
        if repo_input and "/" in repo_input:
            if st.button("🔍 Browse Repository"):
                with st.spinner("Loading structure..."):
                    try:
                        auth = Auth.Token(github_token)
                        gh = Github(auth=auth)
                        repo = gh.get_repo(repo_input)
                        st.session_state.repo_structure = get_repo_structure(repo)
                        st.success("Structure loaded!")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    elif "Create Project" in mode:
        st.subheader("🏗️ Project Options")
        
        project_private = st.checkbox("🔒 Private Repository", value=False)
        
        tech_stack = st.text_input(
            "Tech Stack (optional)",
            placeholder="Python, FastAPI, PostgreSQL"
        )
        
        st.subheader("📚 Quick Templates")
        selected_template = st.selectbox(
            "Choose a template",
            ["Custom Project"] + list(PROJECT_TEMPLATES.keys())
        )
        
        if selected_template != "Custom Project":
            template = PROJECT_TEMPLATES[selected_template]
            st.info(f"**{selected_template}**\n\n{template['description']}")
    
    elif "Local Files" in mode:
        st.subheader("📤 Upload Files")
        uploaded_files = st.file_uploader(
            "Upload Python files",
            type=["py"],
            accept_multiple_files=True,
            help="Upload files to analyze and fix"
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} file(s) uploaded")
    
    elif "Settings" in mode:
        st.subheader("⚙️ Configuration")
        
        st.session_state.settings["max_retries"] = st.slider(
            "Max Fix Retries",
            min_value=1,
            max_value=5,
            value=st.session_state.settings["max_retries"]
        )
        
        st.session_state.settings["show_diffs"] = st.checkbox(
            "Show Code Diffs",
            value=st.session_state.settings["show_diffs"]
        )
        
        st.session_state.settings["auto_commit"] = st.checkbox(
            "Auto-commit fixes",
            value=st.session_state.settings["auto_commit"]
        )
    
    # History Section
    st.divider()
    with st.expander("📜 Recent Activity"):
        if st.session_state.history:
            for item in reversed(st.session_state.history[-5:]):
                st.caption(f"**{item['action']}** - {item['timestamp'][:16]}")
        else:
            st.caption("No recent activity")

# Reset messages if mode changed
if st.session_state.current_mode != mode:
    st.session_state.messages = []
    st.session_state.current_mode = mode

# --- Main Content Area ---
# Create tabs for different views
if "Fix Code" in mode or "Local Files" in mode:
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📂 File Browser", "📊 Results"])
elif "Create Project" in mode:
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Generated Files", "⬇️ Download"])
else:
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📜 History", "ℹ️ Help"])

with tab1:
    # Welcome message based on mode
    if not st.session_state.messages:
        if "Fix Code" in mode:
            welcome = """👋 **Welcome to CodeSentinel Fix Mode!**

I can automatically scan, test, and fix Python code in your GitHub repositories.

**How to use:**
1. Enter your repository in the sidebar (e.g., `username/repo`)
2. Click "Browse Repository" to explore the structure
3. Type "fix" or describe what you want me to do
4. I'll create a PR with all the fixes!

**Supports:** Public & Private repos (with correct token scope)"""
        elif "Create Project" in mode:
            welcome = """👋 **Welcome to CodeSentinel Project Builder!**

I can create complete projects from your descriptions.

**How to use:**
1. Choose a template from the sidebar OR
2. Describe your project idea in detail
3. I'll generate all files with working code
4. Download locally or push to GitHub!

**Try saying:** "Create a REST API for a todo app with user authentication" """
        elif "Local Files" in mode:
            welcome = """👋 **Welcome to Local File Mode!**

Upload Python files to analyze and fix without GitHub.

**How to use:**
1. Upload Python files in the sidebar
2. I'll analyze them for errors
3. Get fixed code and download the results

**Tip:** Great for quick fixes before committing!"""
        else:
            welcome = """👋 **Welcome to CodeSentinel Chat!**

I'm your AI coding assistant, specialized **only** in software development topics.

**I can help you with:**
- 🐛 Debugging and fixing code errors
- 🏗️ Architecture and design patterns
- 📚 Best practices and code reviews
- 🔧 Programming questions & tool recommendations
- 🗄️ Databases, APIs, and DevOps

**Note:** I only answer coding-related questions. For general knowledge or off-topic queries, I'll politely let you know!

**Tip:** Switch to other modes in the sidebar to create projects or fix repos!"""
        
        st.session_state.messages.append({"role": "assistant", "content": welcome})
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    placeholder_map = {
        "🔧 Fix Code": "Type 'fix' to start, or describe what you need...",
        "🏗️ Create Project": "Describe your project idea...",
        "📤 Local Files": "Describe what you want to fix...",
        "💬 Chat Assistant": "Ask a coding question...",
        "⚙️ Settings": "Settings mode - configure in sidebar"
    }
    
    if user_query := st.chat_input(placeholder_map.get(mode, "Type here...")):
        # Add to history and display user message immediately
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").markdown(user_query)
        
        with st.chat_message("assistant"):
            # ==================== FIX CODE MODE ====================
            if "Fix Code" in mode:
                repo = repo_input if 'repo_input' in dir() else ""
                
                # Extract repo from query if needed
                if not repo and "/" in user_query:
                    for word in user_query.split():
                        if "/" in word and len(word.split("/")) == 2:
                            repo = word.strip(",.!?\"'")
                            break
                
                # Validate repository URL
                repo_valid, repo_result = validate_repo_url(repo)
                
                if not repo_valid:
                    st.warning(f"⚠️ {repo_result}")
                    st.info("💡 **Tip:** Enter a repository in the format `username/repo` or paste the full GitHub URL")
                    st.session_state.messages.append({"role": "assistant", "content": f"Please enter a valid repository. {repo_result}"})
                
                # Validate GitHub token
                elif not validate_github_token_for_action("fix")[0]:
                    st.error("❌ GitHub token not configured!")
                    st.info("💡 Set your GitHub token in the `.env` file: `GITHUB_TOKEN=your_token_here`")
                    st.session_state.messages.append({"role": "assistant", "content": "Please configure your GitHub token first."})
                
                else:
                    repo = repo_result  # Use cleaned repo string
                    with st.status("🛠️ CodeSentinel Engine Running...", expanded=True) as status:
                        log_container = st.empty()
                        logs = []
                        
                        def log(msg):
                            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                            log_container.code("\n".join(logs[-10:]), language="")
                        
                        try:
                            log(f"🔗 Connecting to {repo}...")
                            bot = CodeSentinel(repo)
                            
                            log("📦 Detecting dependencies...")
                            log("📚 Indexing codebase with FAISS...")
                            log("🧪 Running tests on all files...")
                            
                            final_state = bot.run()
                            
                            log("🔧 Applying AI-powered fixes...")
                            log("🚀 Creating Pull Request...")
                            
                            status.update(label="✅ Complete!", state="complete", expanded=False)
                            
                            # Store results
                            st.session_state.last_result = final_state
                            
                            # Display summary
                            fixed_files = final_state.get("fixed_files", {})
                            all_files = final_state.get("all_files", [])
                            original_codes = final_state.get("original_codes", {})
                            
                            st.markdown("### ✅ Fix Complete!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📁 Total Files", len(all_files))
                            with col2:
                                st.metric("🧪 Tested", len(all_files))
                            with col3:
                                st.metric("🔧 Fixed", len(fixed_files))
                            
                            # Show diffs if enabled
                            if st.session_state.settings["show_diffs"] and fixed_files:
                                st.markdown("### 📝 Changes Made")
                                for filepath, fixed_code in fixed_files.items():
                                    original = original_codes.get(filepath, "")
                                    if original.strip() != fixed_code.strip():
                                        with st.expander(f"📄 {filepath}", expanded=False):
                                            diff = create_diff_html(original, fixed_code, filepath)
                                            st.code(diff, language="diff")
                            
                            # PR link
                            pr_link = "Check your repository!"
                            for msg in reversed(final_state.get("messages", [])):
                                if hasattr(msg, 'content') and ("PR Created" in msg.content or "PR sent" in msg.content):
                                    pr_link = msg.content
                                    break
                            
                            st.success(f"**{pr_link}**")
                            
                            add_to_history("Fixed Repository", {"repo": repo, "files_fixed": len(fixed_files)})
                            st.session_state.messages.append({"role": "assistant", "content": f"✅ Fixed {len(fixed_files)} files in {repo}!"})
                            
                        except Exception as e:
                            status.update(label="❌ Error", state="error")
                            st.error(f"Error: {str(e)}")
                            import traceback
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())
            
            # ==================== CREATE PROJECT MODE ====================
            elif "Create Project" in mode:
                # Sanitize and validate description
                description = sanitize_input(user_query)
                desc_valid, desc_result = validate_project_description(description)
                
                if not desc_valid:
                    st.warning(f"⚠️ {desc_result}")
                    st.info("💡 **Tip:** Describe what your project should do, what features it needs, and any specific technologies.")
                    st.session_state.messages.append({"role": "assistant", "content": f"Please provide a better description. {desc_result}"})
                
                # Validate GitHub token for pushing
                elif not validate_github_token_for_action("create")[0]:
                    st.warning("⚠️ GitHub token not configured - project will be available for download only")
                    # Continue without GitHub push
                    with st.status("🏗️ Building Project (Local Only)...", expanded=True) as status:
                        try:
                            builder = ProjectBuilder()
                            
                            stack = tech_stack if 'tech_stack' in dir() and tech_stack else None
                            
                            if 'selected_template' in dir() and selected_template != "Custom Project":
                                template = PROJECT_TEMPLATES[selected_template]
                                description = f"{template['description']}. User request: {description}"
                                stack = template['tech_stack']
                            
                            st.write("🧠 Generating project structure...")
                            
                            result = builder.build_project(
                                description=description,
                                tech_stack=stack,
                                push_to_github=False,  # Don't push without token
                                private=False
                            )
                            
                            if result.get("success"):
                                status.update(label="✅ Project Generated!", state="complete")
                                st.session_state.last_result = result
                                structure = result.get("structure", {})
                                
                                st.markdown(f"### 🎉 Generated: **{result.get('project_name')}**")
                                st.info("📥 Go to the **Download** tab to get your project files")
                                
                                add_to_history("Generated Project (Local)", {"name": result.get("project_name")})
                                st.session_state.messages.append({"role": "assistant", "content": f"✅ Generated project: {result.get('project_name')}! Download from the Download tab."})
                            else:
                                status.update(label="❌ Failed", state="error")
                                st.error(f"Error: {result.get('error')}")
                        
                        except Exception as e:
                            status.update(label="❌ Error", state="error")
                            st.error(f"Error: {str(e)}")
                
                else:
                    with st.status("🏗️ Building Project...", expanded=True) as status:
                        try:
                            builder = ProjectBuilder()
                            
                            stack = tech_stack if 'tech_stack' in dir() and tech_stack else None
                            
                            if 'selected_template' in dir() and selected_template != "Custom Project":
                                template = PROJECT_TEMPLATES[selected_template]
                                description = f"{template['description']}. User request: {description}"
                                stack = template['tech_stack']
                            
                            st.write("🧠 Generating project structure...")
                            
                            result = builder.build_project(
                                description=description,
                                tech_stack=stack,
                                push_to_github=True,
                                private=project_private if 'project_private' in dir() else False
                            )
                            
                            if result.get("success"):
                                status.update(label="✅ Project Created!", state="complete")
                                
                                st.session_state.last_result = result
                                structure = result.get("structure", {})
                                
                                st.markdown(f"### 🎉 Created: **{result.get('project_name')}**")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("📁 Files", result.get("files_count", 0))
                                with col2:
                                    st.metric("📦 Dependencies", len(structure.get("dependencies", [])))
                                with col3:
                                    st.metric("🛠️ Stack", len(structure.get("tech_stack", [])))
                                
                                if result.get("repo_url"):
                                    st.success(f"🔗 **Repository:** {result.get('repo_url')}")
                                    st.balloons()
                                
                                add_to_history("Created Project", {"name": result.get("project_name")})
                                st.session_state.messages.append({"role": "assistant", "content": f"✅ Created project: {result.get('project_name')}!"})
                            else:
                                status.update(label="❌ Failed", state="error")
                                st.error(f"Error: {result.get('error')}")
                        
                        except Exception as e:
                            status.update(label="❌ Error", state="error")
                            st.error(f"Error: {str(e)}")
            
            # ==================== LOCAL FILES MODE ====================
            elif "Local Files" in mode:
                if 'uploaded_files' in dir() and uploaded_files:
                    # Validate uploaded files
                    files_valid, files_warning, valid_files = validate_uploaded_files(uploaded_files)
                    
                    if not files_valid:
                        st.error(f"❌ {files_warning}")
                        st.info("💡 **Tip:** Upload valid Python (.py) files under 1MB")
                        st.session_state.messages.append({"role": "assistant", "content": f"File validation failed: {files_warning}"})
                    
                    else:
                        if files_warning:
                            st.warning(f"⚠️ Some files skipped: {files_warning}")
                        
                        with st.status("🔍 Analyzing files...", expanded=True) as status:
                            try:
                                from executor import PythonExecutor
                                executor = PythonExecutor()
                                
                                results = {}
                                for file in valid_files:
                                    try:
                                        content = file.read().decode("utf-8")
                                        file.seek(0)  # Reset for potential re-read
                                        
                                        st.write(f"Testing: {file.name}")
                                        result = executor.execute(content)
                                        
                                        # Handle warnings from input() mocking
                                        warning = result.get("warning", "")
                                        
                                        results[file.name] = {
                                            "original": content,
                                            "success": result.get("success", False),
                                            "error": result.get("error", ""),
                                            "warning": warning
                                        }
                                    except Exception as file_error:
                                        results[file.name] = {
                                            "original": "",
                                            "success": False,
                                            "error": f"Failed to process file: {str(file_error)}",
                                            "warning": ""
                                        }
                                
                                # Show results
                                status.update(label="✅ Analysis Complete", state="complete")
                                
                                passed = sum(1 for r in results.values() if r["success"])
                                failed = len(results) - passed
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("✅ Passed", passed)
                                with col2:
                                    st.metric("❌ Failed", failed)
                                
                                for filename, data in results.items():
                                    if data["success"]:
                                        if data.get("warning"):
                                            st.info(f"⚠️ {filename} - {data['warning']}")
                                        else:
                                            st.success(f"✅ {filename} - No errors!")
                                    else:
                                        with st.expander(f"❌ {filename} - Has errors"):
                                            st.code(data["error"], language="")
                                
                                st.session_state.last_result = results
                                st.session_state.messages.append({"role": "assistant", "content": f"Analyzed {len(results)} files. {passed} passed, {failed} failed."})
                            
                            except Exception as e:
                                status.update(label="❌ Error", state="error")
                                st.error(f"Error: {str(e)}")
                else:
                    st.info("👈 Upload Python files in the sidebar to analyze them!")
            
            # ==================== CHAT MODE ====================
            elif "Chat" in mode:
                if coding_llm:
                    # Build conversation history for context
                    conversation = [
                        SystemMessage(content="""You are CodeSentinel, an AI coding assistant created to help developers with software development.

**ALWAYS answer these types of questions:**
- Questions about yourself (your name is CodeSentinel, you're a coding assistant)
- Code explanations, debugging, and fixing errors
- Programming in any language (Python, JavaScript, Java, C++, Go, Rust, etc.)
- Software architecture, design patterns, best practices
- Frameworks, libraries, databases, APIs
- DevOps, Git, Docker, CI/CD, cloud services
- Any code shared by the user - analyze it, explain it, improve it

**POLITELY DECLINE these types of questions:**
- Politics (elections, politicians, government)
- Celebrities, entertainment, sports
- General knowledge unrelated to tech (geography, history trivia)
- Current events, news, weather
- Medical, legal, or financial advice

**When declining, say something like:**
"I'm CodeSentinel, your coding assistant! I specialize in software development topics like debugging, code reviews, and architecture. I can't help with [topic], but feel free to ask me anything about coding!"

**Key behaviors:**
- If user shares code, ALWAYS help explain, debug, or improve it
- Be thorough when explaining code - go line by line if asked
- For project creation or fixing repos, mention the sidebar modes""")
                    ]
                    
                    # Add conversation history (skip welcome message, keep last 10 exchanges)
                    history_messages = [m for m in st.session_state.messages if m["content"] != st.session_state.messages[0]["content"]]
                    for msg in history_messages[-20:]:  # Keep last 20 messages for context
                        if msg["role"] == "user":
                            conversation.append(HumanMessage(content=msg["content"]))
                        else:
                            conversation.append(AIMessage(content=msg["content"]))
                    
                    # Add current query
                    conversation.append(HumanMessage(content=user_query))
                    
                    response = coding_llm.invoke(conversation)
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                else:
                    st.error("OpenAI API key not configured!")
            
            # ==================== SETTINGS MODE ====================
            elif "Settings" in mode:
                st.info("⚙️ Configure settings in the sidebar. Changes are saved automatically!")
                st.json(st.session_state.settings)

# Tab 2 - File Browser / Generated Files / History
with tab2:
    if "Fix Code" in mode:
        st.subheader("📂 Repository Structure")
        if st.session_state.repo_structure:
            structure = st.session_state.repo_structure
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📁 Folders", len(structure.get("folders", [])))
            with col2:
                st.metric("📄 Files", len(structure.get("files", [])))
            
            st.text(render_file_tree(structure))
        else:
            st.info("👈 Enter a repository and click 'Browse Repository' in the sidebar")
    
    elif "Create Project" in mode:
        st.subheader("📁 Generated Files")
        if (st.session_state.last_result 
            and isinstance(st.session_state.last_result, dict) 
            and st.session_state.last_result.get("structure")):
            structure = st.session_state.last_result["structure"]
            
            for file_info in structure.get("files", []):
                if isinstance(file_info, dict):
                    with st.expander(f"📄 {file_info.get('path', 'unknown')}"):
                        file_path = file_info.get("path", "")
                        lang = "python" if file_path.endswith(".py") else ("javascript" if file_path.endswith(".js") else "")
                        st.code(file_info.get("content", ""), language=lang)
        else:
            st.info("Generate a project to see files here!")
    
    elif "Local Files" in mode:
        st.subheader("📊 Analysis Results")
        if st.session_state.last_result:
            # Check if the result is in the expected format for Local Files mode
            result = st.session_state.last_result
            if isinstance(result, dict) and all(isinstance(v, dict) and "success" in v for v in result.values() if isinstance(v, dict)):
                for filename, data in result.items():
                    if isinstance(data, dict) and "success" in data:
                        status_icon = "✅" if data.get("success") else "❌"
                        with st.expander(f"{status_icon} {filename}"):
                            st.code(data.get("original", ""), language="python")
                            if data.get("error"):
                                st.error(data["error"])
            else:
                st.info("No local file analysis results. Upload files to analyze!")
        else:
            st.info("Upload and analyze files to see results!")
    
    else:
        st.subheader("📜 Session History")
        if st.session_state.history:
            for item in reversed(st.session_state.history):
                st.markdown(f"**{item['action']}** - {item['timestamp'][:19]}")
                st.json(item['details'])
                st.divider()
        else:
            st.info("Your activity history will appear here")

# Tab 3 - Results / Download / Help
with tab3:
    if "Fix Code" in mode:
        st.subheader("📊 Last Run Results")
        if st.session_state.last_result and isinstance(st.session_state.last_result, dict):
            result = st.session_state.last_result
            
            # Check if this is a Fix Code result (has fixed_files key)
            if "fixed_files" in result:
                st.metric("Files Fixed", len(result.get("fixed_files", {})))
                
                if result.get("fixed_files"):
                    st.markdown("### 📥 Download Fixed Files")
                    
                    # Create download for fixed files
                    fixed_files_content = {}
                    for filepath, code in result.get("fixed_files", {}).items():
                        fixed_files_content[filepath] = code
                    
                    if fixed_files_content:
                        zip_data = create_zip_download(fixed_files_content, "fixed_files")
                        st.download_button(
                            label="⬇️ Download All Fixed Files (ZIP)",
                            data=zip_data,
                            file_name="fixed_files.zip",
                            mime="application/zip"
                        )
            else:
                st.info("Run a fix to see results here!")
        else:
            st.info("Run a fix to see results here!")
    
    elif "Create Project" in mode:
        st.subheader("⬇️ Download Project")
        if st.session_state.last_result and st.session_state.last_result.get("structure"):
            structure = st.session_state.last_result["structure"]
            project_name = st.session_state.last_result.get("project_name", "project")
            
            # Create download
            files_content = {}
            for file_info in structure.get("files", []):
                files_content[file_info.get("path")] = file_info.get("content", "")
            
            if files_content:
                zip_data = create_zip_download(files_content, project_name)
                st.download_button(
                    label=f"⬇️ Download {project_name}.zip",
                    data=zip_data,
                    file_name=f"{project_name}.zip",
                    mime="application/zip"
                )
                
                st.divider()
                st.markdown("### 📦 Dependencies")
                deps = structure.get("dependencies", [])
                if deps:
                    st.code("\n".join(deps), language="")
                
                st.markdown("### 🚀 Setup Instructions")
                st.markdown(structure.get("setup_instructions", "See README.md"))
        else:
            st.info("Create a project to download it!")
    
    else:
        st.subheader("ℹ️ Help & Tips")
        st.markdown("""
        ### 🔧 Fix Code Mode
        - Works with public AND private repos (needs `repo` token scope)
        - Automatically detects dependencies
        - Uses FAISS for semantic code understanding
        - Creates PR with all fixes
        
        ### 🏗️ Create Project Mode
        - Describe your project in natural language
        - Use templates for common project types
        - Downloads locally or pushes to GitHub
        - Generates working, production-ready code
        
        ### 📤 Local Files Mode
        - Upload Python files directly
        - Get instant error analysis
        - No GitHub required
        
        ### 💬 Chat Mode
        - Ask any coding question
        - Get code examples and explanations
        - Architecture and design advice
        
        ### ⚡ Keyboard Shortcuts
        - `Enter` - Send message
        - `Ctrl+K` - Clear chat
        """)

# --- Footer ---
st.divider()
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.caption("🛡️ CodeSentinel AI - Built by RuchirAdnaik")
with col2:
    st.caption(f"Mode: {mode}")
with col3:
    st.caption(f"Session: {len(st.session_state.history)} actions")
