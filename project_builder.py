"""
Project Builder: Creates entire projects from natural language descriptions
"""
import os
import json
import re
from typing import Dict, List, Optional
from github import Github, Auth
from github.GithubException import GithubException
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


class ProjectBuilder:
    """Builds complete projects from natural language descriptions"""
    
    # Signature comment for different file types
    SIGNATURE = "Project created by: CodeSentinel AI ❤️"
    
    # Comment syntax for different file types
    COMMENT_SYNTAX = {
        ".py": "#",
        ".js": "//",
        ".ts": "//",
        ".jsx": "//",
        ".tsx": "//",
        ".java": "//",
        ".c": "//",
        ".cpp": "//",
        ".cs": "//",
        ".go": "//",
        ".rs": "//",
        ".swift": "//",
        ".kt": "//",
        ".rb": "#",
        ".php": "//",
        ".sh": "#",
        ".bash": "#",
        ".zsh": "#",
        ".yaml": "#",
        ".yml": "#",
        ".toml": "#",
        ".ini": ";",
        ".sql": "--",
        ".html": "<!-- {} -->",
        ".xml": "<!-- {} -->",
        ".css": "/* {} */",
        ".scss": "/* {} */",
        ".less": "/* {} */",
        ".md": "<!-- {} -->",
    }
    
    def __init__(self):
        api_key = os.getenv("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found!")
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=str(api_key))
        
        # GitHub setup
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            auth = Auth.Token(github_token)
            self.gh = Github(auth=auth)
            self.user = self.gh.get_user()
        else:
            self.gh = None
            self.user = None
    
    def _add_signature(self, content: str, file_path: str) -> str:
        """Add CodeSentinel signature comment to file content"""
        
        # Skip if signature already present
        if self.SIGNATURE in content:
            return content
        
        # Get file extension
        ext = ""
        if "." in file_path:
            ext = "." + file_path.rsplit(".", 1)[-1].lower()
        
        # Skip binary/config files that shouldn't have comments
        skip_extensions = {".json", ".lock", ".gitignore", ".env", ".txt", ".csv"}
        if ext in skip_extensions:
            return content
        
        # Get comment syntax for this file type
        comment_syntax = self.COMMENT_SYNTAX.get(ext)
        
        if not comment_syntax:
            return content
        
        # Format the signature comment
        if "{}" in comment_syntax:
            # Block comment style (HTML, CSS, etc.)
            signature_comment = comment_syntax.format(self.SIGNATURE)
        else:
            # Line comment style (Python, JS, etc.)
            signature_comment = f"{comment_syntax} {self.SIGNATURE}"
        
        # Add signature at the end of the file
        content = content.rstrip()
        content = f"{content}\n\n{signature_comment}\n"
        
        return content
    
    def generate_project_structure(self, description: str, tech_stack: str = None) -> Dict:
        """Generate project structure from description using LLM"""
        
        system_prompt = """You are an expert software architect. Generate a complete project structure based on the user's description.

Return a JSON object with this exact structure:
{
    "project_name": "lowercase-with-dashes",
    "description": "Brief description for README",
    "tech_stack": ["python", "flask", etc],
    "files": [
        {
            "path": "relative/path/to/file.py",
            "content": "full file content here",
            "description": "what this file does"
        }
    ],
    "folders": ["folder1", "folder1/subfolder", "folder2"],
    "dependencies": ["package1", "package2"],
    "setup_instructions": "How to run the project"
}

Guidelines:
1. Create a complete, working project structure
2. Include all necessary files (main code, config, requirements.txt, README.md, .gitignore)
3. Write actual working code, not placeholders
4. Follow best practices for the chosen tech stack
5. Include proper error handling and comments
6. Make it production-ready

IMPORTANT: Return ONLY the JSON object, no markdown formatting."""

        user_prompt = f"""Create a complete project for:

DESCRIPTION: {description}

{f'PREFERRED TECH STACK: {tech_stack}' if tech_stack else 'Choose the most appropriate tech stack.'}

Generate all files with complete, working code. Make it production-ready."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # Parse JSON from response
            content = response.content.strip()
            
            # Try to extract JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            project_structure = json.loads(content)
            return project_structure
            
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            return None
        except Exception as e:
            print(f"Error generating project: {e}")
            return None
    
    def generate_file_content(self, file_path: str, project_context: str, file_purpose: str) -> str:
        """Generate content for a specific file"""
        
        prompt = f"""Generate the complete content for this file:

FILE: {file_path}
PURPOSE: {file_purpose}
PROJECT CONTEXT: {project_context}

Return ONLY the file content, no explanations or markdown formatting."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    
    def create_github_repo(self, project_name: str, description: str, private: bool = False) -> Optional[str]:
        """Create a new GitHub repository"""
        
        if not self.gh or not self.user:
            print("❌ GitHub not configured. Cannot create repository.")
            return None
        
        try:
            # Check if repo already exists
            try:
                existing = self.user.get_repo(project_name)
                print(f"⚠️ Repository '{project_name}' already exists!")
                return existing.html_url
            except GithubException:
                pass  # Repo doesn't exist, we can create it
            
            # Create new repo
            repo = self.user.create_repo(
                name=project_name,
                description=description,
                private=private,
                auto_init=False  # We'll push our own files
            )
            
            print(f"✅ Created repository: {repo.html_url}")
            return repo.html_url
            
        except GithubException as e:
            print(f"❌ Error creating repository: {e}")
            return None
    
    def push_project_to_github(self, project_structure: Dict, private: bool = False) -> Optional[str]:
        """Push generated project to GitHub"""
        
        if not self.gh or not self.user:
            print("❌ GitHub not configured. Cannot push project.")
            return None
        
        project_name = project_structure.get("project_name", "new-project")
        description = project_structure.get("description", "Generated by CodeSentinel")
        
        try:
            # Create the repository
            try:
                repo = self.user.get_repo(project_name)
                print(f"⚠️ Repository '{project_name}' already exists, using existing repo")
            except GithubException:
                repo = self.user.create_repo(
                    name=project_name,
                    description=description,
                    private=private,
                    auto_init=True  # Create with README so we have a branch
                )
                print(f"✅ Created repository: {repo.html_url}")
            
            # Get default branch
            default_branch = repo.default_branch
            
            # Create all files
            files = project_structure.get("files", [])
            for file_info in files:
                file_path = file_info.get("path", "")
                content = file_info.get("content", "")
                
                if not file_path or not content:
                    continue
                
                # Add CodeSentinel signature to the file
                content = self._add_signature(content, file_path)
                
                try:
                    # Check if file exists
                    try:
                        existing_file = repo.get_contents(file_path, ref=default_branch)
                        # Update existing file
                        repo.update_file(
                            path=file_path,
                            message=f"Update {file_path}",
                            content=content,
                            sha=existing_file.sha,
                            branch=default_branch
                        )
                        print(f"  📝 Updated: {file_path}")
                    except GithubException:
                        # Create new file
                        repo.create_file(
                            path=file_path,
                            message=f"Add {file_path}",
                            content=content,
                            branch=default_branch
                        )
                        print(f"  ✅ Created: {file_path}")
                        
                except Exception as e:
                    print(f"  ❌ Error with {file_path}: {e}")
            
            print(f"\n🎉 Project pushed to: {repo.html_url}")
            return repo.html_url
            
        except Exception as e:
            print(f"❌ Error pushing project: {e}")
            return None
    
    def build_project(self, description: str, tech_stack: str = None, 
                     push_to_github: bool = True, private: bool = False) -> Dict:
        """Main method: Build a complete project from description"""
        
        print("🏗️ Generating project structure...")
        project_structure = self.generate_project_structure(description, tech_stack)
        
        if not project_structure:
            return {"success": False, "error": "Failed to generate project structure"}
        
        # Add CodeSentinel signature to all generated files
        print("✍️ Adding CodeSentinel signature to files...")
        files = project_structure.get("files", [])
        for file_info in files:
            file_path = file_info.get("path", "")
            content = file_info.get("content", "")
            if file_path and content:
                file_info["content"] = self._add_signature(content, file_path)
        
        print(f"📁 Project: {project_structure.get('project_name', 'unknown')}")
        print(f"📦 Tech stack: {', '.join(project_structure.get('tech_stack', []))}")
        print(f"📄 Files: {len(project_structure.get('files', []))}")
        
        result = {
            "success": True,
            "project_name": project_structure.get("project_name"),
            "structure": project_structure,
            "files_count": len(project_structure.get("files", [])),
            "repo_url": None
        }
        
        if push_to_github:
            print("\n🚀 Pushing to GitHub...")
            repo_url = self.push_project_to_github(project_structure, private=private)
            result["repo_url"] = repo_url
        
        return result
    
    def enhance_existing_project(self, repo_name: str, enhancement_description: str) -> Dict:
        """Add features to an existing project"""
        
        if not self.gh:
            return {"success": False, "error": "GitHub not configured"}
        
        try:
            repo = self.gh.get_repo(repo_name)
            
            # Get current project structure
            current_files = []
            contents = repo.get_contents("")
            
            def get_all_files(contents, path=""):
                files = []
                for content in contents:
                    if content.type == "file" and content.path.endswith(('.py', '.js', '.ts', '.json', '.md', '.txt', '.yaml', '.yml')):
                        try:
                            file_content = content.decoded_content.decode()
                            files.append({
                                "path": content.path,
                                "content": file_content[:2000]  # Limit for context
                            })
                        except:
                            pass
                    elif content.type == "dir":
                        try:
                            sub_contents = repo.get_contents(content.path)
                            files.extend(get_all_files(sub_contents, content.path))
                        except:
                            pass
                return files
            
            current_files = get_all_files(contents)
            
            # Generate enhancement
            prompt = f"""Analyze this existing project and generate new/modified files for this enhancement:

CURRENT PROJECT FILES:
{json.dumps(current_files[:10], indent=2)}  # First 10 files for context

ENHANCEMENT REQUEST:
{enhancement_description}

Return a JSON object with files to add/modify:
{{
    "files": [
        {{"path": "path/to/file.py", "content": "full content", "action": "create" or "modify"}}
    ],
    "summary": "What was added/changed"
}}

Return ONLY JSON, no markdown."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            enhancement = json.loads(content)
            
            # Apply changes
            for file_info in enhancement.get("files", []):
                file_path = file_info.get("path")
                file_content = file_info.get("content")
                action = file_info.get("action", "create")
                
                # Add CodeSentinel signature to new/modified files
                if file_path and file_content:
                    file_content = self._add_signature(file_content, file_path)
                
                try:
                    if action == "modify":
                        existing = repo.get_contents(file_path)
                        repo.update_file(
                            path=file_path,
                            message=f"🤖 CodeSentinel: {enhancement_description[:50]}",
                            content=file_content,
                            sha=existing.sha
                        )
                        print(f"  📝 Modified: {file_path}")
                    else:
                        repo.create_file(
                            path=file_path,
                            message=f"🤖 CodeSentinel: {enhancement_description[:50]}",
                            content=file_content
                        )
                        print(f"  ✅ Created: {file_path}")
                except Exception as e:
                    print(f"  ❌ Error with {file_path}: {e}")
            
            return {
                "success": True,
                "summary": enhancement.get("summary", "Enhancement applied"),
                "files_changed": len(enhancement.get("files", []))
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# CLI for testing
if __name__ == "__main__":
    builder = ProjectBuilder()
    
    # Example: Create a new project
    result = builder.build_project(
        description="A simple REST API for managing a todo list with user authentication",
        tech_stack="Python, FastAPI, SQLite",
        push_to_github=False  # Set to True to actually create repo
    )
    
    print("\n" + "="*50)
    print("RESULT:")
    print(json.dumps(result, indent=2, default=str))
