"""
Codebase Indexer: Creates embeddings and understands relationships between files
"""
import os
import re
import ast
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
import numpy as np
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available. Install with: pip install faiss-cpu (or faiss-gpu)")


class DependencyAnalyzer:
    """Analyzes import dependencies between files"""
    
    def __init__(self):
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)  # file -> set of files it imports
        self.dependents: Dict[str, Set[str]] = defaultdict(set)  # file -> set of files that import it
        self.file_to_module: Dict[str, str] = {}  # file path -> module name
        
    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to module name"""
        # Remove .py extension and convert / to .
        module = file_path.replace('.py', '').replace('/', '.')
        # Remove leading dots
        module = module.lstrip('.')
        return module
    
    def _resolve_import(self, import_name: str, current_file: str, all_files: List[str]) -> Optional[str]:
        """Resolve an import name to an actual file path"""
        # Handle relative imports
        if import_name.startswith('.'):
            current_dir = os.path.dirname(current_file)
            parts = import_name.lstrip('.').split('.')
            # Try to resolve relative import
            for part in parts:
                potential_path = os.path.join(current_dir, part + '.py')
                if potential_path in all_files:
                    return potential_path
                potential_init = os.path.join(current_dir, part, '__init__.py')
                if potential_init in all_files:
                    return potential_init
            return None
        
        # Handle absolute imports
        # Try direct match first
        for file_path in all_files:
            module_name = self._path_to_module(file_path)
            if module_name == import_name or module_name.endswith('.' + import_name):
                return file_path
        
        # Try matching by filename
        import_base = import_name.split('.')[0]
        for file_path in all_files:
            filename = os.path.basename(file_path).replace('.py', '')
            if filename == import_base:
                return file_path
        
        return None
    
    def analyze_file(self, file_path: str, content: str, all_files: List[str]):
        """Analyze a file and extract its dependencies"""
        self.file_to_module[file_path] = self._path_to_module(file_path)
        
        try:
            tree = ast.parse(content)
            imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_name = alias.name.split('.')[0]  # Get base module
                        imports.add(import_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        import_name = node.module.split('.')[0]  # Get base module
                        imports.add(import_name)
            
            # Resolve imports to actual files
            for import_name in imports:
                resolved = self._resolve_import(import_name, file_path, all_files)
                if resolved and resolved != file_path:
                    self.dependencies[file_path].add(resolved)
                    self.dependents[resolved].add(file_path)
                    
        except SyntaxError:
            # File has syntax errors, skip dependency analysis
            pass
    
    def get_dependencies(self, file_path: str) -> Set[str]:
        """Get all files that this file depends on"""
        return self.dependencies.get(file_path, set())
    
    def get_dependents(self, file_path: str) -> Set[str]:
        """Get all files that depend on this file"""
        return self.dependents.get(file_path, set())
    
    def get_related_files(self, file_path: str, max_depth: int = 2) -> Set[str]:
        """Get all related files (dependencies and dependents) up to max_depth"""
        related = set()
        to_explore = [(file_path, 0)]
        explored = set()
        
        while to_explore:
            current, depth = to_explore.pop(0)
            if current in explored or depth > max_depth:
                continue
            explored.add(current)
            
            # Add dependencies
            for dep in self.get_dependencies(current):
                if dep not in explored:
                    related.add(dep)
                    to_explore.append((dep, depth + 1))
            
            # Add dependents
            for dep in self.get_dependents(current):
                if dep not in explored:
                    related.add(dep)
                    to_explore.append((dep, depth + 1))
        
        return related


class CodebaseIndexer:
    """Indexes the entire codebase using embeddings for semantic understanding"""
    
    def __init__(self, repo, embeddings: Optional[Embeddings] = None):
        self.repo = repo
        api_key = os.getenv("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        try:
            self.embeddings = embeddings or OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key
            )
            self.embeddings_enabled = True
        except Exception as e:
            print(f"⚠️ Embeddings disabled: {e}")
            self.embeddings = None
            self.embeddings_enabled = False
        self.file_embeddings: Dict[str, np.ndarray] = {}
        self.file_contents: Dict[str, str] = {}
        self.file_metadata: Dict[str, Dict] = {}
        self.dependency_analyzer = DependencyAnalyzer()
        # FAISS index for efficient similarity search
        self.faiss_index = None
        self.file_paths_list: List[str] = []  # Maps FAISS index to file path
        self.embedding_dim = None
        
    def _extract_code_features(self, content: str) -> str:
        """Extract meaningful features from code for embedding"""
        # Extract: imports, classes, functions, docstrings
        features = []
        
        try:
            tree = ast.parse(content)
            
            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            if imports:
                features.append(f"Imports: {', '.join(imports)}")
            
            # Extract classes and their methods
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    features.append(f"Class {node.name} with methods: {', '.join(methods)}")
            
            # Extract functions
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not any(
                    isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) 
                    if hasattr(parent, 'body') and node in parent.body
                ):
                    functions.append(node.name)
            if functions:
                features.append(f"Functions: {', '.join(functions)}")
            
            # Extract docstrings
            docstrings = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        docstrings.append(docstring[:200])  # First 200 chars
            if docstrings:
                features.append(f"Documentation: {' '.join(docstrings)}")
                
        except SyntaxError:
            # If parsing fails, use raw content (limited)
            features.append(content[:1000])
        
        return "\n".join(features) if features else content[:1000]
    
    def index_file(self, file_path: str, content: str, all_files: List[str]):
        """Index a single file"""
        self.file_contents[file_path] = content
        
        # Extract features for embedding
        features = self._extract_code_features(content)
        
        # Create embedding if enabled
        if self.embeddings_enabled and self.embeddings:
            try:
                embedding = self.embeddings.embed_query(features)
                self.file_embeddings[file_path] = np.array(embedding)
            except Exception as e:
                print(f"⚠️ Warning: Could not create embedding for {file_path}: {e}")
                # Use zero vector as fallback
                self.file_embeddings[file_path] = np.zeros(1536)  # Default embedding size
        else:
            # Embeddings disabled, use simple hash-based similarity instead
            # Store a simple representation for similarity matching
            self.file_embeddings[file_path] = None
        
        # Analyze dependencies
        self.dependency_analyzer.analyze_file(file_path, content, all_files)
        
        # Store metadata
        self.file_metadata[file_path] = {
            "size": len(content),
            "lines": len(content.splitlines()),
            "has_errors": False
        }
    
    def index_codebase(self, file_paths: List[str]) -> Dict[str, any]:
        """Index the entire codebase"""
        print(f"📚 Indexing {len(file_paths)} files...")
        
        # First pass: collect all file contents
        all_contents = {}
        for file_path in file_paths:
            try:
                content = self.repo.get_contents(file_path).decoded_content.decode()
                all_contents[file_path] = content
            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}")
                continue
        
        # Second pass: index with dependency analysis
        for file_path, content in all_contents.items():
            self.index_file(file_path, content, file_paths)
        
        # Build FAISS index for efficient similarity search
        self._build_faiss_index()
        
        print(f"✅ Indexed {len(self.file_embeddings)} files")
        return {
            "indexed_files": len(self.file_embeddings),
            "dependencies": dict(self.dependency_analyzer.dependencies),
            "dependents": dict(self.dependency_analyzer.dependents)
        }
    
    def _build_faiss_index(self):
        """Build FAISS index from all file embeddings"""
        if not FAISS_AVAILABLE or not self.embeddings_enabled:
            return
        
        # Collect valid embeddings
        valid_embeddings = []
        self.file_paths_list = []
        
        for file_path, embedding in self.file_embeddings.items():
            if embedding is not None:
                valid_embeddings.append(embedding)
                self.file_paths_list.append(file_path)
        
        if not valid_embeddings:
            return
        
        # Determine embedding dimension
        self.embedding_dim = len(valid_embeddings[0])
        
        # Convert to numpy array
        embeddings_matrix = np.array(valid_embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity (FAISS uses L2 distance, so we normalize for cosine)
        faiss.normalize_L2(embeddings_matrix)
        
        # Create FAISS index (using Inner Product for cosine similarity after normalization)
        # For cosine similarity: normalize vectors, then use Inner Product
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings_matrix)
        
        print(f"🔍 Built FAISS index with {len(self.file_paths_list)} embeddings (dim={self.embedding_dim})")
    
    def find_similar_files(self, file_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find files semantically similar to the given file using FAISS"""
        if file_path not in self.file_embeddings or not self.embeddings_enabled:
            # Fallback: use dependency-based similarity
            related = self.dependency_analyzer.get_related_files(file_path, max_depth=2)
            return [(f, 0.5) for f in list(related)[:top_k]]
        
        target_embedding = self.file_embeddings[file_path]
        if target_embedding is None:
            # Embeddings disabled, use dependency-based similarity
            related = self.dependency_analyzer.get_related_files(file_path, max_depth=2)
            return [(f, 0.5) for f in list(related)[:top_k]]
        
        # Use FAISS if available, otherwise fallback to brute force
        if FAISS_AVAILABLE and self.faiss_index is not None and len(self.file_paths_list) > 0:
            try:
                # Prepare query vector (normalize for cosine similarity)
                query_vector = target_embedding.astype('float32').reshape(1, -1)
                faiss.normalize_L2(query_vector)
                
                # Search for top_k + 1 (to exclude the file itself if it's in the index)
                k = min(top_k + 1, len(self.file_paths_list))
                distances, indices = self.faiss_index.search(query_vector, k)
                
                # Build results, excluding the file itself
                results = []
                for dist, idx in zip(distances[0], indices[0]):
                    if idx >= 0 and idx < len(self.file_paths_list):  # Valid index
                        similar_file = self.file_paths_list[idx]
                        if similar_file != file_path:  # Exclude self
                            results.append((similar_file, float(dist)))
                            if len(results) >= top_k:
                                break
                
                if results:
                    return results
                # If no results (e.g., file not in index), fall through to brute force
            except Exception as e:
                # Fallback if FAISS search fails
                print(f"⚠️ FAISS search failed, using fallback: {e}")
        
        # Fallback: brute force search (for small codebases or if FAISS unavailable)
        similarities = []
        target_normalized = target_embedding / np.linalg.norm(target_embedding)
        
        for other_path, other_embedding in self.file_embeddings.items():
            if other_path == file_path or other_embedding is None:
                continue
            
            try:
                other_normalized = other_embedding / np.linalg.norm(other_embedding)
                similarity = np.dot(target_normalized, other_normalized)
                similarities.append((other_path, float(similarity)))
            except Exception:
                continue
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_context_for_file(self, file_path: str, max_files: int = 5) -> str:
        """Get relevant context for a file (dependencies + similar files)"""
        context_parts = []
        
        # Get dependency-related files
        dependencies = self.dependency_analyzer.get_dependencies(file_path)
        dependents = self.dependency_analyzer.get_dependents(file_path)
        related = self.dependency_analyzer.get_related_files(file_path, max_depth=1)
        
        # Get semantically similar files
        similar = self.find_similar_files(file_path, top_k=3)
        
        # Combine and prioritize
        context_files = set()
        
        # Priority 1: Direct dependencies and dependents
        context_files.update(list(dependencies)[:2])
        context_files.update(list(dependents)[:2])
        
        # Priority 2: Semantically similar files
        for similar_path, _ in similar[:2]:
            if similar_path not in context_files:
                context_files.add(similar_path)
        
        # Priority 3: Related files (broader context)
        remaining = max_files - len(context_files)
        if remaining > 0:
            for related_file in related:
                if related_file not in context_files and len(context_files) < max_files:
                    context_files.add(related_file)
        
        # Build context string
        for ctx_file in context_files:
            if ctx_file in self.file_contents:
                content = self.file_contents[ctx_file]
                # Truncate very long files
                if len(content) > 2000:
                    content = content[:2000] + "\n... (truncated)"
                context_parts.append(f"\n\n--- {ctx_file} ---\n{content}")
        
        return "\n".join(context_parts)
    
    def analyze_impact(self, file_path: str, proposed_change: str) -> Dict[str, any]:
        """Analyze potential impact of changing a file on other files"""
        dependents = self.dependency_analyzer.get_dependents(file_path)
        
        # Extract what might have changed (simplified - could be enhanced with AST diff)
        impact_info = {
            "files_that_import_this": list(dependents),
            "potential_breaking_changes": [],
            "recommendation": ""
        }
        
        if dependents:
            impact_info["recommendation"] = (
                f"⚠️ Warning: {len(dependents)} file(s) import this file. "
                f"Changes might affect: {', '.join(list(dependents)[:3])}"
            )
        else:
            impact_info["recommendation"] = "✅ No other files depend on this file. Safe to modify."
        
        return impact_info
