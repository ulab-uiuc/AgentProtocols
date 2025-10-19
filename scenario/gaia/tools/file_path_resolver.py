"""
File Path Resolver for GAIA Tasks

Resolves file references in multimodal tasks using metadata from task definitions.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from .utils.logger import logger


class FilePathResolver:
    """Resolves file paths for GAIA multimodal tasks."""
    
    def __init__(self):
        self._task_metadata: Optional[Dict[str, Any]] = None
        self._dataset_dir: Optional[str] = None
        self._load_environment_config()
    
    def _load_environment_config(self):
        """Load configuration from environment variables."""
        self._dataset_dir = os.environ.get("GAIA_DATASET_DIR")
        
        # Try to load current task metadata
        task_id = os.environ.get("GAIA_TASK_ID")
        if task_id and self._dataset_dir:
            self._load_task_metadata(task_id)
    
    def _load_task_metadata(self, task_id: str):
        """Load task metadata from multimodal.jsonl if available."""
        if not self._dataset_dir:
            return
            
        multimodal_file = os.path.join(self._dataset_dir, "multimodal.jsonl")
        if not os.path.exists(multimodal_file):
            return
            
        try:
            with open(multimodal_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    task_data = json.loads(line)
                    if task_data.get("task_id") == task_id:
                        self._task_metadata = task_data
                        logger.info(f"📋 Loaded metadata for task {task_id}: file_name={task_data.get('file_name', 'N/A')}")
                        break
        except Exception as e:
            logger.warning(f"Failed to load task metadata for {task_id}: {e}")
    
    def get_actual_filename(self) -> Optional[str]:
        """Get the actual filename for the current task."""
        if self._task_metadata:
            return self._task_metadata.get("file_name")
        return None
    
    def resolve_file_path(self, requested_path: str) -> Optional[str]:
        """
        Resolve a file path, handling both direct requests and filename corrections.
        
        Args:
            requested_path: The path requested by the LLM/agent
            
        Returns:
            Actual file path if found, None otherwise
        """
        if not self._dataset_dir:
            logger.warning("No dataset directory available for file resolution")
            return None
            
        # If the requested path is absolute and exists, use it
        if os.path.isabs(requested_path) and os.path.exists(requested_path):
            return requested_path
        
        # Extract just the filename from the requested path
        requested_filename = os.path.basename(requested_path)
        
        # If we have task metadata, check if the requested filename matches
        actual_filename = self.get_actual_filename()
        if actual_filename:
            # Check if requested filename matches the actual filename
            actual_path = os.path.join(self._dataset_dir, actual_filename)
            if os.path.exists(actual_path):
                if requested_filename == actual_filename:
                    logger.info(f"✅ Direct match: {requested_filename} -> {actual_path}")
                    return actual_path
                else:
                    # LLM might be using wrong filename, correct it
                    logger.info(f"🔄 Correcting filename: {requested_filename} -> {actual_filename}")
                    return actual_path
        
        # Try to find the file in dataset directory by requested filename
        candidate_path = os.path.join(self._dataset_dir, requested_filename)
        if os.path.exists(candidate_path):
            logger.info(f"📁 Found in dataset: {requested_filename} -> {candidate_path}")
            return candidate_path
        
        # Look for files with similar names (case-insensitive)
        try:
            dataset_files = os.listdir(self._dataset_dir)
            for file in dataset_files:
                if file.lower() == requested_filename.lower():
                    found_path = os.path.join(self._dataset_dir, file)
                    logger.info(f"🔍 Case-insensitive match: {requested_filename} -> {found_path}")
                    return found_path
        except Exception as e:
            logger.warning(f"Error scanning dataset directory: {e}")
        
        logger.warning(f"❌ Could not resolve file: {requested_path}")
        return None
    
    def list_available_files(self) -> List[str]:
        """List all files available in the dataset directory."""
        if not self._dataset_dir or not os.path.exists(self._dataset_dir):
            return []
        
        try:
            return [f for f in os.listdir(self._dataset_dir) 
                   if os.path.isfile(os.path.join(self._dataset_dir, f))]
        except Exception:
            return []
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the current task."""
        info = {
            "task_id": os.environ.get("GAIA_TASK_ID"),
            "dataset_dir": self._dataset_dir,
            "has_metadata": self._task_metadata is not None,
            "actual_filename": self.get_actual_filename(),
            "available_files": self.list_available_files()
        }
        return info


# Global resolver instance
_resolver = FilePathResolver()

def get_file_path_resolver() -> FilePathResolver:
    """Get the global file path resolver instance."""
    return _resolver

def resolve_file_path(requested_path: str) -> Optional[str]:
    """Convenience function to resolve a file path."""
    return _resolver.resolve_file_path(requested_path)

def get_actual_filename() -> Optional[str]:
    """Convenience function to get the actual filename for current task."""
    return _resolver.get_actual_filename()
