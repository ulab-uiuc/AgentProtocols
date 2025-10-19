"""Custom exceptions for the tool system."""


class ToolError(Exception):
    """Exception raised when a tool encounters an error during execution."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class OpenManusError(Exception):
    """Base exception for all OpenManus errors"""
    
class TokenLimitExceeded(OpenManusError):
    """Exception raised when the token limit is exceeded"""