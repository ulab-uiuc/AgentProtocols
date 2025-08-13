"""Custom exceptions for the tool system."""


class ToolError(Exception):
    """Exception raised when a tool encounters an error during execution."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
