from textwrap import dedent
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import json


# Define a byte stream where the first byte is a byte and the rest is a string
original_bytes = bytes([5]) + b'Hello, World!'

# Convert byte stream to string
converted_str = original_bytes.decode('utf-8', errors='ignore')

# Convert string back to bytes
converted_bytes = converted_str.encode('utf-8')

# Check if the converted bytes match the original bytes
is_equal = original_bytes == converted_bytes

# Log the result
print(f"Original bytes: {original_bytes}")
print(f"Converted string: {converted_str}")
print(f"Converted bytes: {converted_bytes}")
print(f"Do the original and converted bytes match? {is_equal}")


