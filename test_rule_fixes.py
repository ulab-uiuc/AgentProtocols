#!/usr/bin/env python3

"""
Test script for rule-based code fixing in SandboxPythonExecute.
"""

import asyncio
import sys
import os
import re

# Add gaia tools to path
sys.path.append('/root/Multiagent-Protocol/script/gaia/tools')

def test_rule_based_fixes():
    """Test the rule-based code fixing logic."""
    
    print("🧪 Testing Rule-Based Code Fixes\n")
    
    # Simulate the _apply_rule_based_fixes method logic
    def apply_rule_based_fixes(original_code: str, error_message: str):
        """Apply rule-based fixes for common Python errors."""
        fixed_code = original_code
        packages = []
        
        # Fix 1: Add None checks for AttributeError on NoneType
        if "AttributeError" in error_message and "NoneType" in error_message:
            lines = fixed_code.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Simple pattern: if line contains direct attribute access, add None check
                if '.' in line and '=' in line and not line.strip().startswith('#'):
                    # Look for pattern: variable = something.method()
                    attr_match = re.search(r'(\w+)\s*=\s*(\w+)\.(\w+)', line)
                    if attr_match:
                        var_name, obj_name, method_name = attr_match.groups()
                        # Add defensive check
                        indent = len(line) - len(line.lstrip())
                        indent_str = ' ' * indent
                        
                        fixed_lines.append(f"{indent_str}# Added defensive None check")
                        fixed_lines.append(f"{indent_str}if {obj_name} is not None:")
                        fixed_lines.append(f"{indent_str}    {line.strip()}")
                        fixed_lines.append(f"{indent_str}else:")
                        fixed_lines.append(f"{indent_str}    {var_name} = None  # Handle None case safely")
                        continue
                
                fixed_lines.append(line)
            
            if len(fixed_lines) > len(lines):
                fixed_code = '\n'.join(fixed_lines)
                print("🔧 Applied None-check fix for AttributeError")
        
        # Fix 2: Add missing colon for syntax errors
        if "SyntaxError" in error_message and ("expected ':'" in error_message or ":" in error_message):
            lines = fixed_code.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Remove inline comments for pattern matching
                line_without_comment = line.split('#')[0].rstrip()
                line_content = line_without_comment.strip()
                
                # Check for common missing colon patterns
                if re.match(r'^\s*(if|for|while|def|class|try|except|finally|with)\s+.*[^:]$', line_content):
                    # Add colon at the end of the actual content, preserve comments
                    if '#' in line:
                        # Line has comment, insert colon before comment
                        comment_part = line[line.index('#'):]
                        fixed_line = line_without_comment + ':  ' + comment_part
                    else:
                        # No comment, just add colon
                        fixed_line = line + ':'
                    fixed_lines.append(fixed_line)
                    print(f"🔧 Added missing colon to: {line_content}")
                else:
                    fixed_lines.append(line)
            
            fixed_code = '\n'.join(fixed_lines)
        
        return fixed_code, packages
    
    # Test 1: AttributeError fix
    print("=" * 50)
    print("Test 1: AttributeError on NoneType")
    print("=" * 50)
    
    code1 = """data = None
result = data.get('key')
print(result)"""
    
    error1 = "AttributeError: 'NoneType' object has no attribute 'get'"
    
    fixed_code1, packages1 = apply_rule_based_fixes(code1, error1)
    print("Original code:")
    print(code1)
    print("\nFixed code:")
    print(fixed_code1)
    print("\n")
    
    # Test 2: Syntax error fix
    print("=" * 50)
    print("Test 2: Missing colon syntax error")
    print("=" * 50)
    
    code2 = """for i in range(5)
    print(f"Number: {i}")"""
    
    error2 = "SyntaxError: invalid syntax. Perhaps you forgot a comma?"
    
    fixed_code2, packages2 = apply_rule_based_fixes(code2, error2)
    print("Original code:")
    print(code2)
    print("\nFixed code:")
    print(fixed_code2)
    print("\n")
    
    # Test 2b: Syntax error with comment
    print("=" * 50)
    print("Test 2b: Missing colon with comment")
    print("=" * 50)
    
    code2b = """for i in range(5)  # Missing colon
    print(f"Number: {i}")
if True  # Another missing colon
    print("This should work")"""
    
    error2b = "SyntaxError: expected ':'"
    
    fixed_code2b, packages2b = apply_rule_based_fixes(code2b, error2b)
    print("Original code:")
    print(code2b)
    print("\nFixed code:")
    print(fixed_code2b)
    print("\n")
    
    # Test 3: Package analysis
    print("=" * 50)
    print("Test 3: Package analysis for ModuleNotFoundError")
    print("=" * 50)
    
    def analyze_error_for_packages(error_message: str):
        """Analyze error message to suggest missing packages."""
        packages = []
        
        # Common package mappings for ModuleNotFoundError
        package_mappings = {
            'pandas': ['pandas'],
            'numpy': ['numpy'], 
            'matplotlib': ['matplotlib'],
            'seaborn': ['seaborn'],
            'sklearn': ['scikit-learn'],
            'cv2': ['opencv-python'],
            'PIL': ['pillow'],
            'openpyxl': ['openpyxl'],
            'xlrd': ['xlrd'],
            'requests': ['requests']
        }
        
        # Look for ModuleNotFoundError patterns
        module_not_found_pattern = r"No module named ['\"]([^'\"]+)['\"]"
        matches = re.findall(module_not_found_pattern, error_message, re.IGNORECASE)
        
        for module_name in matches:
            # Handle submodule cases (e.g., 'sklearn.metrics' -> 'sklearn')
            base_module = module_name.split('.')[0]
            
            if base_module in package_mappings:
                packages.extend(package_mappings[base_module])
            else:
                # Try the module name as package name
                packages.append(base_module)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(packages))
    
    error3 = "ModuleNotFoundError: No module named 'pandas'"
    packages3 = analyze_error_for_packages(error3)
    print(f"Error: {error3}")
    print(f"Suggested packages: {packages3}")
    
    error4 = "ModuleNotFoundError: No module named 'sklearn.metrics'"
    packages4 = analyze_error_for_packages(error4)
    print(f"Error: {error4}")
    print(f"Suggested packages: {packages4}")
    
    print("\n🎉 Rule-based fixing tests completed!")

if __name__ == "__main__":
    test_rule_based_fixes()
