#!/usr/bin/env python3

"""
Debug template formatting issues in prompt.py
"""

import sys
sys.path.append('/root/Multiagent-Protocol/script/gaia')

try:
    from core.prompt import PromptTemplates
    
    print("Testing TASK_ANALYSIS_USER template...")
    
    # Test TASK_ANALYSIS_USER
    test_document = "Test document content"
    try:
        result = PromptTemplates.TASK_ANALYSIS_USER.format(document=test_document)
        print("✓ TASK_ANALYSIS_USER works")
    except Exception as e:
        print(f"✗ TASK_ANALYSIS_USER error: {e}")
    
    # Test AGENT_SYSTEM_CONTEXT
    print("\nTesting AGENT_SYSTEM_CONTEXT template...")
    try:
        result = PromptTemplates.AGENT_SYSTEM_CONTEXT.format(
            original_query="test query",
            task_type="test",
            complexity="medium",
            domain_areas="test domains",
            total_agents=3,
            execution_pattern="sequential"
        )
        print("✓ AGENT_SYSTEM_CONTEXT works")
    except Exception as e:
        print(f"✗ AGENT_SYSTEM_CONTEXT error: {e}")
    
    # Test TOOL_USAGE_TEMPLATE
    print("\nTesting TOOL_USAGE_TEMPLATE template...")
    try:
        result = PromptTemplates.TOOL_USAGE_TEMPLATE.format(
            tool_name="test_tool",
            tool_description="test description"
        )
        print("✓ TOOL_USAGE_TEMPLATE works")
    except Exception as e:
        print(f"✗ TOOL_USAGE_TEMPLATE error: {e}")
    
    # Test WORKFLOW_CONTEXT_TEMPLATE
    print("\nTesting WORKFLOW_CONTEXT_TEMPLATE template...")
    try:
        result = PromptTemplates.WORKFLOW_CONTEXT_TEMPLATE.format(
            workflow_info="test workflow"
        )
        print("✓ WORKFLOW_CONTEXT_TEMPLATE works")
    except Exception as e:
        print(f"✗ WORKFLOW_CONTEXT_TEMPLATE error: {e}")
    
    # Test QUALITY_STANDARDS_TEMPLATE
    print("\nTesting QUALITY_STANDARDS_TEMPLATE template...")
    try:
        result = PromptTemplates.QUALITY_STANDARDS_TEMPLATE.format(
            max_tokens=1000
        )
        print("✓ QUALITY_STANDARDS_TEMPLATE works")
    except Exception as e:
        print(f"✗ QUALITY_STANDARDS_TEMPLATE error: {e}")
    
    # Test AGENT_PROMPT_TEMPLATE
    print("\nTesting AGENT_PROMPT_TEMPLATE template...")
    try:
        result = PromptTemplates.AGENT_PROMPT_TEMPLATE.format(
            system_context="test context",
            base_prompt="test prompt",
            tool_instructions="test instructions",
            workflow_context="test workflow",
            quality_standards="test standards"
        )
        print("✓ AGENT_PROMPT_TEMPLATE works")
    except Exception as e:
        print(f"✗ AGENT_PROMPT_TEMPLATE error: {e}")
    
    # Test build_agent_prompt method
    print("\nTesting build_agent_prompt method...")
    try:
        result = PromptTemplates.build_agent_prompt(
            role="test_role",
            agent_name="test_agent",
            task_type="test",
            complexity="medium",
            domain_areas=["test"],
            total_agents=1,
            execution_pattern="sequential",
            workflow_info="test workflow",
            original_query="test query",
            task_file_names=["test.txt"],
            tool_name="test_tool",
            tool_description="test description",
            max_tokens=1000
        )
        print("✓ build_agent_prompt works")
        print(f"Result length: {len(result)} characters")
    except Exception as e:
        print(f"✗ build_agent_prompt error: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
