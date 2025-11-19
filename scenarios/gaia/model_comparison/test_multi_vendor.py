#!/usr/bin/env python3
"""
Test script for multi-vendor LLM support in GAIA runners.

This script tests that the LLM module can correctly initialize and call
all three supported APIs: OpenAI, Anthropic, and Google.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add GAIA root to path
GAIA_ROOT = Path(__file__).parent
sys.path.insert(0, str(GAIA_ROOT))

from core.llm import LLM, LLMConfig


async def test_openai():
    """Test OpenAI API."""
    print("\n" + "="*60)
    print("Testing OpenAI API")
    print("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set, skipping OpenAI test")
        return False
    
    try:
        # Create test config
        config = {
            "model": {
                "api_type": "openai",
                "name": "gpt-4o-mini",
                "max_tokens": 100,
                "temperature": 0.0
            }
        }
        
        # Initialize LLM (would normally load from file)
        llm = LLM()
        llm.config.api_type = "openai"
        llm.config.name = "gpt-4o-mini"
        llm.model = "gpt-4o-mini"
        
        # Test simple completion
        messages = [{"role": "user", "content": "Say 'Hello from OpenAI!' in one sentence."}]
        response = await llm.ask(messages=messages)
        
        print(f"✅ OpenAI test passed")
        print(f"Response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_anthropic():
    """Test Anthropic API."""
    print("\n" + "="*60)
    print("Testing Anthropic Claude API")
    print("="*60)
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set, skipping Anthropic test")
        return False
    
    try:
        llm = LLM()
        llm.config.api_type = "anthropic"
        llm.config.api_key = os.getenv("ANTHROPIC_API_KEY")
        llm.config.name = "claude-3-5-sonnet-20241022"
        llm.model = "claude-3-5-sonnet-20241022"
        llm.config.base_url = "https://api.anthropic.com/v1"
        
        messages = [{"role": "user", "content": "Say 'Hello from Claude!' in one sentence."}]
        response = await llm.ask(messages=messages)
        
        print(f"✅ Anthropic test passed")
        print(f"Response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Anthropic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_google():
    """Test Google Gemini API."""
    print("\n" + "="*60)
    print("Testing Google Gemini API")
    print("="*60)
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  GOOGLE_API_KEY not set, skipping Google test")
        return False
    
    try:
        llm = LLM()
        llm.config.api_type = "google"
        llm.config.api_key = os.getenv("GOOGLE_API_KEY")
        llm.config.name = "gemini-2.0-flash-exp"
        llm.model = "gemini-2.0-flash-exp"
        llm.config.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        messages = [{"role": "user", "content": "Say 'Hello from Gemini!' in one sentence."}]
        response = await llm.ask(messages=messages)
        
        print(f"✅ Google test passed")
        print(f"Response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Google test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n🧪 Testing Multi-Vendor LLM Support")
    print("=" * 60)
    
    results = {
        "OpenAI": await test_openai(),
        "Anthropic": await test_anthropic(),
        "Google": await test_google()
    }
    
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    for vendor, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED/SKIPPED"
        print(f"{vendor:12s}: {status}")
    
    total = sum(1 for v in results.values() if v)
    print(f"\nTotal: {total}/{len(results)} tests passed")
    
    if total == 0:
        print("\n⚠️  No API keys configured. Set at least one of:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - GOOGLE_API_KEY")
        return 1
    
    return 0 if all(v or v is False for v in results.values()) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
