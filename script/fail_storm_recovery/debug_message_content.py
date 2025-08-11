#!/usr/bin/env python3
"""
Debug script to understand why message content is still empty.
"""

import json

# Test the _extract_text_content method with different payloads
def test_extract_text_content(payload):
    """Test content extraction logic"""
    print(f"Testing payload: {json.dumps(payload, indent=2)}")
    
    if isinstance(payload, dict):
        # 处理Gaia文档广播
        if payload.get("type") == "gaia_document_init" and "document" in payload:
            document = payload["document"]
            print(f"Found Gaia document: {type(document)}")
            if isinstance(document, dict):
                # 提取文档的主要内容
                content_parts = []
                if "title" in document:
                    content_parts.append(f"Title: {document['title']}")
                if "content" in document:
                    content_parts.append(f"Content: {document['content']}")
                if "question" in document:
                    content_parts.append(f"Question: {document['question']}")
                result = "\n".join(content_parts) if content_parts else json.dumps(document)
                print(f"Extracted content: {result}")
                return result
            else:
                result = str(document)
                print(f"Document as string: {result}")
                return result
        
        # 尝试多种可能的文本字段
        for field in ["text", "content", "message", "query", "input"]:
            if field in payload:
                result = str(payload[field])
                print(f"Found field '{field}': {result}")
                return result
        # 如果没有找到特定字段，返回整个负载的字符串表示
        result = json.dumps(payload)
        print(f"Fallback to JSON dump: {result}")
        return result
    else:
        result = str(payload)
        print(f"Non-dict payload: {result}")
        return result

if __name__ == "__main__":
    print("🔍 Testing message content extraction...")
    
    # Test 1: Simulate Gaia document broadcast payload
    print("\n=== Test 1: Gaia Document Broadcast ===")
    gaia_payload = {
        "type": "gaia_document_init",
        "sender": "gaia_agent_0",
        "timestamp": 1754877038.7818298,
        "document": {
            "title": "Sample Gaia Document",
            "content": "This is the main content of the document",
            "question": "What is the meaning of life?"
        },
        "broadcast_id": "gaia_init_1754877038781"
    }
    test_extract_text_content(gaia_payload)
    
    # Test 2: Empty document
    print("\n=== Test 2: Empty Document ===")
    empty_doc_payload = {
        "type": "gaia_document_init",
        "sender": "gaia_agent_0",
        "timestamp": 1754877038.7818298,
        "document": {},
        "broadcast_id": "gaia_init_1754877038781"
    }
    test_extract_text_content(empty_doc_payload)
    
    # Test 3: Regular message
    print("\n=== Test 3: Regular Message ===")
    regular_payload = {
        "text": "Hello, this is a regular message"
    }
    test_extract_text_content(regular_payload)
    
    # Test 4: Completely empty payload
    print("\n=== Test 4: Empty Payload ===")
    empty_payload = {}
    test_extract_text_content(empty_payload)