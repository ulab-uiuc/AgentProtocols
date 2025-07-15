"""
Simplified ACP SDK integration test using the ACP Server directly.
This follows the exact pattern from your reference code.
"""

import asyncio
from collections.abc import AsyncGenerator

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, Server


# Example ACP agents using the exact SDK pattern from reference
async def echo_agent(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, None]:
    """Echoes everything"""
    for message in input:
        await asyncio.sleep(0.5)
        yield {"thought": "I should echo everything"}
        await asyncio.sleep(0.5)

        # Create proper message response
        echo_content = f"Echo: {message.parts[0].content}" if message.parts else "Echo: (no content)"
        echo_message = Message(
            role="assistant",
            parts=[MessagePart(content=echo_content)]
        )
        yield echo_message


async def smart_agent(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, None]:
    """Smart agent with intelligent responses"""
    for message in input:
        content = message.parts[0].content.lower() if message.parts else ""

        # Processing thought
        yield {"thought": "Analyzing the user's message..."}
        await asyncio.sleep(0.4)

        # Generate response based on content
        if "hello" in content or "hi" in content:
            response_content = "Hello! I'm an ACP agent. How can I help you today?"
        elif "compute" in content or "calculate" in content:
            yield {"thought": "This looks like a computation request"}
            await asyncio.sleep(0.3)
            response_content = "I can help with calculations! Please provide the specific math problem you'd like me to solve."
        elif "weather" in content:
            yield {"thought": "User is asking about weather"}
            await asyncio.sleep(0.2)
            response_content = "I don't have access to real-time weather data, but I'd be happy to help you find weather information!"
        elif "?" in content:
            yield {"thought": "This is a question, let me provide a helpful response"}
            await asyncio.sleep(0.5)
            response_content = "That's an interesting question! I'm an ACP agent designed to help with various tasks. What specifically would you like to know?"
        else:
            yield {"thought": "Processing general message"}
            await asyncio.sleep(0.3)
            response_content = f"I understand you said: '{message.parts[0].content if message.parts else '(no content)'}'. How can I assist you further?"

        # Final response
        response_message = Message(
            role="assistant",
            parts=[MessagePart(content=response_content)]
        )
        yield response_message


def test_echo_server():
    """Test using ACP Server directly - Echo Agent"""
    print("🚀 Starting ACP Echo Server (Port 8001)")

    server = Server()

    # Register the echo agent
    server.agent()(echo_agent)

    # Run the server
    try:
        server.run(host="127.0.0.1", port=8001)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")


def test_smart_server():
    """Test using ACP Server directly - Smart Agent"""
    print("🚀 Starting ACP Smart Server (Port 8002)")

    server = Server()

    # Register the smart agent
    server.agent()(smart_agent)

    # Run the server
    try:
        server.run(host="127.0.0.1", port=8002)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")


async def test_client_communication():
    """Test client communication with the ACP servers"""
    import httpx

    print("📞 Testing client communication...")

    async with httpx.AsyncClient() as client:
        # Test echo agent
        try:
            print("\n📨 Testing Echo Agent")
            response = await client.post(
                "http://127.0.0.1:8001/acp/message",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "parts": [{"content": "Hello Echo Agent!"}]
                        }
                    ]
                }
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Echo agent responded: {result}")
            else:
                print(f"❌ Echo agent error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"❌ Echo agent communication failed: {e}")

        # Test smart agent
        try:
            print("\n🧠 Testing Smart Agent")
            response = await client.post(
                "http://127.0.0.1:8002/acp/message",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "parts": [{"content": "Can you help me with a calculation?"}]
                        }
                    ]
                }
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Smart agent responded: {result}")
            else:
                print(f"❌ Smart agent error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"❌ Smart agent communication failed: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "echo":
            test_echo_server()
        elif sys.argv[1] == "smart":
            test_smart_server()
        elif sys.argv[1] == "client":
            asyncio.run(test_client_communication())
        else:
            print("Usage: python test_acp_simple.py [echo|smart|client]")
    else:
        print("🧪 ACP SDK Direct Integration Test")
        print("=" * 40)
        print()
        print("Available commands:")
        print("  python test_acp_simple.py echo   - Start echo server on port 8001")
        print("  python test_acp_simple.py smart  - Start smart server on port 8002")
        print("  python test_acp_simple.py client - Test client communication")
        print()
        print("💡 This uses the ACP SDK Server directly, exactly like your reference code!")
        print("   The servers follow the same pattern as your echo example.")
