#!/usr/bin/env python3
"""
CLI Client for Anigma F1 AI Agent
Simple interface to chat with the FastAPI server
"""

import requests
import json
from typing import Dict, Any

# Default server URL
DEFAULT_URL = "http://localhost:8000"


class AgentClient:
    def __init__(self, base_url: str = DEFAULT_URL, user_id: str = "cli_user"):
        self.base_url = base_url.rstrip('/')
        self.user_id = user_id
        self.session = requests.Session()

    def ask(self, text: str) -> Dict[str, Any]:
        """Send question to agent"""
        try:
            response = self.session.post(
                f"{self.base_url}/ask",
                json={"user_id": self.user_id, "text": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to agent server. Is it running?"}
        except requests.exceptions.Timeout:
            return {"error": "Request timed out"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}

    def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Health check failed: {str(e)}"}

    def add_memory(
            self, text: str,
            metadata: Dict[str, Any] = None
            ) -> Dict[str, Any]:
        """Add memory manually"""
        try:
            response = self.session.post(
                f"{self.base_url}/memory/add",
                json={"text": text, "metadata": metadata or {}},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Memory add failed: {str(e)}"}


def print_response(result: Dict[str, Any]):
    """Pretty print agent response"""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"🤖 {result.get('reply', 'No response')}")

    # Show metadata
    model = result.get('model_used', 'unknown')
    time_taken = result.get('processing_time', 0)
    context_items = result.get('context_items', 0)

    print(f"   └─ Model: {model} | Time: {time_taken}s"
          f" | Context:"
          f" {context_items} items")

    # Show saved memories
    if result.get('memory_saved'):
        print(f"   └─ Saved to memory: {', '.join(result['memory_saved'])}")


def interactive_mode(client: AgentClient):
    """Interactive chat mode"""
    print("🚀 Anigma F1 AI Agent - Interactive Mode")
    print("Type 'quit', 'exit', or 'q' to exit")
    print("Type 'health' to check server status")
    print("Type 'remember: <text>' to save something to memory")
    print("Type 'use_remote:true <question>' to force remote model")
    print("-" * 50)

    # Health check first
    health = client.health_check()
    if "error" in health:
        print(f"⚠️  Warning: {health['error']}")
    else:
        models = health.get('models_available', {})
        local_status = "✅" if models.get('local') else "❌"
        remote_status = "✅" if models.get('remote') else "❌"
        print(f"Server Status: {health.get('status', 'unknown')}")
        print(f"Local Model: {local_status} | Remote Model: {remote_status}")
        if not models.get('remote'):
            print("💡 Tip: Set HF_TOKEN environment"
                  " variable for remote model access")

    print("-" * 50)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if user_input.lower() == 'health':
                health = client.health_check()
                print(f"🏥 Health: {json.dumps(health, indent=2)}")
                continue

            if not user_input:
                continue

            # Send to agent
            print("🤔 Thinking...")
            result = client.ask(user_input)
            print_response(result)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


def single_question_mode(client: AgentClient, question: str):
    """Ask single question and exit"""
    result = client.ask(question)
    print_response(result)


def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Anigma F1 AI Agent CLI Client")
    parser.add_argument("--url", default=DEFAULT_URL, help="Agent server URL")
    parser.add_argument("--user-id",
                        default="cli_user",
                        help="User ID for memory separation")
    parser.add_argument("--question",
                        "-q",
                        help="Ask single question and exit")
    parser.add_argument("--health",
                        action="store_true", help="Check health and exit")

    args = parser.parse_args()

    # Initialize client
    client = AgentClient(args.url, args.user_id)

    # Health check mode
    if args.health:
        health = client.health_check()
        print(json.dumps(health, indent=2))
        return

    # Single question mode
    if args.question:
        single_question_mode(client, args.question)
        return

    # Interactive mode
    interactive_mode(client)


if __name__ == "__main__":
    main()
