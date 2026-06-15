#!/usr/bin/env python3
"""
MAYA AI - Main Entry Point
The Ultimate AI Assistant
"""

import sys
import os
from typing import Optional

# Add modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.maya import MayaAI
from modules.coding import MayaCoding, Language
from modules.multimodal import MultimodalProcessor
from modules.tools import MayaTools
from modules.memory import MemoryStore

class MayaApp:
    """MAYA AI Application"""
    
    def __init__(self):
        print("=" * 60)
        print("  MAYA AI - The Ultimate AI Assistant")
        print("  Version 1.0.0")
        print("  Inspired by: Claude, GPT-4, Gemini, and all modern LLMs")
        print("=" * 60)
        print()
        
        # Initialize core systems
        self.ai = MayaAI()
        self.coding = MayaCoding()
        self.multimodal = MultimodalProcessor()
        self.tools = MayaTools()
        self.memory = MemoryStore()
        
        self.user_id = "default_user"
    
    def run_cli(self):
        """Run interactive CLI"""
        print("Welcome! I'm MAYA, your AI assistant.")
        print("I can help you with coding, analysis, creativity, and more.")
        print("Type 'help' for commands, 'exit' to quit.")
        print("-" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Exit
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nMAYA: Goodbye! Have a great day!")
                    break
                
                # Help
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                # Stats
                if user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                
                # Process message
                response = self.ai.chat(user_input, user_id=self.user_id)
                
                # Display response
                print(f"\nMAYA: {response['content']}")
                
                # Show reasoning if available
                if 'reasoning' in response and response['reasoning']:
                    print(f"\n[Analysis complete]")
                
            except KeyboardInterrupt:
                print("\nMAYA: Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    def _show_help(self):
        """Display help information"""
        print("\n" + "=" * 60)
        print("  MAYA AI Commands")
        print("=" * 60)
        print("\nGeneral:")
        print("  help       - Show this help menu")
        print("  stats      - Show system stats")
        print("  exit/quit  - Exit the application")
        print("\nCapabilities:")
        print("  - Natural language conversation")
        print("  - Code generation and debugging")
        print("  - Image analysis and generation")
        print("  - Document processing")
        print("  - Web search and research")
        print("  - Mathematical calculations")
        print("  - Task planning and automation")
        print("  - Long-term memory and personalization")
        print("=" * 60)
    
    def _show_stats(self):
        """Display system statistics"""
        stats = self.ai.get_stats()
        print("\n" + "=" * 60)
        print("  MAYA AI System Statistics")
        print("=" * 60)
        print(f"Version: {stats['version']}")
        print(f"Conversations: {stats['conversations']}")
        print(f"Projects: {stats['projects']}")
        print(f"Artifacts: {stats['artifacts']}")
        print(f"Memories: {stats['memories']}")
        print("\nCapabilities:")
        for cap, enabled in stats['capabilities'].items():
            status = "✓" if enabled else "✗"
            print(f"  {status} {cap}")
        print("=" * 60)

def main():
    """Main entry point"""
    app = MayaApp()
    app.run_cli()

if __name__ == "__main__":
    main()
