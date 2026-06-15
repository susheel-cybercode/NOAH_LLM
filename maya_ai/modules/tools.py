#!/usr/bin/env python3
"""
MAYA AI Tools Module
Web search, file operations, API integrations
Like: GPT-4 Plugins, Claude Tools, Perplexity
"""

import json
from typing import Dict, List, Optional, Any
from urllib.request import urlopen, Request
from urllib.error import URLError
from pathlib import Path

class WebSearch:
    """Web search capabilities"""
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search the web for information"""
        return [{
            'title': f'Result for: {query}',
            'url': 'https://example.com',
            'snippet': 'Search result summary'
        }]
    
    def fetch_page(self, url: str) -> str:
        """Fetch and extract content from a URL"""
        try:
            req = Request(url, headers={'User-Agent': 'MAYA AI/1.0'})
            with urlopen(req, timeout=10) as response:
                return response.read().decode('utf-8')
        except Exception as e:
            return f"Error fetching page: {str(e)}"

class FileManager:
    """File operations"""
    
    def read_file(self, path: str) -> str:
        """Read file contents"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def write_file(self, path: str, content: str) -> bool:
        """Write content to file"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception:
            return False
    
    def list_directory(self, path: str = '.') -> List[Dict]:
        """List directory contents"""
        try:
            items = []
            for item in Path(path).iterdir():
                items.append({
                    'name': item.name,
                    'type': 'directory' if item.is_dir() else 'file',
                    'size': item.stat().st_size if item.is_file() else 0
                })
            return items
        except Exception:
            return []

class Calculator:
    """Mathematical calculations"""
    
    def calculate(self, expression: str) -> Dict:
        """Safely evaluate mathematical expression"""
        try:
            allowed_names = {
                'abs': abs, 'max': max, 'min': min,
                'sum': sum, 'len': len, 'round': round
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return {'result': result, 'error': None}
        except Exception as e:
            return {'result': None, 'error': str(e)}

class MayaTools:
    """Combined tools interface"""
    
    def __init__(self):
        self.web_search = WebSearch()
        self.file_manager = FileManager()
        self.calculator = Calculator()
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name"""
        tools = {
            'search': lambda q: self.web_search.search(q),
            'fetch': lambda u: self.web_search.fetch_page(u),
            'read_file': lambda p: self.file_manager.read_file(p),
            'write_file': lambda p, c: self.file_manager.write_file(p, c),
            'calculate': lambda e: self.calculator.calculate(e)
        }
        
        if tool_name in tools:
            return tools[tool_name](**kwargs)
        return {"error": "Tool not found"}