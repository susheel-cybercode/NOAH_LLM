#!/usr/bin/env python3
"""
MAYA AI Web Interface
"""

from flask import Flask, request, jsonify, render_template_string
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.maya import MayaAI

app = Flask(__name__)
maya_ai = MayaAI()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MAYA AI</title>
    <style>
        body { font-family: Arial; background: linear-gradient(135deg, #667eea, #764ba2); min-height: 100vh; margin: 0; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: white; text-align: center; }
        .chat { background: white; border-radius: 10px; padding: 20px; min-height: 400px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background: #667eea; color: white; text-align: right; }
        .ai { background: #f0f0f0; color: #333; }
        input { width: 80%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        button { padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MAYA AI</h1>
        <div class="chat" id="chat">
            <div class="message ai">Hello! I'm MAYA. How can I help you today?</div>
        </div>
        <div style="margin-top: 20px; display: flex; gap: 10px;">
            <input type="text" id="message" placeholder="Type your message..." onkeypress="if(event.key==='Enter') send()">
            <button onclick="send()">Send</button>
        </div>
    </div>
    <script>
        function send() {
            const msg = document.getElementById('message').value;
            if (!msg) return;
            
            const chat = document.getElementById('chat');
            chat.innerHTML += '<div class="message user">' + msg + '</div>';
            document.getElementById('message').value = '';
            
            fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: msg, user_id: 'web_user'})
            })
            .then(r => r.json())
            .then(data => {
                chat.innerHTML += '<div class="message ai">' + data.response + '</div>';
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    response = maya_ai.chat(data.get('message', ''), user_id=data.get('user_id', 'default'))
    return jsonify({'response': response['content']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
