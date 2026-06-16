#!/usr/bin/env python3
"""
NOAH API Server - OpenAI-compatible API for Apple of Eden integration.
Run: python noah_api.py
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn

from noah_model import NoahLLM, NoahConfig, NOAH_CONFIGS, NoahTokenizer


app = FastAPI(title="NOAH LLM API", version="1.0.0")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "noah"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.95
    top_k: int = 50


class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatResponse(BaseModel):
    choices: List[ChatChoice]


# Global model and tokenizer
model = None
tokenizer = None
device = None


def load_model(model_size: str = "small"):
    """Load model and tokenizer at startup."""
    global model, tokenizer, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_path = Path("training_data") / "noah_tokenizer.json"
    if not tokenizer_path.exists():
        raise RuntimeError("Tokenizer not found. Run: python noah_tokenizer.py")

    tokenizer = NoahTokenizer(str(tokenizer_path))

    config = NOAH_CONFIGS[model_size]
    config.vocab_size = tokenizer.vocab_size

    model = NoahLLM(config).to(device)

    model_path = f"noah_{model_size}.pt"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained model: {model_path}")
    else:
        print("WARNING: No trained model found. Using random weights.")

    model.eval()
    print(f"NOAH API ready on {device}")
    print(f"Model: NOAH-{model_size.upper()} | Vocab: {tokenizer.vocab_size} | Params: {sum(p.numel() for p in model.parameters()):,}")


@app.on_event("startup")
async def startup():
    model_size = os.getenv("NOAH_MODEL_SIZE", "small")
    load_model(model_size)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "NOAH", "device": str(device)}


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert messages to prompt
    prompt_parts = []
    for msg in req.messages:
        prompt_parts.append(f"{msg.role}: {msg.content}")
    prompt = "\n".join(prompt_parts) + "\nassistant:"

    input_ids = torch.tensor([[tokenizer.cls_token_id] + tokenizer.encode(prompt)], device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p
        )

    response_ids = output_ids[0][len(input_ids[0]):].tolist()
    response_text = tokenizer.decode(response_ids)

    return ChatResponse(choices=[ChatChoice(
        index=0,
        message=Message(role="assistant", content=response_text),
        finish_reason="stop"
    )])


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "noah-small", "object": "model", "owned_by": "noah"},
            {"id": "noah-medium", "object": "model", "owned_by": "noah"},
            {"id": "noah-large", "object": "model", "owned_by": "noah"},
        ]
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="NOAH API Server")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Pre-load for validation
    tokenizer_path = Path("training_data") / "noah_tokenizer.json"
    if not tokenizer_path.exists():
        print("ERROR: Tokenizer not found. Run: python noah_tokenizer.py")
        sys.exit(1)

    print(f"Starting NOAH API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)