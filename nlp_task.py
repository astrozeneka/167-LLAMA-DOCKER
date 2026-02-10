#!/usr/bin/env python3
import sys
from llama_cpp import Llama

model_path = sys.argv[1] if len(sys.argv) > 1 else "/app/models/model.gguf"
prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello, how are you?"

print(f"Loading model: {model_path}")
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=8,
    verbose=False,
    chat_format="chatml"  # Qwen uses ChatML format
)

print(f"Prompt: {prompt}")

# Use create_chat_completion instead of __call__
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant. You will answer briefly without extra unnecessary verbosity."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=1024,
    temperature=0.7
)

print(f"Response: {response['choices'][0]['message']['content']}")
