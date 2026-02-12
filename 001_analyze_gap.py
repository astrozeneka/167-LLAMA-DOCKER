

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=False, help="Path to the article .txt file to process")
parser.add_argument('--use-remote', action='store_true', help="Use remote llama.cpp server instead of local model")
args = parser.parse_args()

API = "http://192.168.0.25:50001/v1/chat/completions"

article_content = open(args.input).read()
model_path = "models/Qwen3-8B-Q4_K_M.gguf"
prompt = (
    "You will read this article and identify what kind of \"database\" is lacking in the current (the time at which the article is written) lacks. What kind of public database would be helpful to the author throughout their studies. Answer a list of idea in a json format."
    "The JSON MUST follow the following format exactly (no extra text, no deviation):\n"
    "{\n"
    '  "missing_databases": [\n'
    '    {\n'
    '      "name": "Potential name of the missing database",\n'
    '      "description": "A brief description of the database and why it is needed"\n'
    '    },\n'
    '    {\n'
    '      "name": "Another missing database",\n'
    '      "description": "A brief description of this database and why it is needed"\n'
    '    }\n'
    '  ]\n'
    "}\n"
    "\n\nArticle:\n"
    f"{article_content}\n\n"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant. You will answer briefly without extra unnecessary verbosity."},
    {"role": "user", "content": prompt}
]

if __name__ == '__main__':
    print(f"Prompt: {prompt}")
    basename = os.path.basename(args.input) if args.input else "default_input"
    outfile_name = f"nlp_outputs/{basename}.txt"

    if args.use_remote:
        import requests
        print(f"Using remote API: {API}")
        response = requests.post(API, json={
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.7
        }).json()
    else:
        from llama_cpp import Llama
        print(f"Loading model: {model_path}")
        llm = Llama(
            model_path=model_path,
            #n_ctx=16384,
            #n_ctx=32768,
            n_ctx=35000,
            n_threads=8,
            verbose=False,
            chat_format="chatml"
        )
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=8192,
            temperature=0.7
        )

    result = response['choices'][0]['message']['content']
    print(f"Response: {result}")
    print("Done.")

    os.makedirs("nlp_outputs", exist_ok=True)
    with open(outfile_name, 'w') as f:
        f.write(response['choices'][0]['message']['content'])
    print(f"Saved output to {outfile_name}")
