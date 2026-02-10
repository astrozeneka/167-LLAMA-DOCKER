

import sys
from llama_cpp import Llama

article_content = open("articles/32868913-Hartmann et al. 2020.xml").read()
model_path = "models/Qwen3-8B-Q4_K_M.gguf"
prompt = (
    "You will read this article and identify what kind of \"database\" is lacking in the current (the time at which the article is written) lacks. What kind of public database would be helpful to the author throughout their studies. Answer a list of idea in a json format."
    "\n\nArticle:\n"
    f"{article_content}\n\n"
)

print(f"Loading model: {model_path}")
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=8,
    verbose=False,
    chat_format="chatml"  # Qwen uses ChatML format
)

if __name__ == '__main__':
    print(f"Prompt: {prompt}")

    # Use create_chat_completion instead of __call__
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You will answer briefly without extra unnecessary verbosity."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
        temperature=0.7
    )

    print(f"Response: {response['choices'][0]['message']['content']}")
    print("Done.")