import argparse
import os
import json
from llama_cpp import Llama
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=False, help="Path to the article .txt file to process")
args = parser.parse_args()



if __name__ == '__main__':
    file_path = args.input
    print("Processing file:", file_path)
    data = open(file_path, "r", encoding="utf-8").read()
    data_json = json.loads(data)

    for db_detail in data_json["missing_databases"]:
        db_txt = (
            f"Name: {db_detail["name"]}\n"
            f"Description: {db_detail["description"]}\n"
        )
        llm = Llama(
            model_path="models/Qwen3-Embedding-8B-Q4_K_M.gguf",
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
        print()