import argparse
import os
import json
import csv
from llama_cpp import Llama

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help="Path to the JSON file to process")
args = parser.parse_args()

if __name__ == '__main__':
    model_path = "models/Qwen3-Embedding-8B-Q4_K_M.gguf"
    print(f"Loading embedding model: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False,
        embedding=True
    )

    file_path = args.input
    print(f"Processing file: {file_path}")
    data = open(file_path, "r", encoding="utf-8").read()
    data_json = json.loads(data)

    basename = os.path.basename(file_path).replace(".json", "")
    outfile_name = f"nlp_output_embedding/{basename}.csv"
    os.makedirs("nlp_output_embedding", exist_ok=True)

    rows = []
    for db_detail in data_json["missing_databases"]:
        db_txt = (
            f"Name: {db_detail['name']}\n"
            f"Description: {db_detail['description']}"
        )
        embedding = llm.embed(db_txt)
        rows.append({
            "name": db_detail["name"],
            "description": db_detail["description"],
            "embedding": embedding
        })
        print(f"  Embedded: {db_detail['name']}")

    with open(outfile_name, 'w', newline='', encoding='utf-8') as f:
        dim = len(rows[0]["embedding"]) if rows else 0
        fieldnames = ["name", "description"] + [f"dim_{i}" for i in range(dim)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {"name": row["name"], "description": row["description"]}
            for i, val in enumerate(row["embedding"]):
                csv_row[f"dim_{i}"] = val
            writer.writerow(csv_row)

    print(f"Saved {len(rows)} embeddings to {outfile_name}")
