
import os
from glob import glob
import json

if __name__ == '__main__':
    file_list = glob("nlp_outputs/*.txt")
    for file in file_list:
        print(f"Processing file: {file}")
        basename = os.path.basename(file)
        content = open(file, "r", encoding="utf-8").read().strip()
        if "</think>" in content:
            json_content = content.split("</think>")[-1].strip()
        else:
            json_content = content

        data = json.loads(json_content)

        outfile_name = f"nlp_output_json/{basename.replace('.txt', '')}.json"
        os.makedirs("nlp_output_json", exist_ok=True)
        with open(outfile_name, 'w', encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved extracted JSON to {outfile_name}")
    print("All done.")