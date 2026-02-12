from glob import glob
import json

if __name__ == '__main__':
    json_list = glob("nlp_output_json/*.json")
    count = 0
    for path in json_list:
        print(f"Processing file: {path}")
        data = open(path, "r", encoding="utf-8").read()
        data_json = json.loads(data)

        for db_detail in data_json["missing_databases"]:
            db_txt = (
                f"Name: {db_detail["name"]}\n"
                f"Description: {db_detail["description"]}\n"
            )
            count+= 1
            print(db_txt)
    print(f"Processed {count} entries")