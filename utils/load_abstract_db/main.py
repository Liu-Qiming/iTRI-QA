from file_readers import read_bib_file, read_json_file
from data_processing import merge_dataframes
from data_savers import save_to_mongodb, save_to_yaml

if __name__ == "__main__":
    bibtex_file_path = "/Users/maotian/Documents/Project/Project-GPT/iTRI-GPT/utils/load_abstract_db/pubmed-telomerenh-set.bib"
    json_file_path = "path/to/your/file.json"
    db_name = "research_db"
    collection_name = "papers"
    yaml_file_path = "merged_data.yaml"
    cluster_uri = None  # Replace with your cluster URI if applicable

    try:
        bib_df = read_bib_file(bibtex_file_path)
        json_df = read_json_file(json_file_path)
        merged_df = merge_dataframes(bib_df, json_df)

        print("Merged DataFrame:")
        print(merged_df.head())

        save_to_mongodb(merged_df, db_name, collection_name, cluster_uri)
        save_to_yaml(merged_df, yaml_file_path)

    except Exception as e:
        print(f"An error occurred: {e}")