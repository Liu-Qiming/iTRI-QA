from file_readers import read_bib_file
from data_processing import merge_dataframes
from data_savers import save_to_mongodb, save_to_yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BibTeX files and save to MongoDB or YAML.")
    parser.add_argument(
        "--bibtex_files", nargs='+', required=True,
        help="Paths to one or more BibTeX files."
    )
    parser.add_argument(
        "--output_type", choices=['mongodb', 'yaml'], required=True,
        help="Type of output: 'mongodb' or 'yaml'."
    )
    parser.add_argument(
        "--db_name", type=str, default="research_db",
        help="Name of the MongoDB database (required if output_type is 'mongodb')."
    )
    parser.add_argument(
        "--collection_name", type=str, default="papers",
        help="Name of the MongoDB collection (required if output_type is 'mongodb')."
    )
    parser.add_argument(
        "--yaml_file", type=str, default="output.yaml",
        help="Path to the output YAML file (required if output_type is 'yaml')."
    )
    parser.add_argument(
        "--cluster_uri", type=str, default=None,
        help="MongoDB cluster URI (optional, defaults to localhost)."
    )

    args = parser.parse_args()

    try:
        # Read and combine all BibTeX files
        bib_dfs = [read_bib_file(bib_file) for bib_file in args.bibtex_files]
        combined_df = merge_dataframes(*bib_dfs)

        print("Combined DataFrame:")
        print(combined_df.head())

        # Save to the specified output
        if args.output_type == "mongodb":
            save_to_mongodb(combined_df, args.db_name, args.collection_name, args.cluster_uri)
        elif args.output_type == "yaml":
            save_to_yaml(combined_df, args.yaml_file)

    except Exception as e:
        print(f"An error occurred: {e}")
