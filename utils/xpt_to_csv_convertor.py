import argparse
import pandas as pd
import xport


def convert_xport_to_csv(input_file, output_file, rows):
    try:
        # Read SAS XPORT file into a pandas DataFrame
        with open(input_file, 'rb') as f:
            # Use xport.Reader to read the data
            reader = xport.Reader(f)
            df = pd.DataFrame(reader)

            # Take only the first 'rows' rows
            df = df.head(rows)

            # Check if the DataFrame is not empty
            if not df.empty:
                # Save DataFrame to CSV file
                df.to_csv(output_file, index=False)
                print(f"Conversion successful. CSV file saved as {output_file}")
            else:
                print("Error: Empty DataFrame from the XPORT file.")

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert SAS XPORT file to CSV format.')
    parser.add_argument('--file_name', required=True, help='Name of the SAS XPORT file')
    parser.add_argument('--rows', type=int, default=50, help='Number of rows to extract (default: 50)')
    args = parser.parse_args()

    input_file = f"src/training-data/raw-data/{args.file_name}"
    output_file = f"src/training-data/csv-data/{args.file_name[:args.file_name.rfind('.')]}.csv"

    convert_xport_to_csv(input_file, output_file, rows=args.rows)


if __name__ == "__main__":
    main()
