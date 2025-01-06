import os
import yaml
from pymongo import MongoClient

def save_to_mongodb(df, db_name, collection_name, cluster_uri=None):
    """
    Save a pandas DataFrame to a MongoDB collection.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        db_name (str): Name of the MongoDB database.
        collection_name (str): Name of the MongoDB collection.
        cluster_uri (str, optional): MongoDB connection URI for a cluster. Defaults to localhost.
    """
    if cluster_uri is None:
        cluster_uri = "mongodb://localhost:27017/"

    client = MongoClient(cluster_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Convert DataFrame to dictionary and insert into MongoDB
    records = df.to_dict('records')
    collection.insert_many(records)

    print(f"Data successfully saved to MongoDB database '{db_name}', collection '{collection_name}'.")

def save_to_yaml(df, yaml_file):
    """
    Save a pandas DataFrame to a YAML file.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        yaml_file (str): Path to the YAML file.
    """
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(df.to_dict('records'), f, default_flow_style=False, allow_unicode=True)

    print(f"Data successfully saved to YAML file '{yaml_file}'.")
