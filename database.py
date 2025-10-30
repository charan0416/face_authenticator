# database.py (Final corrected version)

import os
import json
import uuid
import numpy as np
import time
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# --- Milvus Configuration ---
# We connect to localhost from Python, but tell Milvus to use the special Docker host for its internal connections.
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
# This is the special DNS name that Docker provides for containers to reach the host machine.
DOCKER_INTERNAL_HOST = "host.docker.internal"

COLLECTION_NAME = "face_recognition_system"

# ... (Other constants remain the same) ...
INSTANCE_FOLDER = 'instance'
METADATA_FILE = os.path.join(INSTANCE_FOLDER, 'metadata.json')
EMBEDDING_DIMENSION = 512
metadata = {"last_user_id": 0}


def init_milvus():
    """Connects to Milvus and ensures the collection and index exist."""
    global metadata

    # --- THIS IS THE CRITICAL CHANGE ---
    # We are setting advanced connection parameters to override Milvus's internal config.
    # This forces it to look for etcd and minio on the host machine from its perspective.
    connections.connect(
        "default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        server_pem_path="",
        server_key_path="",
        ca_pem_path="",
        server_name="",
        secure=False,
        # This tells the Milvus SDK to pass these config values to the server upon connection.
        # It forces the server to use the correct address for its dependencies.
        server_configs={
            "etcd.endpoints": f"{DOCKER_INTERNAL_HOST}:2379",
            "minio.address": f"{DOCKER_INTERNAL_HOST}:9000",
        },
    )
    print("Successfully connected to Milvus.")

    # Load metadata for frX IDs (no changes here)
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"last_user_id": 0}

    # The rest of the function remains the same as the Milvus version
    if not utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' does not exist. Creating...")
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION),
            FieldSchema(name="user_uuid", dtype=DataType.VARCHAR, max_length=36),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="human_id", dtype=DataType.VARCHAR, max_length=20),
        ]
        schema = CollectionSchema(fields, "Face recognition collection")
        collection = Collection(COLLECTION_NAME, schema)
        index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        collection.create_index(field_name="embedding", index_params=index_params)
        print("Collection and index created successfully.")

    Collection(COLLECTION_NAME).load()
    print("Milvus initialized and collection loaded.")

    # --- THE REST OF THE database.py FILE REMAINS THE SAME ---

    # Load metadata for frX IDs
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"last_user_id": 0}

    # Check if our collection already exists
    if not utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' does not exist. Creating...")
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION),
            FieldSchema(name="user_uuid", dtype=DataType.VARCHAR, max_length=36),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="human_id", dtype=DataType.VARCHAR, max_length=20),
        ]
        schema = CollectionSchema(fields, "Face recognition collection")
        collection = Collection(COLLECTION_NAME, schema)
        index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        collection.create_index(field_name="embedding", index_params=index_params)
        print("Collection and index created successfully.")

    Collection(COLLECTION_NAME).load()
    print("Milvus initialized and collection loaded.")


def save_metadata():
    """Saves the sequential ID counter to its file."""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Metadata saved.")


def get_next_human_id():
    """Increments and returns the next sequential user ID string."""
    metadata['last_user_id'] += 1
    save_metadata()
    return f"fr{metadata['last_user_id']}"


def add_user(name, avg_embedding):
    """Adds a new user to the Milvus collection."""
    collection = Collection(COLLECTION_NAME)
    user_uuid = str(uuid.uuid4())
    human_id = get_next_human_id()
    data_to_insert = [[avg_embedding], [user_uuid], [name], [human_id]]
    collection.insert(data_to_insert)
    collection.flush()
    print(f"Added user '{name}' with Human ID {human_id} to Milvus.")
    return user_uuid, human_id


def find_similar_user(embedding, threshold):
    """Searches Milvus for a similar face."""
    collection = Collection(COLLECTION_NAME)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param=search_params,
        limit=1,
        output_fields=["user_uuid", "name", "human_id"]
    )
    if not results or not results[0]:
        return None, None, None, None

    top_hit = results[0][0]
    distance = top_hit.distance

    if distance < threshold:
        user_uuid = top_hit.entity.get("user_uuid")
        name = top_hit.entity.get("name")
        human_id = top_hit.entity.get("human_id")
        return user_uuid, name, human_id, distance

    return None, None, None, None


def delete_user(user_uuid):
    """Deletes a user from Milvus by their UUID."""
    collection = Collection(COLLECTION_NAME)
    expr = f"user_uuid == '{user_uuid}'"
    result = collection.delete(expr)
    collection.flush()
    return result.delete_coun
def get_all_users():
    """Queries Milvus to get all stored users."""
    collection = Collection(COLLECTION_NAME)
    results = collection.query(expr="pk >= 0", output_fields=["user_uuid", "name", "human_id"])
    users = [{"uuid": r["user_uuid"], "name": r["name"], "human_id": r["human_id"]} for r in results]
    return users