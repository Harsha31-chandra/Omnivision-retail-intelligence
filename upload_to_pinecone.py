from pinecone import Pinecone
import pandas as pd
import numpy as np
from tqdm import tqdm

# Initialize with your API Key
pc = Pinecone(api_key="pcsk_6UGTG1_NbStZZy8DVSCppTteVetowHiwKy5TcCiuxMg6x8VcgxC4MKxkioMLiKBGp21XbM")
index = pc.Index("fashion-search")
# 1. Initialize Pinecon

# 2. Load your local Kaggle data
print("Loading data from local files...")
df = pd.read_csv("cleaned_metadata.csv")
embeddings = np.load("clip_embeddings.npy")

# 3. Prepare data for Pinecone (Batch Uploading)
batch_size = 100
total_records = len(df)

print(f"Uploading {total_records} items to Pinecone...")

for i in tqdm(range(0, total_records, batch_size)):
    # Get the current batch
    batch_df = df.iloc[i : i + batch_size]
    batch_embeddings = embeddings[i : i + batch_size]
    
    vectors_to_upsert = []
    
    for j, row in batch_df.iterrows():
        # Pinecone requires: (id, vector, metadata)
        # We store metadata so the Frontend knows what to display!
        vector_data = {
            "id": str(row['id']), 
            "values": batch_embeddings[j - i].tolist(), 
            "metadata": {
                "productDisplayName": str(row.get('productDisplayName', 'Unknown')),
                "masterCategory": str(row.get('masterCategory', 'Unknown')),
                "subCategory": str(row.get('subCategory', 'Unknown'))
            }
        }
        vectors_to_upsert.append(vector_data)
        
    # Upload the batch to Pinecone
    index.upsert(vectors=vectors_to_upsert)

print("✅ All data successfully uploaded to the Vector Database!")