import pandas as pd
import os
from sentence_transformers import SentenceTransformer

def encode_user_stories(df, model_name='all-mpnet-base-v2', batch_size=512):
    model = SentenceTransformer(model_name)
    user_stories = df['user_story'].tolist()
    embeddings = model.encode(user_stories, batch_size=batch_size)
    return embeddings


load_path = "datasets/neodataset/original"
save_path = "datasets/neodataset/noise_intended.csv"

file_names = os.listdir(load_path)
all_data = pd.DataFrame()
for file_name in file_names:
    file_path = os.path.join(load_path, file_name)
    data = pd.read_csv(file_path)
    all_data = pd.concat([all_data, data], ignore_index=True)

max_len = pd.read_csv("datasets/dalpiaz/noise_intended.csv").shape[0]
all_data_sampled = all_data.sample(n=max_len, random_state=42).reset_index(drop=True)
all_data_sampled_reduced = all_data_sampled.drop(columns=["issuekey", "created", "description", "storypoints"])
all_data_sampled_reduced["epic"] = "UNDEFINED"
all_data_sampled_reduced.rename(columns={"title": "user_story"}, inplace=True)
all_data_sampled_reduced = all_data_sampled_reduced[["project_id", "epic", "user_story"]]
all_data_sampled_reduced["embedding"] = encode_user_stories(all_data_sampled_reduced).tolist()
all_data_sampled_reduced.to_csv(save_path, index=False)