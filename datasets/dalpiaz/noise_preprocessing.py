import pandas as pd
import os
from sentence_transformers import SentenceTransformer

external_data_path = "datasets/dalpiaz/original/"

def dalpiaz_to_dataframe(dalpiaz_path):
    files = [f for f in os.listdir(dalpiaz_path) if os.path.isfile(os.path.join(dalpiaz_path, f))]
    dalpiaz_data = []
    project_ids = []
    project_names = []
    exclusion_list = ['g19-alfred.txt']
    for file in files:
        if file in exclusion_list:
            continue
        with open(os.path.join(dalpiaz_path, file), 'r') as f:
            for line in f:
                data = line.strip()
                dalpiaz_data.append(data)
                cur_id, cur_name = file.replace('.txt', '').split('-')
                project_ids.append(cur_id)
                project_names.append(cur_name)
    data = {'project_id': project_ids, 'epic': project_names, 'user_story': dalpiaz_data}
    df = pd.DataFrame(data)
    df.iloc[0, 2] = df.iloc[0, 2][3:] # Remove BOM if present
    return df

def encode_user_stories(df, model_name='all-mpnet-base-v2', batch_size=512):
    model = SentenceTransformer(model_name)
    user_stories = df['user_story'].tolist()
    embeddings = model.encode(user_stories, batch_size=batch_size)
    return embeddings

dalpiaz = dalpiaz_to_dataframe(external_data_path)
dalpiaz["embedding"] = encode_user_stories(dalpiaz, model_name='all-mpnet-base-v2', batch_size=512).tolist()
dalpiaz.to_csv("datasets/dalpiaz/noise_intended.csv", index=False)