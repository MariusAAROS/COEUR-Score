import pandas as pd
import pypdf
import json
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Generator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import random
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import Subset

class SpecRetriever:
    def load_specs(self, path: str) -> list[str]:
        directory = PyPDFLoader(path).load()
        chunk_overlap = int(self.chunk_size * self.overlap_rate)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=chunk_overlap
        )

        docs = text_splitter.split_documents(directory)
        for doc in docs:
            doc.page_content = doc.page_content.replace("\n", " ")\
                                            .replace("•", " ") \
                                            .replace("♦", " ") \
                                            .replace("  ", " ") \
                                            .strip()
        return docs

    def __init__(self, path: str, docs=None, index=None, model=None,
                 chunk_size: int = 200, overlap_rate: float = 0.15):
        self.model = model if model is not None else SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = chunk_size
        self.overlap_rate = overlap_rate

        if index is not None and docs is not None:
            self.index = index
            self.docs = docs
        else:
            self.docs = self.load_specs(path)
            vectors = []
            for doc in self.docs:
                vectors.append(self.model.encode(doc.page_content))
            dimension = vectors[0].shape[0]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(vectors))

    def get_similar_specs(self, spec: str, top_k: int = 5):
        spec_vector = self.model.encode(spec)
        _, indices = self.index.search(np.array([spec_vector]), top_k)
        similar_docs = [self.docs[i].page_content for i in indices[0]]
        return similar_docs

def req_to_dataframe(data: dict) -> pd.DataFrame:
        rows = []
        for source in data.keys():
            current_rows = []
            for epic in data[source]["epics"]:
                epic_name = epic["epic"].strip()
                for story in epic["user_stories"]:
                    row = {
                        "source": source,
                        "epic": epic_name,
                        "user_story": story["user_story"].strip(),
                        "acceptance_criteria": story["acceptance_criteria"],
                        "independent": story["independent"],
                        "negotiable": story["negotiable"],
                        "valuable": story["valuable"],
                        "estimable": story["estimable"],
                        "small": story["small"],
                        "testable": story["testable"]
                    }
                    current_rows.append(row)
            rows.extend(current_rows)
        df = pd.DataFrame(rows)
        return df

def load_data(ref_path: str, cand_path: str, 
                ref_mode: str = "pdf", cand_mode: str = "csv") -> tuple[str, pd.DataFrame]:
    if ref_mode == "pdf":
        reader = pypdf.PdfReader(ref_path)
        X = ""
        for page in reader.pages:
            X += page.extract_text()
    else:
        raise ValueError("ref_mode must be 'pdf'")
    if cand_mode == "json":
        with open(cand_path, 'r', encoding='utf-8') as f:
            X_hat = json.load(f)
        if isinstance(X_hat, list):
            X_hat = {"model": {"epics": X_hat}}
        elif isinstance(X_hat, dict) and "epics" in X_hat.keys():
            X_hat = {"model": X_hat}
        else:
            pass
        X_hat = req_to_dataframe(X_hat)
    elif cand_mode == "csv":
        X_hat = pd.read_csv(cand_path)
    else:
        raise ValueError("cand_mode must be either 'json' or 'csv'")
    return X, X_hat

class UserStoryDataset(Dataset):
    def __init__(self, dataset_names: list[str] | str,
                 top_k_specs: int = 3,
                 top_k_related: int = 3
                 ):
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        self.specs = {}
        self.backlog = pd.DataFrame()
        self.retrievers = {}
        self.top_k_specs = top_k_specs
        self.top_k_related = top_k_related

        for name in dataset_names:
            spec, backlog = load_data(
                ref_path=f"datasets/{name}/{name}_specs.pdf",
                cand_path=f"datasets/{name}/{name}_backlog.csv"
            )
            self.specs[name] = UserStoryDataset.preprocess_specs(spec)
            backlog["dataset"] = name
            self.backlog = pd.concat([self.backlog, backlog], ignore_index=True)
        
        for name in dataset_names:
            retriever = SpecRetriever(
                path=f"datasets/{name}/{name}_specs.pdf",
                chunk_size=200
            )
            self.retrievers[name] = retriever
            
    @staticmethod
    def preprocess_specs(spec: str, threshold: int = 60) -> str:
        new_lines = []
        for line in spec.split("•"):
            line = line.strip()
            line = line.replace("\n", " ")
            new_lines.extend(line.split("♦"))

        final_lines = []
        current = ""
        for line in new_lines:

            if len(line.strip()) > threshold:
                final_lines.append(line.strip())
            else:
                current += " " + line.strip()
                if len(current.strip()) > threshold:
                    final_lines.append(current.strip())
                    current = ""
        return "\n".join(final_lines)
    
    def __len__(self):
        return len(self.backlog)
    
    def __getitem__(self, idx):
        row = self.backlog.iloc[idx]
        dataset_name = row["dataset"]
        epic = row['epic']
        user_story = row['user_story']
        spec = "\n".join(self.retrievers[dataset_name].get_similar_specs(user_story, top_k=self.top_k_specs))
        related_user_stories = self.backlog[
            (self.backlog['dataset'] == dataset_name) & 
            (self.backlog['epic'] == row['epic']) & 
            (self.backlog.index != idx)
        ]['user_story'].tolist()
        random.shuffle(related_user_stories)
        related_user_stories = related_user_stories[:self.top_k_related]
        return spec, user_story, related_user_stories, epic
    
def get_dataloader(dataset_names: list[str] | str, batch_size: int = 1, train_test_ratio=(0.8, 0.2), shuffle: bool = True) -> DataLoader:
    dataset = UserStoryDataset(dataset_names)

    # Get unique epics to prevent data leakage between train and test
    unique_epics = dataset.backlog[['dataset', 'epic']].drop_duplicates()
    epics_list = list(unique_epics.itertuples(index=False, name=None))
    
    # Shuffle epics for random splitting
    generator = random.Random(42)
    generator.shuffle(epics_list)

    split_idx = int(len(epics_list) * train_test_ratio[0])
    train_epics = set(epics_list[:split_idx])
    
    # Identify indices for train and test
    train_indices = []
    test_indices = []
    
    for idx, row in dataset.backlog.iterrows():
        if (row['dataset'], row['epic']) in train_epics:
            train_indices.append(idx)
        else:
            test_indices.append(idx)

    train_data = Subset(dataset, train_indices)
    test_data = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # Also return the datasets for Trainer access if needed
    return train_dataloader, test_dataloader, train_data, test_data