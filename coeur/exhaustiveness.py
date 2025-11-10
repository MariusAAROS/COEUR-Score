from plotly import graph_objects as go

import torch
import string
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from rouge_score import rouge_scorer

class PCAtoTSNE:
    def __init__(self, pca_components=50, tsne_components=2, random_state=42, pca_kwargs={}, tsne_kwargs={}):
        self.pca = PCA(n_components=pca_components, random_state=random_state, **pca_kwargs)
        self.tsne = TSNE(n_components=tsne_components, random_state=random_state, **tsne_kwargs)

    def fit_transform(self, X):
        X_pca = self.pca.fit_transform(X)
        X_tsne = self.tsne.fit_transform(X_pca)
        return X_tsne

class PCAtoUMAP:
    def __init__(self, pca_components=50, umap_components=2, random_state=42, pca_kwargs={}, umap_kwargs={}):
        self.pca = PCA(n_components=pca_components, random_state=random_state, **pca_kwargs)
        self.umap = UMAP(n_components=umap_components, random_state=random_state, **umap_kwargs)

    def fit_transform(self, X):
        X_pca = self.pca.fit_transform(X)
        X_umap = self.umap.fit_transform(X_pca)
        return X_umap

class ExhaustivenessAbstract:
    def __init__(self, tokenizer = None, model = None, include_ac: bool = False, remove_stopwords: bool = True,
                 lemmatization: bool = True, stemming: bool = False, remove_re_se_stopwords: bool = True, random_state: int = 42):
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = model if model is not None else AutoModel.from_pretrained("bert-base-uncased")
        self.include_ac = include_ac
        self.remove_stopwords = remove_stopwords
        self.lemmatization = lemmatization
        self.stemming = stemming
        self.remove_re_se_stopwords = remove_re_se_stopwords
        self.random_state = random_state
        self.re_se_stopwords = [
            "as", "a", "i", "want", "to", "so", "that", "i", "the", "and", 
            "in", "is", "it", "of", "for", "on", "with", 
            "this", "by", "an", "be", "are", "at", "from"]

    def get_embs(self, text: str):
        def chunk_encode(text: str, length: int):
            input_ids = []
            attention_mask = []

            batched = [text[i:i+512] for i in range(0, len(text), length)]
            inputs = self.tokenizer.batch_encode_plus(batched,
                                            max_length=length,
                                            padding=True, #implements dynamic padding
                                            truncation=True,
                                            return_attention_mask=True,
                                            return_token_type_ids=False)
            input_ids.extend(inputs['input_ids'])
            attention_mask.extend(inputs['attention_mask'])
            return torch.tensor(input_ids), torch.tensor(attention_mask)
        
        input_ids, attention_mask = chunk_encode(text=text, length=self.model.config.max_position_embeddings)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state
        # return with reshaping without batch dimension
        last_hidden_states = last_hidden_states.reshape(-1, last_hidden_states.size(-1))
        input_ids = input_ids.reshape(-1)
        attention_mask = attention_mask.reshape(-1)
        return last_hidden_states, input_ids, attention_mask

    def preprocess_text(self, text: str):
        en_stopwords = set(nltk_stopwords.words('english'))
        fr_stopwords = set(nltk_stopwords.words('french'))
        stopwords = en_stopwords.union(fr_stopwords)
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        
        def get_wordnet_pos(word):
            """Map POS tag to first character used by WordNetLemmatizer"""
            tag = pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        
        tokens = word_tokenize(text.lower())
        if self.remove_stopwords:
            filtered_tokens = [word for word in tokens if word.isalnum() and word not in stopwords]
        else:
            filtered_tokens = [word for word in tokens if word.isalnum()]
        if self.remove_re_se_stopwords:
            filtered_tokens = [word for word in filtered_tokens if word not in self.re_se_stopwords]
        if self.lemmatization:
            filtered_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in filtered_tokens]
        if self.stemming:
            filtered_tokens = [stemmer.stem(word) for word in filtered_tokens]
        processed_text = " ".join(filtered_tokens)
        return processed_text
    
    def get_coverage_embs(self, X: str, X_hat: str):
        def filter_tokens(embs, tokens):
            special_tokens = set(self.tokenizer.all_special_tokens)
            punctuation = set(string.punctuation)
            filtered_tokens = []
            filtered_embs = []

            for tok, emb in zip(tokens, embs):
                # skip special tokens
                dec_tok = self.tokenizer.decode(tok)
                if dec_tok in special_tokens:
                    continue
                # skip punctuation or subword artifacts
                if dec_tok in punctuation or dec_tok.startswith("##"):
                    continue
                filtered_tokens.append(tok)
                filtered_embs.append(emb)

            filtered_embs = torch.stack(filtered_embs)
            return filtered_embs, filtered_tokens
        
        X_proc = self.preprocess_text(X)
        X_hat_proc = self.preprocess_text(X_hat)

        x_embs, x_tokens, _ = self.get_embs(X_proc)
        x_hat_embs, x_hat_tokens, _ = self.get_embs(X_hat_proc)

        x_embs_filt, x_tokens_filt = filter_tokens(x_embs, x_tokens)
        x_hat_embs_filt, x_hat_tokens_filt = filter_tokens(x_hat_embs, x_hat_tokens)

        return x_embs_filt, x_tokens_filt, x_hat_embs_filt, x_hat_tokens_filt
    
class ExhaustivenessViz(ExhaustivenessAbstract):
    def __init__(self, tokenizer=None, model=None, reducer=None, include_ac = False, remove_stopwords = True, 
                 lemmatization = True, stemming = False, remove_re_se_stopwords = True, pca_components=50, 
                 random_state = 42):
        super().__init__(tokenizer=tokenizer, model=model, include_ac=include_ac, remove_stopwords=remove_stopwords, 
                         lemmatization=lemmatization, stemming=stemming, remove_re_se_stopwords=remove_re_se_stopwords, 
                         random_state=random_state)
        self.reducer = UMAP(n_components=2, n_neighbors=5, n_jobs=1, random_state=random_state, metric="cosine")

    def plot_exhaustiveness(self, X: str, X_hat: str):

        x_embs_filt, x_tokens_filt, x_hat_embs_filt, x_hat_tokens_filt = self.get_coverage_embs(X, X_hat)

        combined = torch.cat([x_embs_filt, x_hat_embs_filt], dim=0).cpu().numpy()
        combined_2d = self.reducer.fit_transform(combined)

        n_orig = x_embs_filt.shape[0]
        n_hat = x_hat_embs_filt.shape[0]
        orig_2d = combined_2d[:n_orig]
        hat_2d = combined_2d[n_orig:]

        # simple hover text
        hover_orig = [f"{self.tokenizer.decode(x_tokens_filt[i])}" for i in range(n_orig)]
        hover_hat = [f"{self.tokenizer.decode(x_hat_tokens_filt[i])}" for i in range(n_hat)]

        # build plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=orig_2d[:, 0],
            y=orig_2d[:, 1],
            mode="markers",
            marker=dict(size=6, color="royalblue", opacity=0.8),
            name="X (original)",
            text=hover_orig,
            hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=hat_2d[:, 0],
            y=hat_2d[:, 1],
            mode="markers",
            marker=dict(size=6, color="crimson", opacity=0.9, symbol="diamond"),
            name="X_hat (pred)",
            text=hover_hat,
            hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
        ))

        fig.update_layout(
            title="UMAP 2D: x_embs (blue) vs x_hat_embs (red)",
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            width=900,
            height=700,
            legend=dict(itemsizing="constant")
        )
        fig.show()

class ExhaustivenessScore(ExhaustivenessAbstract):
    def __init__(self, tokenizer=None, model=None, reducer=None, include_ac = False, remove_stopwords = True, 
                 lemmatization = True, stemming = False, remove_re_se_stopwords = True, pca_components=50, 
                 non_linear_components=3, random_state = 42):
        super().__init__(tokenizer=tokenizer, model=model, include_ac=include_ac, remove_stopwords=remove_stopwords, 
                         lemmatization=lemmatization, stemming=stemming, remove_re_se_stopwords=remove_re_se_stopwords, 
                         random_state=random_state)
        self.reducer = UMAP(n_components=non_linear_components, n_neighbors=15, n_jobs=1, random_state=random_state, metric="cosine")

    def score_exhaustiveness(self, X: str, X_hat: str):
        # rouge 1 2 l in precision recall f1
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        scores = rouge.score(prediction=X_hat, target=X)
        return scores