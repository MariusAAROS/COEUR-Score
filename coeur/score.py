import random
import numpy as np
import os
import pandas as pd
import re
import pypdf
import json
from coeur.cohesion import CohesionViz, CohesionScore
from coeur.exhaustiveness import ExhaustivenessViz, ExhaustivenessScore
from transformers import AutoTokenizer, AutoModel
import torch
from evaluate import load as load_metric
from rouge_score.rouge_scorer import RougeScorer
from bert_score import BERTScorer
import mauve
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

class Coeur:
    def __init__(self, model_name: str = "bert-base-uncased", 
                 stemming: bool = False, 
                 lemmatization: bool = False, 
                 remove_stopwords: bool = False,
                 remove_re_se_stopwords: bool = False,
                 include_ac: bool = False,
                 batch_size: int = 128,
                 pca_components: int = 50,
                 non_linear_components: int = 3,
                 random_state: int = 56):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.X = None
        self.X_hat = None
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.remove_stopwords = remove_stopwords
        self.remove_re_se_stopwords = remove_re_se_stopwords
        self.include_ac = include_ac

        self.pca_components = pca_components
        self.non_linear_components = non_linear_components

        self.random_state = random_state
        self.set_global_seed(random_state)
        try:
            self.device = torch.cuda.current_device()
            self.batch_size = batch_size
        except:
            self.device = -1
            self.batch_size = 1

    def set_global_seed(self, seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def load_data(self, ref_path: str, cand_path: str, 
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
            X_hat = CohesionViz.req_to_dataframe(X_hat)
        elif cand_mode == "csv":
            X_hat = pd.read_csv(cand_path)
        else:
            raise ValueError("cand_mode must be either 'json' or 'csv'")
        self.X = X
        self.X_hat = X_hat
        return X, X_hat
    
    def get_embs(self, text: str):
        def chunk_encode(text, length):
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
    
    def plot_coverage(self, reducer: str = None):
        rv = ExhaustivenessViz(include_ac=self.include_ac, remove_stopwords=self.remove_stopwords,
                    remove_re_se_stopwords=self.remove_re_se_stopwords,
                    stemming=self.stemming, lemmatization=self.lemmatization,
                    reducer=reducer, random_state=self.random_state)
        if self.include_ac:
            X_pred = " ".join((self.X_hat["user_story"] + " " + self.X_hat["acceptance_criteria"].str.join(" ")).tolist())
        else:
            X_pred = " ".join(self.X_hat["user_story"].tolist())

        rv.plot_coverage(self.X, X_pred)

    def experimental_score_coverage(self, verbose: bool = True, include_mauve: bool = True,
                       include_bleu: bool = True, include_bleurt: bool = True,
                       include_meteor: bool = True,
                       backlog_wise: bool = True, epic_wise: bool = True,
                       story_wise: bool = True) -> dict:
        def chunk_for_mauve(p_text, q_text, chunk_size=100):
            def chunk_text(text, n_tokens=100):
                sentences = text.split()
                return [" ".join(sentences[i:i+n_tokens]) for i in range(0, len(sentences), n_tokens)]
            
            p_chunks = chunk_text(p_text, chunk_size)
            q_chunks = re.split(r'(?<=\.)\s+', q_text)

            return p_chunks, q_chunks

        def get_single_score(X: str, X_pred: str, include_mauve: bool = True, include_bleu: bool = True,
                             include_bleurt: bool = True, include_meteor: bool = True) -> dict:
            bscore = bert_scorer.score(cands=[X_pred], refs=[X],
                                        batch_size=self.batch_size,
                                        verbose=False)
            rscore = rouge_scorer.score(prediction=X_pred, target=X)
            P_exhaustiveness, R_exhaustiveness, F_exhaustiveness = np.nan, np.nan, np.nan # This is a placeholder for a removed functionality
            if include_mauve:
                p_chunks, q_chunks = chunk_for_mauve(p_text=X, q_text=X_pred, chunk_size=100)
                mscore = mauve.compute_mauve(p_text=p_chunks, q_text=q_chunks,
                                            device_id=self.device, batch_size=self.batch_size,
                                            verbose=False, seed=self.random_state)
                mauve_scores = (mscore.mauve, mscore.mauve_star)
            if include_bleu:
                bleu = load_metric("bleu")
                bleu_score = bleu.compute(predictions=[X_pred], references=[[X]])["bleu"]
            if include_bleurt:
                bleurt = load_metric("bleurt", module_type="metric")
                bleurt_score = bleurt.compute(predictions=[X_pred], references=[X])["scores"][0]
            if include_meteor:
                meteor = load_metric("meteor")
                meteor_score = meteor.compute(predictions=[X_pred], references=[[X]])["meteor"]

            scores = {
                    "BERTScore Precision": bscore[0].numpy()[0],
                    "BERTScore Recall": bscore[1].numpy()[0],
                    "BERTScore F1": bscore[2].numpy()[0],
                    "ROUGE-1 Precision": rscore['rouge1'].precision,
                    "ROUGE-1 Recall": rscore['rouge1'].recall,
                    "ROUGE-1 F1": rscore['rouge1'].fmeasure,
                    "ROUGE-2 Precision": rscore['rouge2'].precision,
                    "ROUGE-2 Recall": rscore['rouge2'].recall,
                    "ROUGE-2 F1": rscore['rouge2'].fmeasure,
                    "ROUGE-L Precision": rscore['rougeL'].precision,
                    "ROUGE-L Recall": rscore['rougeL'].recall,
                    "ROUGE-L F1": rscore['rougeL'].fmeasure,
                    "Exhaustiveness Precision": P_exhaustiveness,
                    "Exhaustiveness Recall": R_exhaustiveness,
                    "Exhaustiveness F1": F_exhaustiveness,
                    "MAUVE": mauve_scores[0] if include_mauve else np.nan,
                    "MAUVE (Star)": mauve_scores[1] if include_mauve else np.nan,
                    "BLEU": bleu_score if include_bleu else np.nan,
                    "BLEURT": bleurt_score if include_bleurt else np.nan,
                    "METEOR": meteor_score if include_meteor else np.nan
                }
            
            return scores
        
        def get_multi_scores(X: str, X_pred: list[str]) -> list[dict]:
            bscores = bert_scorer.score(cands=X_pred, refs=[X]*len(X_pred), 
                                        batch_size=self.batch_size,
                                        verbose=False)
            rscores = []
            # exhscores = []
            for v in X_pred:
                rscores.append(rouge_scorer.score(target=X, prediction=v))
                # exhscores.append(es.score_coverage(X, v))
            all_scores = []
            for i, rs in enumerate(rscores):
                scores = {
                    "BERTScore Precision": bscores[0].numpy()[i],
                    "BERTScore Recall": bscores[1].numpy()[i],
                    "BERTScore F1": bscores[2].numpy()[i],
                    "ROUGE-1 Precision": rs['rouge1'].precision,
                    "ROUGE-1 Recall": rs['rouge1'].recall,
                    "ROUGE-1 F1": rs['rouge1'].fmeasure,
                    "ROUGE-2 Precision": rs['rouge2'].precision,
                    "ROUGE-2 Recall": rs['rouge2'].recall,
                    "ROUGE-2 F1": rs['rouge2'].fmeasure,
                    "ROUGE-L Precision": rs['rougeL'].precision,
                    "ROUGE-L Recall": rs['rougeL'].recall,
                    "ROUGE-L F1": rs['rougeL'].fmeasure,
                    "Exhaustiveness Precision": np.nan,
                    "Exhaustiveness Recall": np.nan,
                    "Exhaustiveness F1": np.nan,
                    "MAUVE": np.nan,
                    "MAUVE (Star)": np.nan,
                    "BLEU": np.nan,
                    "BLEURT": np.nan,
                    "METEOR": np.nan
                    # "Exhaustiveness Precision": exhscores[i][0],
                    # "Exhaustiveness Recall": exhscores[i][1],
                    # "Exhaustiveness F1": exhscores[i][2]
                }
                all_scores.append(scores)
            
            return all_scores

        bert_scorer = BERTScorer(lang="en", rescale_with_baseline=False, batch_size=self.batch_size, 
                                 device="cuda" if self.device != -1 else "cpu")
        rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        es = ExhaustivenessScore(tokenizer=self.tokenizer, model=self.model, include_ac=self.include_ac,
                                remove_stopwords=self.remove_stopwords,
                                remove_re_se_stopwords=self.remove_re_se_stopwords,
                                stemming=self.stemming, lemmatization=self.lemmatization,
                                pca_components=self.pca_components, non_linear_components=self.non_linear_components,
                                random_state=self.random_state)
        include_ac = False
        scores = []
        index = []

        if backlog_wise:
            if include_ac:
                X_pred = " ".join((self.X_hat["user_story"] + " " + self.X_hat["acceptance_criteria"].str.join(" ")).tolist())
            else:
                X_pred = " ".join(self.X_hat["user_story"].tolist())
            X_preproc = es.preprocess_text(self.X)
            X_pred_preproc = es.preprocess_text(X_pred)
            scores.append(get_single_score(X=X_preproc, X_pred=X_pred_preproc, include_mauve=include_mauve, include_bleu=include_bleu,
                                           include_bleurt=include_bleurt, include_meteor=include_meteor))
            index.append("Backlog-wise")
        if epic_wise:
            if include_ac:
                raise NotImplementedError("Epic-wise scoring with acceptance criteria is not implemented yet.")
            else:
                X_pred = self.X_hat.groupby("epic")["user_story"].apply(lambda x: " ".join(x)).tolist()
                X_preproc = es.preprocess_text(self.X)
                X_pred_preproc = [es.preprocess_text(v) for v in X_pred]
                subscore = get_multi_scores(X=X_preproc, X_pred=X_pred_preproc)
                mean_subscore = pd.DataFrame(subscore).mean().to_dict()
                scores.append(mean_subscore)
                index.append(f"Epic-wise ({len(X_pred)} epics)")
        if story_wise:
            if include_ac:
                raise NotImplementedError("Story-wise scoring with acceptance criteria is not implemented yet.")
            else:
                subscore = []
                X_pred = self.X_hat["user_story"].tolist()
                X_preproc = es.preprocess_text(self.X)
                X_pred_preproc = [es.preprocess_text(v) for v in X_pred]
                subscore = get_multi_scores(X=X_preproc, X_pred=X_pred_preproc)
                mean_subscore = pd.DataFrame(subscore).mean().to_dict()
                scores.append(mean_subscore)
                index.append(f"Story-wise ({len(subscore)} stories)")

        df_scores = pd.DataFrame(scores, index=index)
        if verbose:
            print(df_scores)
        return df_scores
    
    def plot_coherence(self, unwind: bool = False):
        rv = CohesionViz(include_ac=self.include_ac, remove_stopwords=self.remove_stopwords,
                    remove_re_se_stopwords=self.remove_re_se_stopwords,
                    stemming=self.stemming, lemmatization=self.lemmatization,
                    reducer=Isomap(n_components=2), random_state=self.random_state)
        if unwind:
            X_hat_unwind = rv.unwind_ac(self.X_hat)
            rv.plot(X_hat_unwind)
        else:
            rv.plot(self.X_hat)

    def experimental_score_coherence(self, clusterers = None, unwind: bool = False, highlight: bool = True):
        rs = CohesionScore(lemmatization=self.lemmatization, stemming=self.stemming,
                      remove_stopwords=self.remove_stopwords, include_ac=self.include_ac,
                      remove_re_se_stopwords=self.remove_re_se_stopwords,
                      random_state=self.random_state)
        if unwind:
            X_hat_current = rs.unwind_ac(self.X_hat)
        else:
            X_hat_current = self.X_hat    
        coherence_scores = []
        if clusterers is not None:
            for method in clusterers:
                rs.clusterer = method
                if hasattr(rs.clusterer, 'random_state'):
                    rs.clusterer.random_state = self.random_state
                current_metrics = rs.fit(X_hat_current)
                coherence_scores.append(current_metrics)
        else:
            n_epics = self.X_hat["epic"].nunique()
            clusterers = [KMeans(n_clusters=n_epics, random_state=self.random_state), 
                          AgglomerativeClustering(n_clusters=n_epics, metric='cosine', linkage='average'),
                          SpectralClustering(n_clusters=n_epics, random_state=self.random_state)]
            for method in clusterers:
                rs.clusterer = method
                current_metrics = rs.fit(X_hat_current)
                coherence_scores.append(current_metrics)
        highlighted_scores = rs.compare(
            *coherence_scores,
            index_names=[type(c).__name__ for c in clusterers],
            highlight=highlight
        )
        return highlighted_scores
    
    def score_count(self, verbose: bool = True) -> pd.DataFrame:
        len_x = len(self.X)
        n_stories = len(self.X_hat)
        n_stories_per_epic = self.X_hat.groupby("epic").size().mean()
        n_ac_per_story = self.X_hat["acceptance_criteria"].apply(len).mean()
        df_counts = pd.DataFrame({
            "Length of X (chars)": [len_x],
            "Number of User Stories": [n_stories],
            "Avg User Stories per Epic": [n_stories_per_epic],
            "Avg AC per User Story": [n_ac_per_story]
        })
        if verbose:
            print(df_counts)
        return df_counts
    
    def score(self, R: str, B: pd.DataFrame, l="s", lmbd=0.5, sigma="auto", psi="auto", phi="auto") -> dict:
        def rs_to_dict(rs):
            raw = {
                    "ROUGE-1 Precision": rs['rouge1'].precision,
                    "ROUGE-1 Recall": rs['rouge1'].recall,
                    "ROUGE-1 F1": rs['rouge1'].fmeasure,
                    "ROUGE-2 Precision": rs['rouge2'].precision,
                    "ROUGE-2 Recall": rs['rouge2'].recall,
                    "ROUGE-2 F1": rs['rouge2'].fmeasure,
                    "ROUGE-L Precision": rs['rougeL'].precision,
                    "ROUGE-L Recall": rs['rougeL'].recall,
                    "ROUGE-L F1": rs['rougeL'].fmeasure,
                }
            return raw
        
        n_epics = B["epic"].nunique()
        if sigma == "auto":
            sigma = "ROUGE-2 Precision"
        else:
            raise NotImplementedError("Custom sigma is not implemented yet.")
        if phi == "auto":
            phi = SpectralClustering(n_clusters=n_epics, random_state=self.random_state)
        if psi == "auto":
            psi = "adjusted_mutual_info"
        coh_scorer = CohesionScore(clusterer=phi, lemmatization=self.lemmatization, stemming=self.stemming,
                    remove_stopwords=self.remove_stopwords, include_ac=self.include_ac,
                    remove_re_se_stopwords=self.remove_re_se_stopwords,
                    random_state=self.random_state)
        
        exh_scorer = ExhaustivenessScore(tokenizer=self.tokenizer, model=self.model, include_ac=self.include_ac,
                    remove_stopwords=self.remove_stopwords,
                    remove_re_se_stopwords=self.remove_re_se_stopwords,
                    stemming=self.stemming, lemmatization=self.lemmatization,
                    random_state=self.random_state)
        
        coh = coh_scorer.fit(B)[psi]
        if l == "s":
            exh_subscores = []
            stories = B["user_story"].tolist()
            for story in stories:
                rs = exh_scorer.score_exhaustiveness(R, story)
                raw = rs_to_dict(rs)
                exh_subscores.append(raw)
            exh = pd.DataFrame(exh_subscores).mean().to_dict()[sigma]
        elif l == "e":
            exh_subscores = []
            epics = B.groupby("epic")["user_story"].apply(lambda x: " ".join(x)).tolist()
            for epic in epics:
                rs = exh_scorer.score_exhaustiveness(R, epic)
                raw = rs_to_dict(rs)
                exh_subscores.append(raw)
            exh = pd.DataFrame(exh_subscores).mean().to_dict()[sigma]
        elif l == "b":
            backlog = " ".join(B["user_story"].tolist())
            exh = exh_scorer.score_exhaustiveness(R, backlog)
            raw = rs_to_dict(exh)
            exh = raw[sigma]
        else:
            raise ValueError("l must be one of 's' (story-wise), 'e' (epic-wise), or 'b' (backlog-wise)")
        coeur = lmbd * exh + (1 - lmbd) * coh
        return {"COEUR": coeur, "Cohesion": coh, "Exhaustiveness": exh}