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

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

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

    def explain(self, X: str, X_hat: pd.DataFrame, plot: bool = False):
        def rs_to_dict(rs):
            return {
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

        # 1. Scoring Logic
        story_subscores = []
        stories = X_hat["user_story"].tolist()
        for story in stories:
            rs = self.score_exhaustiveness(X, story)
            story_subscores.append(rs_to_dict(rs))

        epic_names = X_hat["epic"].unique()
        epics = X_hat.groupby("epic")["user_story"].apply(lambda x: " ".join(x)).tolist()
        epic_subscores = []
        for epic in epics:
            rs = self.score_exhaustiveness(X, epic)
            epic_subscores.append(rs_to_dict(rs))

        backlog = " ".join(X_hat["user_story"].tolist())
        backlog_score_raw = self.score_exhaustiveness(X, backlog)
        backlog_dict = rs_to_dict(backlog_score_raw)

        if plot:
            # Calculate Precision scores and averages
            story_precision = [scores['ROUGE-2 Precision'] for scores in story_subscores]
            mean_story_precision = np.mean(story_precision)
            
            epic_precision = [scores['ROUGE-2 Precision'] for scores in epic_subscores]
            mean_epic_precision = np.mean(epic_precision)
            
            # Identify problematic stories and epics (below average precision)
            problematic_stories = []
            for i, (story, scores) in enumerate(zip(stories, story_subscores)):
                precision = scores['ROUGE-2 Precision']
                if precision < mean_story_precision:
                    problematic_stories.append({
                        'index': i,
                        'epic': X_hat.iloc[i]['epic'],
                        'user_story': story,
                        'precision': precision,
                        'rouge_2_f1': scores['ROUGE-2 F1']
                    })
            
            problematic_epics = []
            for i, (epic_name, scores) in enumerate(zip(epic_names, epic_subscores)):
                precision = scores['ROUGE-2 Precision']
                if precision < mean_epic_precision:
                    problematic_epics.append({
                        'name': epic_name,
                        'precision': precision,
                        'rouge_2_f1': scores['ROUGE-2 F1']
                    })
            
            # Calculate dynamic figure height
            base_height = 14  # Increased base height for stacked plots
            cards_per_row = 2
            story_rows = max(1, (len(problematic_stories) + cards_per_row - 1) // cards_per_row) if problematic_stories else 0
            epic_rows = max(1, (len(problematic_epics) + cards_per_row - 1) // cards_per_row) if problematic_epics else 0
            
            # Increased height multiplier to prevent cropping
            dashboard_height = max(6, 6 + story_rows * 1.6 + epic_rows * 1.6)
            total_height = base_height + dashboard_height
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, total_height))
            dashboard_ratio = dashboard_height / total_height
            top_plots_ratio = base_height / total_height
            
            # Adjusted grid spec for 4 rows: 1 for summary, 2 for plots, 1 for dashboard
            gs = fig.add_gridspec(4, 1, height_ratios=[top_plots_ratio/5, top_plots_ratio*2/5, top_plots_ratio*2/5, dashboard_ratio], 
                                hspace=0.15)
            
            # First row - Backlog-level metrics (summary boxes)
            ax0 = fig.add_subplot(gs[0])
            ax0.axis('off')
            ax0.set_xlim(0, 10)
            ax0.set_ylim(0, 2)
            
            ax0.text(5, 1.7, 'Backlog-Level Exhaustiveness', fontsize=16, fontweight='bold', 
                    ha='center', va='top', color='#2c3e50')
            
            # Backlog summary boxes
            box_width = 2.2
            box_height = 0.9
            box_y = 0.5
            
            # Changed to Precision, Recall, and F1, all blue
            metrics = [
                ('ROUGE-2 Precision', backlog_dict['ROUGE-2 Precision'], 2.0),
                ('ROUGE-2 Recall', backlog_dict['ROUGE-2 Recall'], 5.0),
                ('ROUGE-2 F1', backlog_dict['ROUGE-2 F1'], 8.0)
            ]
            
            for label, value, x_pos in metrics:
                color = '#d0ebff'
                edge_color = '#339af0'
                text_color = '#1864ab'
                
                rect = plt.Rectangle((x_pos - box_width/2, box_y - box_height/2), box_width, box_height, 
                                    facecolor=color, edgecolor=edge_color, linewidth=2)
                ax0.add_patch(rect)
                ax0.text(x_pos, box_y, f'{label}\n{value:.3f}', 
                        ha='center', va='center', fontsize=11, fontweight='bold', color=text_color)
            
            # Second row - Epic-level Precision
            ax1 = fig.add_subplot(gs[1])
            
            # Color bars based on average
            epic_colors = ['#51cf66' if p >= mean_epic_precision else '#ff6b6b' for p in epic_precision]
            
            x = np.arange(len(epic_names))
            
            ax1.bar(x, epic_precision, color=epic_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
            
            # Add average line
            ax1.axhline(y=mean_epic_precision, color='#e03131', linestyle='--', linewidth=2,
                       label=f'Average: {mean_epic_precision:.3f}')
            
            ax1.set_ylabel('Precision Score', fontsize=11)
            ax1.set_title('Epic-Level Exhaustiveness (ROUGE-2 Precision)', fontsize=12, fontweight='bold', pad=15)
            ax1.set_xticks(x)
            
            # Truncate epic names
            truncated_names = [name[:20] + '...' if len(name) > 20 else name for name in epic_names]
            ax1.set_xticklabels(truncated_names, rotation=20, ha='right', fontsize=9)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_facecolor('#f8f9fa')
            
            # Third row - Story-level Precision
            ax2 = fig.add_subplot(gs[2])
            
            # Color bars based on average
            story_colors = ['#51cf66' if p >= mean_story_precision else '#ff6b6b' for p in story_precision]
            
            x_stories = np.arange(len(stories))
            
            ax2.bar(x_stories, story_precision, color=story_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
            
            # Add average line
            ax2.axhline(y=mean_story_precision, color='#e03131', linestyle='--', linewidth=2,
                       label=f'Average: {mean_story_precision:.3f}')
            
            ax2.set_ylabel('Precision Score', fontsize=11)
            ax2.set_xlabel('Story Index', fontsize=11)
            ax2.set_title('Story-Level Exhaustiveness (ROUGE-2 Precision)', fontsize=12, fontweight='bold', pad=15)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_facecolor('#f8f9fa')
            
            # Fourth section - Dashboard
            ax3 = fig.add_subplot(gs[3])
            ax3.axis('off')
            
            # Updated y_max calculation to match height
            dashboard_y_max = 6 + story_rows * 1.6 + epic_rows * 1.6
            ax3.set_xlim(0, 10)
            ax3.set_ylim(0, dashboard_y_max)
            
            # Title
            ax3.text(5, dashboard_y_max - 0.5, 'Exhaustiveness Quality Dashboard', fontsize=18, fontweight='bold', 
                    ha='center', va='top', color='#2c3e50')
            
            # Summary stats
            total_stories = len(stories)
            good_stories = total_stories - len(problematic_stories)
            total_epics = len(epic_names)
            good_epics = total_epics - len(problematic_epics)
            
            # Summary boxes
            summary_y = dashboard_y_max - 1.5
            box_width = 1.8
            box_height = 0.8
            
            # Stories summary
            story_rect = plt.Rectangle((1.5, summary_y - box_height/2), box_width, box_height, 
                                      facecolor='#d4edda', edgecolor='#2ed573', linewidth=2)
            ax3.add_patch(story_rect)
            ax3.text(2.4, summary_y, f'✓ Good Stories\n{good_stories}/{total_stories}', ha='center', va='center', 
                    fontsize=11, fontweight='bold', color='#155724')
            
            issue_rect = plt.Rectangle((3.5, summary_y - box_height/2), box_width, box_height, 
                                      facecolor='#f8d7da', edgecolor='#ff4757', linewidth=2)
            ax3.add_patch(issue_rect)
            ax3.text(4.4, summary_y, f'⚠ Low Stories\n{len(problematic_stories)}', ha='center', va='center', 
                    fontsize=11, fontweight='bold', color='#721c24')
            
            # Epics summary
            epic_good_rect = plt.Rectangle((5.5, summary_y - box_height/2), box_width, box_height, 
                                          facecolor='#d4edda', edgecolor='#2ed573', linewidth=2)
            ax3.add_patch(epic_good_rect)
            ax3.text(6.4, summary_y, f'✓ Good Epics\n{good_epics}/{total_epics}', ha='center', va='center', 
                    fontsize=11, fontweight='bold', color='#155724')
            
            epic_issue_rect = plt.Rectangle((7.5, summary_y - box_height/2), box_width, box_height, 
                                           facecolor='#f8d7da', edgecolor='#ff4757', linewidth=2)
            ax3.add_patch(epic_issue_rect)
            ax3.text(8.4, summary_y, f'⚠ Low Epics\n{len(problematic_epics)}', ha='center', va='center', 
                    fontsize=11, fontweight='bold', color='#721c24')
            
            # Details section
            current_y = summary_y - 1.5
            
            # Show problematic epics first
            if problematic_epics:
                ax3.text(5, current_y, 'Low Exhaustiveness Epics', fontsize=14, fontweight='bold', 
                        ha='center', va='center', color='#495057')
                current_y -= 0.8
                
                card_width = 4.5
                card_height = 1.2 # Increased from 1.0
                start_x = 0.25
                cols = 2
                
                for idx, epic_data in enumerate(problematic_epics):
                    row = idx // cols
                    col = idx % cols
                    x = start_x + col * (card_width + 0.5)
                    # Increased spacing multiplier from 0.3 to 0.4
                    y = current_y - row * (card_height + 0.4)
                    
                    card_rect = plt.Rectangle((x, y - card_height), card_width, card_height, 
                                            facecolor='white', edgecolor='#ff6b6b', 
                                            linewidth=2, alpha=0.9)
                    ax3.add_patch(card_rect)
                    
                    epic_text = epic_data['name'][:35] + '...' if len(epic_data['name']) > 35 else epic_data['name']
                    ax3.text(x + 0.1, y - 0.2, f"Epic: {epic_text}", fontsize=10, fontweight='bold', 
                            color='#495057', va='top', ha='left')
                    
                    ax3.text(x + 0.1, y - 0.5, 
                            f"Precision: {epic_data['precision']:.3f} | R-2 F1: {epic_data['rouge_2_f1']:.3f}", 
                            fontsize=9, color='#6c757d', va='top', ha='left')
                    
                    ax3.text(x + 0.1, y - 0.8, f"● Low Precision", 
                            fontsize=10, fontweight='bold', color='#ff6b6b', va='top', ha='left')
                
                current_y = y - card_height - 1.0
            
            # Show problematic stories
            if problematic_stories:
                ax3.text(5, current_y, 'Low Exhaustiveness Stories', fontsize=14, fontweight='bold', 
                        ha='center', va='center', color='#495057')
                current_y -= 0.8
                
                card_width = 4.5
                card_height = 1.2
                start_x = 0.25
                cols = 2
                
                for idx, story_data in enumerate(problematic_stories):
                    row = idx // cols
                    col = idx % cols
                    x = start_x + col * (card_width + 0.5)
                    y = current_y - row * (card_height + 0.4)
                    
                    card_rect = plt.Rectangle((x, y - card_height), card_width, card_height, 
                                            facecolor='white', edgecolor='#ff6b6b', 
                                            linewidth=2, alpha=0.9)
                    ax3.add_patch(card_rect)
                    
                    epic_text = story_data['epic'][:30] + '...' if len(story_data['epic']) > 30 else story_data['epic']
                    ax3.text(x + 0.1, y - 0.2, f"Epic: {epic_text}", fontsize=9, fontweight='bold', 
                            color='#6c757d', va='top', ha='left')
                    
                    story_text = story_data['user_story'][:60] + '...' if len(story_data['user_story']) > 60 else story_data['user_story']
                    ax3.text(x + 0.1, y - 0.5, story_text, fontsize=8, color='#495057', 
                            va='top', ha='left')
                    
                    ax3.text(x + 0.1, y - 0.9, 
                            f"Precision: {story_data['precision']:.3f} | R-2 F1: {story_data['rouge_2_f1']:.3f}", 
                            fontsize=8, color='#6c757d', va='top', ha='left')
                    
                    ax3.text(x + card_width - 0.1, y - 0.1, f"#{story_data['index']}", 
                            fontsize=11, fontweight='bold', color='#ff6b6b', 
                            va='top', ha='right')
            
            # All good message
            if not problematic_stories and not problematic_epics:
                ax3.text(5, current_y, '✓ All epics and stories meet exhaustiveness standards!', 
                        fontsize=16, fontweight='bold', ha='center', va='center', color='#2ed573',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='#d4edda', edgecolor='#2ed573'))
            
            # Footer
            footer_y = 0.3
            if problematic_stories or problematic_epics:
                footer_text = f'Showing {len(problematic_epics)} problematic epics and {len(problematic_stories)} problematic stories'
            else:
                footer_text = 'All items meet quality standards'
            ax3.text(5, footer_y, footer_text, 
                    fontsize=10, ha='center', va='center', color='#6c757d', style='italic')
            
            plt.show()

        return {
            "story_subscores": story_subscores,
            "epic_subscores": epic_subscores,
            "backlog_score": backlog_dict
        }