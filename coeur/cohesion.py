import pandas as pd

pd.options.mode.chained_assignment = None

from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from umap import UMAP
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, \
                            calinski_harabasz_score, \
                            davies_bouldin_score, \
                            rand_score, \
                            adjusted_rand_score, \
                            normalized_mutual_info_score, \
                            adjusted_mutual_info_score, \
                            homogeneity_completeness_v_measure, \
                            fowlkes_mallows_score
from itertools import count, product
import warnings

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

RE_SE_STOPWORDS = ["as", "a", "i", "want", "to", "so", "that", "i", "the", "and", 
                    "in", "is", "it", "of", "for", "on", "with", 
                    "this", "by", "an", "be", "are", "at", "from"]

class CohesionAbstract:
    def __init__(self, embedder = None, clusterer = None, reducer = None,
                 stemming: bool = False, lemmatization: bool = False, 
                 remove_stopwords: bool = False, remove_re_se_stopwords: bool = False,
                 include_ac: bool = False, random_state: int = 56):
        self.embedder = embedder if embedder is not None else SentenceTransformer("all-mpnet-base-v2")
        self.random_state = random_state
        self.clusterer = clusterer if clusterer is not None else HDBSCAN()
        if hasattr(self.clusterer, 'random_state'):
            self.clusterer.random_state = self.random_state
        self.reducer = reducer if reducer is not None else UMAP(random_state=self.random_state)
        if hasattr(self.reducer, 'random_state'):
            self.reducer.random_state = self.random_state
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.remove_stopwords = remove_stopwords
        self.remove_re_se_stopwords = remove_re_se_stopwords
        self.include_ac = include_ac

    @staticmethod
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
    
    @staticmethod
    def unwind_ac(data: pd.DataFrame) -> pd.DataFrame:
        df_new = data.copy()
        df_new["type"] = "us"
        
        ac = data.copy()
        ac["type"] = "ac"
        ac["item"] = ac["acceptance_criteria"]
        ac = ac.explode("item")
        ac.drop(columns=["user_story"], inplace=True)
        ac.rename(columns={"item": "user_story"}, inplace=True)
        
        df_new = pd.concat([df_new, ac], ignore_index=True)
        return df_new

    def compute_epic_embeddings(self, data, epics):
            epic_embeddings = self.embedder.encode(epics, convert_to_tensor=True)
            epic_dict = {epic: emb.tolist() for epic, emb in zip(epics, epic_embeddings)}
            return data["epic"].map(epic_dict)
        
    def compute_story_embeddings(self, data):
        en_stop_words = set(stopwords.words('english'))
        fr_stop_words = set(stopwords.words('french'))
        stop_words = en_stop_words.union(fr_stop_words)
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
        
        if self.include_ac:
            stories = data["user_story"].copy()
            acs = data["acceptance_criteria"].copy().fillna("")
            stories = [f"{story}"+"\n".join(ac) if ac else story for story, ac in zip(stories, acs)]
        else:
            stories = data["user_story"].copy()
        for i, story in enumerate(stories):
            tokens = word_tokenize(story.lower())
            if self.remove_stopwords:
                filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            else:
                filtered_tokens = [word for word in tokens if word.isalnum()]
            if self.remove_re_se_stopwords:
                filtered_tokens = [word for word in filtered_tokens if word not in RE_SE_STOPWORDS]
            if self.lemmatization:
                filtered_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in filtered_tokens]
            if self.stemming:
                filtered_tokens = [stemmer.stem(word) for word in filtered_tokens]
            stories.iat[i] = " ".join(filtered_tokens)

        story_embeddings = self.embedder.encode(stories.tolist(), convert_to_tensor=True)
        return story_embeddings.tolist()
    
    def compute_labels(self, data, epics):
        dict_epic_label = {epic: i for i, epic in enumerate(epics)}
        epic_labels = data["epic"].map(dict_epic_label)
        return epic_labels
    
    def compute_clusters(self, data):
        self.clusterer.fit(np.array(data["story_embedding"].tolist()))
        cluster_labels = np.asarray(self.clusterer.labels_)
        return cluster_labels

    def preprocess(self, data):        
        data.reset_index(drop=True, inplace=True)
        epics = data["epic"].unique()
        data.loc[:, "epic_embedding"] = self.compute_epic_embeddings(data, epics)
        data["story_embedding"] = self.compute_story_embeddings(data)
        data["cluster_label"] = self.compute_clusters(data)
        data["epic_label"] = self.compute_labels(data, epics)
        self.preprocessed_data = data
        return data

class CohesionScore(CohesionAbstract):
    def __init__(self, embedder = None, clusterer = None,
                 stemming: bool = False, lemmatization: bool = False, 
                 remove_stopwords: bool = False, remove_re_se_stopwords: bool = False,
                 include_ac: bool = False, random_state: int = 56):
        super().__init__(embedder=embedder, clusterer=clusterer, 
                         stemming=stemming, lemmatization=lemmatization,
                         remove_stopwords=remove_stopwords, remove_re_se_stopwords=remove_re_se_stopwords,
                         include_ac=include_ac, random_state=random_state)

    def compute_metrics(self, data):
        mask = data["cluster_label"] != -1
        lbls_core = data["cluster_label"][mask]
        epics_core = data["epic_label"][mask]
        emb_core = np.array(data["story_embedding"].tolist())[mask]
        results = {}
        if len(np.unique(lbls_core)) < 2:
            print(f" Not enough clusters (>=2) after removing noise to compute metrics.")
            return results
        if emb_core.shape[0] < 2:
            print(f" Not enough samples after removing noise to compute metrics.")
            return results

        results["rand_index"] = rand_score(epics_core, lbls_core)
        results["adjusted_rand_index"] = adjusted_rand_score(epics_core, lbls_core)
        results["normalized_mutual_info"] = normalized_mutual_info_score(epics_core, lbls_core)
        results["adjusted_mutual_info"] = adjusted_mutual_info_score(epics_core, lbls_core)
        results["homogeneity"], results["completeness"], results["v_measure"] = homogeneity_completeness_v_measure(epics_core, lbls_core)
        results["fowlkes_mallows"] = fowlkes_mallows_score(epics_core, lbls_core)
        results["silhouette"] = silhouette_score(emb_core, lbls_core)
        results["calinski_harabasz"] = calinski_harabasz_score(emb_core, lbls_core)
        results["davies_bouldin"] = davies_bouldin_score(emb_core, lbls_core)


        unique, counts = np.unique(data["cluster_label"], return_counts=True)
        has_noise = -1 in unique
        n_clusters = unique.size - (1 if has_noise else 0)
        results["infos"] = {
            "Total points": len(data["cluster_label"]),
            "Unique labels": dict(zip(unique, counts)),
            "Number of clusters": n_clusters,
            "Noise points": counts[unique.tolist().index(-1)] if has_noise else 0
        }
        return results

    def fit(self, X):
        data = self.preprocess(X)
        metrics = self.compute_metrics(data)
        return metrics
    
    @staticmethod
    def compare(*arg, **kwargs):
        def highlight_best(s):
            """Highlight the best (green) and worst (red) value in each column"""
            best_color = 'background-color: seagreen'
            worst_color = 'background-color: darkred'
            if s.name in maximize_metrics:
                # For metrics where higher is better
                is_max = s == s.max()
                is_min = s == s.min()
                colors = []
                for i, v in enumerate(s):
                    if is_max.iloc[i]:
                        colors.append(best_color)
                    elif is_min.iloc[i]:
                        colors.append(worst_color)
                    else:
                        colors.append('')
                return colors
            elif s.name in minimize_metrics:
                # For metrics where lower is better
                is_max = s == s.max()
                is_min = s == s.min()
                colors = []
                for i, v in enumerate(s):
                    if is_min.iloc[i]:
                        colors.append(best_color)
                    elif is_max.iloc[i]:
                        colors.append(worst_color)
                    else:
                        colors.append('')
                return colors
            else:
                # For informational columns (Total Points, Number of Clusters, Noise Points)
                return ['' for _ in s]
        rows = []
        for model in arg:
            rows.append({
                "Rand Index": model["rand_index"],
                "Adjusted Rand Index": model["adjusted_rand_index"],
                "Normalized Mutual Info": model["normalized_mutual_info"],
                "Adjusted Mutual Info": model["adjusted_mutual_info"],
                "Homogeneity": model["homogeneity"],
                "Completeness": model["completeness"],
                "V-Measure": model["v_measure"],
                "Fowlkes-Mallows": model["fowlkes_mallows"],
                "Silhouette": model["silhouette"],
                "Calinski-Harabasz": model["calinski_harabasz"],
                "Davies-Bouldin": model["davies_bouldin"],
                "Total Points": model["infos"]["Total points"],
                "Number of Clusters": model["infos"]["Number of clusters"],
                "Noise Points": model["infos"]["Noise points"]
            })
        results = pd.DataFrame(rows)
        idx_names = kwargs.get("index_names", None)
        if idx_names is not None:
            results.index = idx_names
        
        if kwargs.get("highlight", False):
            maximize_metrics = [
                "Rand Index", "Adjusted Rand Index", "Normalized Mutual Info", 
                "Adjusted Mutual Info", "Homogeneity", "Completeness", 
                "V-Measure", "Fowlkes-Mallows", "Silhouette", "Calinski-Harabasz"
            ]
            minimize_metrics = ["Davies-Bouldin"]
            return results.round(2).style.apply(highlight_best, axis=0).format(precision=2)
        return results.round(2)

    def finetune(self, X, **kwargs):
        def grid_search(estimator, param_grid, x, y):
            
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            best_score = -1
            best_params = {}
            results = []
            for param_combo in product(*param_values):
                params = dict(zip(param_names, param_combo))            
                
                # Suppress warnings during clustering (common with bad parameters)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        current_clusterer = type(estimator)(**params)
                        current_clusterer.fit(x)
                        labels = current_clusterer.labels_
                        mask = labels != -1
                        if len(np.unique(labels[mask])) >= 2:
                            score = rand_score(y[mask], labels[mask])
                            results.append((params, score))
                            if score > best_score:
                                best_score = score
                                best_params = params
                                best_estimator = current_clusterer
                    except Exception as e:
                        # Skip parameter combinations that cause errors
                        print(f"Skipping params {params} due to error: {str(e)[:50]}...")
                        continue
                        
            return best_estimator, best_params, best_score, results

        X.reset_index(drop=True, inplace=True)
        epics = X["epic"].unique()
        counter = count()
        
        base_embedder = self.embedder
        base_clusterer = self.clusterer

        embedders = kwargs.get("embedders", [base_embedder])
        clusterers = kwargs.get("clusterers", [base_clusterer])
        if clusterers:
            clusterer_grids = kwargs.get("clusterer_grids", {})

        results = []
        for embedder in embedders:
            self.embedder = embedder
            X["epic_embedding"] = self.compute_epic_embeddings(X, epics)
            X["story_embedding"] = self.compute_story_embeddings(X)
            for (clusterer, clusterer_grid) in zip(clusterers, clusterer_grids):
                x = np.array(X["story_embedding"].tolist())
                y = self.compute_labels(X, epics)
                X["epic_label"] = y
                best_clusterer, best_params, _, _ = grid_search(clusterer, clusterer_grid, x, y)
                self.clusterer = best_clusterer
                X["cluster_label"] = self.compute_clusters(X)
                metrics = self.compute_metrics(X)
                results.append(
                    {
                        "embedder": self.embedder[0].auto_model.config.model_type,
                        "clusterer": type(clusterer).__name__,
                        "params": best_params,
                        "name": f"Model {next(counter)}",
                        "metrics": metrics
                    }
                )
        return results
    
class CohesionViz(CohesionAbstract):
    def __init__(self, embedder = None, clusterer = None, reducer = None,
                 stemming: bool = False, lemmatization: bool = False, 
                 remove_stopwords: bool = False, remove_re_se_stopwords: bool = False,
                 include_ac: bool = False, random_state: int = 56):
        super().__init__(embedder, clusterer, reducer,
                         stemming=stemming, lemmatization=lemmatization,
                         include_ac=include_ac, remove_stopwords=remove_stopwords,
                         remove_re_se_stopwords=remove_re_se_stopwords, random_state=random_state)
        self.preprocessed_data = pd.DataFrame()

    def reduce_dimensions(self, data):
        story_embeddings = data["story_embedding"].tolist()
        embs_2d = self.reducer.fit_transform(np.array(story_embeddings))
        return embs_2d.tolist()

    def format_acceptance_criteria(self, criteria, max_width=50):
        """Format acceptance criteria as bullet points with line breaks"""
        if any(pd.isna(criteria)) or criteria == "" or criteria is None:
            return "No acceptance criteria"
        
        # Convert to string and handle different formats
        criteria_str = str(criteria)
        
        # Split by common delimiters and clean up
        if isinstance(criteria, list):
            criteria_list = criteria
        else:
            # Try to split by common patterns
            import re
            criteria_list = re.split(r'[;\n\r]|(?:\d+\.)|(?:-\s)', criteria_str)
            criteria_list = [c.strip() for c in criteria_list if c.strip()]
        
        # Format as bullet points with word wrapping
        formatted_criteria = []
        for criterion in criteria_list:
            if len(criterion) > max_width:
                # Simple word wrapping
                words = criterion.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) <= max_width:
                        current_line += (" " + word) if current_line else word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                formatted_criteria.append("• " + "<br>  ".join(lines))
            else:
                formatted_criteria.append("• " + criterion)
        
        return "<br>".join(formatted_criteria)

    def format_user_story(self, story, max_width=60):
        """Format user story with line breaks for better readability"""
        if len(story) <= max_width:
            return story
        
        words = story.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) <= max_width:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        return "<br>".join(lines)

    def plot_epics(self, X, **kwargs):
        data = self.preprocess(X)
        data["story_embedding_2d"] = [list(coord) for coord in self.reduce_dimensions(data)]
        
        # Extract coordinates
        x_coords = [coord[0] for coord in data["story_embedding_2d"]]
        y_coords = [coord[1] for coord in data["story_embedding_2d"]]
        
        # Format data for hover
        formatted_stories = [self.format_user_story(story) for story in data['user_story']]
        if self.include_ac:
            formatted_criteria = [self.format_acceptance_criteria(criteria) for criteria in data['acceptance_criteria']]
        
        # Truncate epic names for legend
        max_legend_length = 15
        data['epic_display'] = data['epic'].apply(lambda x: x[:max_legend_length] + '...' if len(x) > max_legend_length else x)

        fig = px.scatter(
            x=x_coords, 
            y=y_coords,
            color=data['epic_display'],
            hover_name=data['epic'],
            custom_data=list(zip(formatted_stories, formatted_criteria)),
            title="Colored by Epic",
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
            width=800,
            height=600
        )
        
        # Update hover template with fixed width
        if self.include_ac:
            hovertemplate = "<b>Epic:</b> %{hovertext}<br>" + \
                            "<b>User Story:</b><br>%{customdata[0]}<br>" + \
                            "<b>Acceptance Criteria:</b><br>%{customdata[1]}<br>" + \
                            "<extra></extra>"
        else:
            hovertemplate = "<b>Epic:</b> %{hovertext}<br>" + \
                            "<b>User Story:</b><br>%{customdata[0]}<br>" + \
                            "<extra></extra>"
        fig.update_traces(
            hovertemplate=hovertemplate,
            hovertext=data['epic']
        )
        
        # Set hover box width limit
        fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font_size=12,
                font_family="Arial",
                align="left"
            )
        )
        
        if not kwargs.get("legend", True):
            fig.update_layout(showlegend=False)
            
        fig.show()

    def plot(self, X, **kwargs):
        data = self.preprocess(X)
        data["story_embedding_2d"] = [list(coord) for coord in self.reduce_dimensions(data)]
        
        # Format data for hover
        formatted_stories = [self.format_user_story(story) for story in data['user_story']]
        if self.include_ac:
            formatted_criteria = [self.format_acceptance_criteria(criteria) for criteria in data['acceptance_criteria']]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Colored by Epic", "Colored by Cluster"),
            horizontal_spacing=0.12  # Increased spacing for legend
        )
        
        # Epic coloring (left plot)
        epics = data["epic"].unique()
        colors_epic = px.colors.qualitative.Set3[:len(epics)]
        max_legend_length = 15
        
        for i, epic in enumerate(epics):
            epic_data = data[data['epic'] == epic]
            epic_indices = epic_data.index
            epic_x = [coord[0] for coord in epic_data["story_embedding_2d"]]
            epic_y = [coord[1] for coord in epic_data["story_embedding_2d"]]
            
            epic_formatted_stories = [formatted_stories[idx] for idx in epic_indices]
            if self.include_ac:
                epic_formatted_criteria = [formatted_criteria[idx] for idx in epic_indices]
            
            # Truncate epic name for legend
            epic_display = epic[:max_legend_length] + '...' if len(epic) > max_legend_length else epic
            
            if self.include_ac:
                hovertemplate = "<b>Epic:</b> %{text}<br>" + \
                                "<b>User Story:</b><br>%{customdata[0]}<br>" + \
                                "<b>Acceptance Criteria:</b><br>%{customdata[1]}<br>" + \
                                "<extra></extra>"
                customdata = list(zip(epic_formatted_stories, epic_formatted_criteria))
            else:
                hovertemplate = "<b>Epic:</b> %{text}<br>" + \
                                "<b>User Story:</b><br>%{customdata}<br>" + \
                                "<extra></extra>"
                customdata = list(epic_formatted_stories)

            fig.add_trace(
                go.Scatter(
                    x=epic_x,
                    y=epic_y,
                    mode='markers',
                    name=epic_display,  # Truncated name for legend
                    marker=dict(color=colors_epic[i % len(colors_epic)], size=8),
                    hovertemplate=hovertemplate,
                    text=[epic] * len(epic_data),  # Full epic name for hover
                    customdata=customdata,
                    legendgroup="epics",
                    showlegend=kwargs.get("legend1", True)
                ),
                row=1, col=1
            )
        
        # Cluster coloring (right plot)
        cluster_labels = data['cluster_label'].unique()
        colors_cluster = px.colors.qualitative.Plotly[:len(cluster_labels)]
        
        for i, cluster in enumerate(cluster_labels):
            cluster_data = data[data['cluster_label'] == cluster]
            cluster_indices = cluster_data.index
            cluster_x = [coord[0] for coord in cluster_data["story_embedding_2d"]]
            cluster_y = [coord[1] for coord in cluster_data["story_embedding_2d"]]
            
            cluster_formatted_stories = [formatted_stories[idx] for idx in cluster_indices]
            if self.include_ac:
                cluster_formatted_criteria = [formatted_criteria[idx] for idx in cluster_indices]
            
            if self.include_ac:
                hovertemplate = "<b>Epic:</b> %{customdata[2]}<br>" + \
                                "<b>Cluster:</b> %{text}<br>" + \
                                "<b>User Story:</b><br>%{customdata[0]}<br>" + \
                                "<b>Acceptance Criteria:</b><br>%{customdata[1]}<br>" + \
                                "<extra></extra>"
                customdata = list(zip(cluster_formatted_stories, cluster_formatted_criteria, cluster_data['epic']))
            else:
                hovertemplate = "<b>Epic:</b> %{customdata[1]}<br>" + \
                                "<b>Cluster:</b> %{text}<br>" + \
                                "<b>User Story:</b><br>%{customdata[0]}<br>" + \
                                "<extra></extra>"
                customdata = list(zip(cluster_formatted_stories, cluster_data['epic']))

            fig.add_trace(
                go.Scatter(
                    x=cluster_x,
                    y=cluster_y,
                    mode='markers',
                    name=f'Cluster {cluster}',
                    marker=dict(color=colors_cluster[i % len(colors_cluster)], size=8),
                    hovertemplate=hovertemplate,
                    text=[f'Cluster {cluster}'] * len(cluster_data),
                    customdata=customdata,
                    legendgroup="clusters",
                    showlegend=kwargs.get("legend2", True)
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="Dimension 1", row=1, col=1)
        fig.update_yaxes(title_text="Dimension 2", row=1, col=1)
        fig.update_xaxes(title_text="Dimension 1", row=1, col=2)
        fig.update_yaxes(title_text="Dimension 2", row=1, col=2)
        
        fig.update_layout(
            width=1300,  # Increased width to accommodate legend
            height=600,
            title_text="Requirements Clustering Visualization",
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font_size=12,
                font_family="Arial",
                align="left"
            ),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10),
                traceorder="normal",
                itemsizing="constant"
            )
        )
        
        fig.show()