import os, sys
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

import pandas as pd
import numpy as np
from coeur.score import Coeur
from coeur.baseline.qus.aqusacore import AQUSA
from coeur.baseline.usqa.usqa import USQA
import random
import json
from copy import deepcopy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from transformers.utils import logging
logging.set_verbosity_error()

#UTILITY FUNCTIONS
def encode_user_stories(df, model_name='all-mpnet-base-v2', batch_size=512):
    model = SentenceTransformer(model_name)
    user_stories = df['user_story'].tolist()
    embeddings = model.encode(user_stories, batch_size=batch_size)
    return embeddings

def shuffle_user_stories_between_sources(df_ref: pd.DataFrame, noise_level=0.5, ref_name="ref", noise_name="noise", include_ac=False,
                                  seed=123):
    def minimize_df(df: pd.DataFrame, name: str, has_embedding: bool = False) -> pd.DataFrame:
        new_df = df.copy()
        new_df['source'] = name
        new_df.reset_index(drop=True, inplace=True)
        cols_to_include = ['source', 'epic', 'user_story']

        if include_ac:
            cols_to_include.append('acceptance_criteria')
        if has_embedding:
            cols_to_include.append('embedding')
            if isinstance(new_df.at[0, 'embedding'], str):
                new_df['embedding'] = new_df['embedding'].apply(literal_eval)

        return new_df[cols_to_include]

    np.random.seed(seed)
    random.seed(seed)
    df_noise_dalpiaz = pd.read_csv("datasets/dalpiaz/noise_intended.csv")
    df_noise_neodataset = pd.read_csv("datasets/neodataset/noise_intended.csv")
    df_noise = pd.concat([df_noise_dalpiaz, df_noise_neodataset], ignore_index=True)

    len_ref = df_ref.shape[0]
    len_noise = df_noise.shape[0]
    n_user_stories_to_shuffle = min([int(len_ref * noise_level), len_noise])

    if n_user_stories_to_shuffle > 0:
        ref_has_embedding = "embedding" in df_ref.columns
        noise_has_embedding = "embedding" in df_noise.columns
        df_ref_noised = minimize_df(df_ref, ref_name, has_embedding=ref_has_embedding)
        if not ref_has_embedding:
            df_ref_noised["embedding"] = encode_user_stories(df_ref_noised, model_name='all-mpnet-base-v2', batch_size=512).tolist()
        df_noise = minimize_df(df_noise, noise_name, has_embedding=noise_has_embedding)
        if not noise_has_embedding:
            df_noise["embedding"] = encode_user_stories(df_noise, model_name='all-mpnet-base-v2', batch_size=512).tolist()
        user_stories_ref_index = df_ref_noised.index.tolist()
        user_stories_to_shuffle_indices_ref = np.random.choice(list(user_stories_ref_index), size=n_user_stories_to_shuffle, replace=False)
        cosim = cosine_similarity(
            np.vstack(df_ref_noised.loc[user_stories_to_shuffle_indices_ref, 'embedding'].values),
            np.vstack(df_noise['embedding'].values)
        )
        # Find the less similar user stories in noise for each selected user story in ref
        user_stories_to_shuffle_indices_noise = []
        already_selected = set()
        for i in range(cosim.shape[0]):
            sorted_indices = np.argsort(cosim[i, :])
            for idx in sorted_indices:
                if idx not in already_selected:
                    user_stories_to_shuffle_indices_noise.append(df_noise.index[idx])
                    already_selected.add(idx)
                    break
        
        # Swap user stories
        for ref_idx, noise_idx in zip(user_stories_to_shuffle_indices_ref, user_stories_to_shuffle_indices_noise):
            temp = df_ref_noised.at[ref_idx, 'user_story']
            df_ref_noised.at[ref_idx, 'user_story'] = df_noise.at[noise_idx, 'user_story']
            df_noise.at[noise_idx, 'user_story'] = temp
            df_ref_noised.at[ref_idx, 'source'] = noise_name
        final_df = df_ref_noised
    else:
        final_df = minimize_df(df_ref, ref_name)
    return final_df

def shuffle_user_stories_between_epics(df_ref, noise_level=0.5, ref_name="ref", include_ac=False,
                                       seed=123):
    def minimize_df(df, name):
        new_df = df.copy()
        new_df['source'] = name
        if include_ac:
            return new_df[['source', 'epic', 'user_story', 'acceptance_criteria']]
        else:
            return new_df[['source', 'epic', 'user_story']]
    np.random.seed(seed)
    random.seed(seed)
    id_to_epic = {i: v for i, v in enumerate(df_ref['epic'].unique())}
    n_user_stories_to_shuffle = int((df_ref.shape[0] * noise_level)/2)
    if n_user_stories_to_shuffle > 0:
        idx_source = np.random.choice(df_ref.index, size=n_user_stories_to_shuffle, replace=False)
        idx_target = []
        for idx in idx_source:
            current_epic = df_ref.loc[idx, 'epic']
            possible_epics = list(id_to_epic.values())
            possible_epics.remove(current_epic)
            new_epic = np.random.choice(possible_epics, size=1)[0]
            idx_candidates = df_ref.index[df_ref["epic"] == new_epic]
            idx_candidates_reduced = [i for i in idx_candidates if i not in idx_source and i not in idx_target]
            idx_candidates = idx_candidates_reduced if len(idx_candidates_reduced) > 0 else idx_candidates
            idx_chosen = np.random.choice(idx_candidates, size=1)[0]
            idx_target.append(idx_chosen)
        
        df_ref_noised = minimize_df(df_ref, ref_name)
        for idx_s, idx_t in zip(idx_source, idx_target):
            df_ref_noised.loc[[idx_s, idx_t], 'user_story'] = df_ref_noised.loc[[idx_t, idx_s], 'user_story'].values
    else:
        df_ref_noised = minimize_df(df_ref, ref_name)
    return df_ref_noised

def get_actual_information_level(df_noised, df_ref):
    return 1 - (df_noised["user_story"] == df_ref["user_story"]).sum() / len(df_ref)

def compute_coeur_with_incremental_noise(X_ref, df_ref, noise_levels=[0.0, 0.5, 1.0], 
                                         epic_level=True, story_level=True,
                                         include_ac=False, include_mauve=False,
                                         include_bleu=False, include_bleurt=False, include_meteor=False,
                                         bwise=True, ewise=False, swise=False,
                                         pca_components=50, non_linear_components=3,
                                         stemming=True, remove_stopwords=True, lemmatization=True,
                                         remove_re_se_stopwords=False, include_aqusa=False,
                                         include_usqa=False, batch_size=128, seed=123):
    target_noise_levels = noise_levels
    actual_epic_noise_levels = []
    actual_story_noise_levels = []
    coherence_scores = []
    coverage_scores = []
    if include_aqusa:
        aqusa_scores = []
    if include_usqa:
        usqa_scores = []

    coeur = Coeur(random_state=seed, batch_size=batch_size, include_ac=include_ac,
                  stemming=stemming, remove_stopwords=remove_stopwords, lemmatization=lemmatization,
                  remove_re_se_stopwords=remove_re_se_stopwords, pca_components=pca_components,
                  non_linear_components=non_linear_components)
    for tnl in target_noise_levels:
        df_ref_noised = df_ref.copy()
        if epic_level:
            df_ref_noised = shuffle_user_stories_between_sources(df_ref_noised, noise_level=tnl, include_ac=include_ac,
                                                            seed=seed)
        if story_level:
            df_ref_noised = shuffle_user_stories_between_epics(df_ref_noised, noise_level=tnl, include_ac=include_ac,
                                                            seed=seed)
        actual_noise = get_actual_information_level(df_ref_noised, df_ref)
        if epic_level:
            actual_epic_noise_levels.append(actual_noise)
            actual_story_noise_levels.append(0.0)
        if story_level:
            actual_story_noise_levels.append(actual_noise)
            actual_epic_noise_levels.append(0.0)

        coeur.X = X_ref
        coeur.X_hat = df_ref_noised
        coherence_score = coeur.score_coherence(highlight=False)
        coverage_score = coeur.score_coverage(include_mauve=include_mauve, include_bleu=include_bleu,
                                              include_bleurt=include_bleurt, include_meteor=include_meteor,
                                              backlog_wise=bwise, epic_wise=ewise, story_wise=swise, verbose=False)
        if include_aqusa:
            aqusa = AQUSA(df_ref_noised['user_story'].tolist()).compute()
        if include_usqa:
            usqa = USQA(df_ref_noised['user_story'].tolist()).compute()

        coherence_score["noise_level"] = tnl
        coverage_score["noise_level"] = tnl
        coherence_scores.append(coherence_score)
        coverage_scores.append(coverage_score)
        if include_aqusa:
            aqusa_scores.append(aqusa)
        if include_usqa:
            usqa_scores.append(usqa)
            

    result = {
        "coherence_scores": pd.concat(coherence_scores),
        "coverage_scores": pd.concat(coverage_scores),
        "aqusa_scores": np.array(aqusa_scores) if include_aqusa else None,
        "usqa_scores": np.array(usqa_scores) if include_usqa else None,
        "target_noise_levels": target_noise_levels,
        "actual_epic_information_levels": 1 - np.array(actual_epic_noise_levels),
        "actual_story_information_levels": 1 - np.array(actual_story_noise_levels) if story_level else actual_story_noise_levels
    }
    return result

def split_coeur_results(results, coherence=True, coverage=True,
                        bwise_coverage=True, ewise_coverage=True, swise_coverage=True,
                        coherence_subcols="default", coverage_subcols="F1", include_mauve=False,
                        include_bleu=False, include_bleurt=False, include_meteor=False,
                        include_aqusa=False, include_usqa=False, coverage_metric=None):
    if coherence:
        if isinstance(coherence_subcols, str) and coherence_subcols == "all":
            small_concat_coherence_scores = results["coherence_scores"].drop(columns=["Total Points", "Number of Clusters", "Noise Points", "noise_level"])
        elif isinstance(coherence_subcols, list):
            small_concat_coherence_scores = results["coherence_scores"][coherence_subcols]
        elif isinstance(coherence_subcols, str) and coherence_subcols == "default":
            small_concat_coherence_scores = results["coherence_scores"][
                ["Adjusted Rand Index", "Normalized Mutual Info",
                "V-Measure", "Fowlkes-Mallows"]]
        model_wise_coherence = {}
        for coherence_index in range(small_concat_coherence_scores.shape[0]):
            model_wise_coherence[small_concat_coherence_scores.index[coherence_index]] = small_concat_coherence_scores[
            small_concat_coherence_scores.index == small_concat_coherence_scores.index[coherence_index]]
        
        results["model_wise_coherence"] = model_wise_coherence

    if coverage:
        if not coverage_metric:
            metrics = ["BERTScore", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
        elif isinstance(coverage_metric, list):
            metrics = coverage_metric
        if isinstance(coverage_subcols, str) and coverage_subcols == "F1":
            coverage_cols = [m+" F1" for m in metrics]
        elif isinstance(coverage_subcols, str) and coverage_subcols == "P":
            coverage_cols = [m+" Precision" for m in metrics]
        elif isinstance(coverage_subcols, str) and coverage_subcols == "R":
            coverage_cols = [m+" Recall" for m in metrics]
        elif isinstance(coverage_subcols, str) and coverage_subcols == "all":
            coverage_cols = []
            for m in metrics:
                coverage_cols.extend([m+" Precision", m+" Recall", m+" F1"])
        if include_mauve:
            coverage_cols.extend(["MAUVE", "MAUVE (Star)"])
        if include_bleu:
            coverage_cols.append("BLEU")
        if include_bleurt:
            coverage_cols.append("BLEURT")
        if include_meteor:
            coverage_cols.append("METEOR")
        small_concat_coverage_scores = results["coverage_scores"][coverage_cols]
        if bwise_coverage:
            bwise_coverage = small_concat_coverage_scores[small_concat_coverage_scores.index == "Backlog-wise"]
            results["bwise_coverage"] = bwise_coverage
        if ewise_coverage:
            ewise_coverage = small_concat_coverage_scores[small_concat_coverage_scores.index.str.contains("Epic-wise")]
            results["ewise_coverage"] = ewise_coverage
        if swise_coverage:
            swise_coverage = small_concat_coverage_scores[small_concat_coverage_scores.index.str.contains("Story-wise")]
            results["swise_coverage"] = swise_coverage
        if not include_aqusa:
            results.pop("aqusa_scores", None)
        if not include_usqa:
            results.pop("usqa_scores", None)
    return results

def save_results(results, main_folder_path, experiment_type):
    results = deepcopy(results)
    current_path = os.path.join(main_folder_path, experiment_type)
    folder_index = len([f for f in os.listdir(current_path) if f.startswith("run_")]) + 1
    folder_path = os.path.join(current_path, f"run_{folder_index}.json")

    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            value = value.reset_index()
            results[key] = value.to_json(indent=4)
        elif key == "model_wise_coherence":
            for model_name, df in value.items():
                results[key][model_name] = df.reset_index().to_json(indent=4)
        elif isinstance(value, np.ndarray):
            results[key] = value.tolist()
    with open(folder_path, "w") as f:
        json.dump({"results": results}, f, indent=4)

#HYPERPARAMETERS (User can modify)
DATASET_NAME = "all" # "all" or "retro" or "trident" or "alfred"
EXPERIMENT_NAME = "improved_noise"
N_SEEDS = 10
N_NOISE_LEVELS = 30
INCLUDE_AC = False
INCLUDE_MAUVE = False
INCLUDE_BLEU = False
INCLUDE_BLEURT = False
INCLUDE_METEOR = False
INCLUDE_AQUSA = True
INCLUDE_USQA = True
BWISE = True
EWISE = True
SWISE = True
COHERENCE_SUBCOLS = "all"
COVERAGE_SUBCOLS = "all"
COVERAGE_METRICS = ["BERTScore", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Exhaustiveness"]
PCA_COMPONENTS = 50
NON_LINEAR_COMPONENTS = 2
LEMMATIZATION = True
STEMMING = True
REMOVE_RE_SE_STOPWORDS = True
REMOVE_STOPWORDS = True
BATCH_SIZE = 64

#GLOBAL VARIABLES (User should not modify)
all_available_datasets = ["retro", "trident", "alfred"]
backlog_path = "datasets/{0}/{0}_backlog.csv"
specs_path = "datasets/{0}/{0}_specs.pdf"
output_folder_path = "experiments/output/{1}/"
seeds = [i for i in range(N_SEEDS)]
noise_levels = np.linspace(0.0, 1.0, N_NOISE_LEVELS)

#RUN EXPERIMENTS
if isinstance(DATASET_NAME, str):
    if DATASET_NAME == "all":
        DATASET_NAME = all_available_datasets
    else:
        DATASET_NAME = [DATASET_NAME]
init_coeur = Coeur()
pbar = tqdm(total=len(seeds)*len(DATASET_NAME)*2, desc="Experiments: ")
for dataset_name in DATASET_NAME:
    current_backlog_path = backlog_path.format(dataset_name)
    current_specs_path = specs_path.format(dataset_name)
    current_output_folder_path = output_folder_path.format(dataset_name)
    specs_ref, backlog_ref = init_coeur.load_data(
        current_specs_path,
        current_backlog_path
    )
    for seed in seeds:
        for epic_level in [False, True]:
            for story_level in [False, True]:
                if (epic_level and story_level) or not(epic_level or story_level): #noise are studied separately
                    continue
                X_ref = specs_ref
                df_ref = backlog_ref
                results = compute_coeur_with_incremental_noise(
                    X_ref, df_ref, noise_levels=noise_levels,
                    epic_level=epic_level, story_level=story_level,
                    include_ac=INCLUDE_AC, include_mauve=INCLUDE_MAUVE,
                    include_bleu=INCLUDE_BLEU, include_bleurt=INCLUDE_BLEURT, 
                    include_meteor=INCLUDE_METEOR,
                    bwise=BWISE, ewise=EWISE, swise=SWISE, batch_size=BATCH_SIZE,
                    pca_components=PCA_COMPONENTS, non_linear_components=NON_LINEAR_COMPONENTS,
                    stemming=STEMMING, remove_stopwords=REMOVE_STOPWORDS, lemmatization=LEMMATIZATION,
                    remove_re_se_stopwords=REMOVE_RE_SE_STOPWORDS, seed=seed, include_aqusa=INCLUDE_AQUSA,
                    include_usqa=INCLUDE_USQA
                )
                results = split_coeur_results(
                    results,
                    coherence=True,
                    coverage=True,
                    bwise_coverage=BWISE,
                    ewise_coverage=EWISE,
                    swise_coverage=SWISE,
                    coherence_subcols=COHERENCE_SUBCOLS,
                    coverage_subcols=COVERAGE_SUBCOLS,
                    include_mauve=INCLUDE_MAUVE,
                    include_bleu=INCLUDE_BLEU,
                    include_bleurt=INCLUDE_BLEURT,
                    include_meteor=INCLUDE_METEOR,
                    include_aqusa=INCLUDE_AQUSA,
                    include_usqa=INCLUDE_USQA,
                    coverage_metric=COVERAGE_METRICS
                )

                if epic_level and not story_level:
                    experiment_type = "epic"
                elif not epic_level and story_level:
                    experiment_type = "story"
                else:
                    raise ValueError("Invalid experiment configuration.")
                
                results["hyperparameters"] = {
                    "dataset_name": dataset_name,
                    "include_ac": INCLUDE_AC,
                    "include_mauve": INCLUDE_MAUVE,
                    "include_bleu": INCLUDE_BLEU,
                    "include_bleurt": INCLUDE_BLEURT,
                    "include_meteor": INCLUDE_METEOR,
                    "include_aqusa": INCLUDE_AQUSA,
                    "include_usqa": INCLUDE_USQA,
                    "bwise": BWISE,
                    "ewise": EWISE,
                    "swise": SWISE,
                    "coherence_subcols": COHERENCE_SUBCOLS,
                    "coverage_subcols": COVERAGE_SUBCOLS,
                    "coverage_metrics": COVERAGE_METRICS,
                    "pca_components": PCA_COMPONENTS,
                    "non_linear_components": NON_LINEAR_COMPONENTS,
                    "lemmatization": LEMMATIZATION,
                    "stemming": STEMMING,
                    "remove_stopwords": REMOVE_STOPWORDS,
                    "remove_re_se_stopwords": REMOVE_RE_SE_STOPWORDS,
                    "batch_size": BATCH_SIZE,
                    "seed": seed,
                    "epic_level": epic_level,
                    "story_level": story_level
                }
                save_results(results, current_output_folder_path, experiment_type)
                pbar.update(1)
pbar.close()