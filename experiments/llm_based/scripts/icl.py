from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
import os
from pyprojroot import here
import sys, time
from random import randint

os.chdir("/home/mortega/persisted/coeur-cohesion-and-exhaustiveness-of-user-story-representations")
sys.path.append("/home/mortega/persisted/coeur-cohesion-and-exhaustiveness-of-user-story-representations")

from coeur.score import Coeur

import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1",
    api_version="2025-01-01-preview",
    temperature=0,
    max_tokens=2000,
    timeout=None,
    max_retries=2
)

class Interesting(BaseModel):
     interesting: bool = Field(
        description="Indicates whether the input text is valuable to write a user story or not. True if it is valuable, False otherwise."
    )

class UserStory(BaseModel):
    epic: str = Field(
        description="An epic is a high-level feature or capability of the system. Multiple user stories can be grouped under the same epic."
    )
    description: str = Field(
        min_length=20,
        max_length=500,
        pattern=r"^As a .+, I want .+, so that .+$",
        examples=[
            "As a user, I want to log in, so that I can access my profile.",
            "As a product owner, I want to prioritize the backlog, so that the team works on the most valuable tasks."
        ],
        description="A user story is a description of a feature from an end-user perspective." + \
            "It should follow the 'As a [role], I want [feature], so that [benefit]' format." + \
            "Additionnaly, it should be between 20 and 500 characters long." 
    )

def save_results(mean_results: pd.DataFrame, std_results: pd.DataFrame, dataset_name: str, strategy_name: str):
    results_dir = os.path.join('experiments', 'llm_validation', 'output', dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    mean_path = os.path.join(results_dir, f'mean_results_{strategy_name}.csv')
    std_path = os.path.join(results_dir, f'std_results_{strategy_name}.csv')

    mean_results.to_csv(mean_path, index=False)
    std_results.to_csv(std_path, index=False)
    print(f"Results saved to {results_dir}")

def load_results(dataset_name: str, strategy_name: str):
    results_dir = os.path.join('experiments', 'llm_validation', 'results', dataset_name)

    mean_path = os.path.join(results_dir, f'mean_results_{strategy_name}.csv')
    std_path = os.path.join(results_dir, f'std_results_{strategy_name}.csv')

    mean_results = pd.read_csv(mean_path)
    std_results = pd.read_csv(std_path)
    print(f"Results loaded from {results_dir}")
    return mean_results, std_results

def generate_B(R: str, B: pd.DataFrame, llm: AzureChatOpenAI, max_iter=5, check_prompt="", generation_prompt="") -> pd.DataFrame:
    chunk_size = max(1, len(R) // max_iter)
    
    R_chunks = [R[i: i+chunk_size] for i in range(0, len(R), chunk_size)]
    epics_titles = B.epic.unique()

    y_pred = []

    for i in tqdm(range(max_iter)):
        if check_prompt != "":
            interesting = llm.with_structured_output(Interesting).invoke(
                check_prompt.format(
                    specs_chunk=R_chunks[i%max_iter]
                )
            )
        else:
            interesting = Interesting(interesting=True)

        if not interesting.interesting:
            print(f"Specifications chunk {i+1} deemed uninteresting, skipping...")
            continue
        us = None
        while not us:
            try:
                us = llm.with_structured_output(UserStory).invoke(
                    generation_prompt.format(
                        specs_chunk=R_chunks[i],
                        epic_titles=epics_titles,
                        existing_user_stories="\n".join(["Epic: " + us["epic"] + "|| Story: " + us["user_story"] for us in y_pred])
                    )
                )
            except:
                print("OpenAI API Failed")
                time.sleep(0.5)
        
        y_pred.append({"epic": us.epic, "user_story": us.description})
    B_pred = pd.DataFrame(y_pred)
    return B_pred

def generate_multiple_B(R: str, seeds=[5], max_iter=5, check_prompt="", generation_prompt="") -> pd.DataFrame:
    all_B_preds = []
    for seed in seeds:
        print(f"Generating backlog with seed {seed}...")
        seeded_llm = AzureChatOpenAI(
            azure_deployment="gpt-4.1",
            api_version="2025-01-01-preview",
            temperature=0.7,
            max_tokens=2000,
            timeout=None,
            max_retries=2,
            seed=seed
        )
        B_pred = generate_B(R, B, seeded_llm, max_iter=max_iter, check_prompt=check_prompt, generation_prompt=generation_prompt)
        all_B_preds.append(B_pred)
    return all_B_preds

def iterative_coeur(B: pd.DataFrame, seed, cs_offset=10, sigma="ROUGE-2 Precision", l="s"):
    # Iterative Calculation
    results = []
    coeur = Coeur(random_state=seed, lemmatization=True, remove_stopwords=True, stemming=True,
              remove_re_se_stopwords=True)

    for i in tqdm(range(cs_offset, len(B) + 1)):
        current_backlog = B.iloc[:i]
        
        scores = coeur.score(R=R, B=current_backlog, sigma=sigma, l=l)
        scores['n_stories'] = i
        results.append(scores)
        
    # Create DataFrame from results
    df_results = pd.DataFrame(results)
    return df_results

def multiple_iterative_coeur(all_B_preds, seeds, cs_offset=10, sigma="ROUGE-2 Precision", l="s"):
    all_results = []
    for B_pred, seed in zip(all_B_preds, seeds):
        df_results = iterative_coeur(B_pred, seed, cs_offset=cs_offset, sigma=sigma, l=l)
        all_results.append(df_results)
    
    # Calculate mean and std deviation across different runs
    mean_results = pd.concat(all_results).groupby('n_stories').mean().reset_index()
    std_results = pd.concat(all_results).groupby('n_stories').std().reset_index()
    
    return mean_results, std_results

interesting_evaluator_prompt = """
Considering the following software specifications chunk, and the list of available epic story titles, 
determine if the chunk of specification that is provided sufficient information to write at least one user story from it.

Specifications Chunk:
{specs_chunk}

Return True if it is interesting, False otherwise.
"""

context_epic_prompt = """
Given the following software specifications chunk and the list of epic titles it is attached to you have two tasks :
    1. Determine which epic title the specifications chunk is best suited for.
    2. Generate a user story fitting the epic and the specifications chunk.
You must not try to encapsulate all the specifications and epic in a single user story, but rather focus on a specific feature or functionality that would be valuable to an end-user.

Specifications Chunk :
{specs_chunk}

List of Epic Titles :
{epic_titles}
"""

no_context_no_epic_prompt = """
Generate a user story based on the theme of your choice.
"""

context_no_epic_prompt = """
Given the following software specifications chunk, generate a user story that captures a specific feature or functionality described in the text.
You will also need to assign an appropriate epic title to this user story.

Specifications Chunk :
{specs_chunk}
"""

no_context_epic_prompt = """
Generate a user story that fits into one of the provided epic titles.

List of Epic Titles :
{epic_titles}

# Avoid redundant user stories with existing user stories that follows (format is "Epic: [epic] || Story: [story]"):
# {existing_user_stories}
"""

DATASET_NAME = "all" # "all" or "trident" or "retro" or "alfred"
N_SEEDS = 10
MAX_ITER = 50
CS_OFFSET = 5
SIGMA = "ROUGE-2 F1"
SIGMA_LEVEL = "b"

seeds = [randint(0, 10000) for _ in range(N_SEEDS)]
if DATASET_NAME == "all":
    all_dataset_names = ["trident", "retro", "alfred"]
else:
    all_dataset_names = [DATASET_NAME]

for current_dataset_name in all_dataset_names:
    specs_path = os.path.join('datasets', current_dataset_name, f'{current_dataset_name}_specs.pdf')
    backlog_path = os.path.join('datasets', current_dataset_name, f'{current_dataset_name}_backlog.csv')

    coeur = Coeur(random_state=42, lemmatization=True, remove_stopwords=True, stemming=True,
              remove_re_se_stopwords=True)
    R, B = coeur.load_data(ref_path=specs_path, cand_path=backlog_path, ref_mode="pdf", cand_mode="csv")

    for strategy_name, check_prompt, generation_prompt in [
        ("context_epic", interesting_evaluator_prompt, context_epic_prompt),
        ("no_context_no_epic", interesting_evaluator_prompt, no_context_no_epic_prompt),
        ("context_no_epic", interesting_evaluator_prompt, context_no_epic_prompt),
        ("no_context_epic", interesting_evaluator_prompt, no_context_epic_prompt)
    ]:
        print(f"Running strategy: {strategy_name}")
        all_B_preds = generate_multiple_B(R, seeds=seeds, max_iter=MAX_ITER, check_prompt=check_prompt, generation_prompt=generation_prompt)
        mean_results, std_results = multiple_iterative_coeur(all_B_preds, seeds, cs_offset=CS_OFFSET, sigma=SIGMA, l=SIGMA_LEVEL)
        save_results(mean_results, std_results, current_dataset_name, strategy_name)