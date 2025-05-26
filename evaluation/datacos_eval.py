import os
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import gradio as gr

from csi_models.ModelBase import ModelBase
from csi_models.ByteCoverModel import ByteCoverModel
from csi_models.CoverHunterModel import CoverHunterModel
from csi_models.LyricoverModel import LyricoverModel
from csi_models.RemoveModel import RemoveModel
from csi_models.DaTonalModel import DaTonalModel
from feature_extraction.feature_extraction import MFCCModel, SpectralCentroidModel
from evaluation.metrics import compute_mean_metrics_for_rankings


# i used here the absolute path to these files (as in wsl)
# it should be changed to relative path in a potential fix
# now i cant do it as im doing the evaluation in the background
# but i suppose it should be like this:
# /datasets/datacos/da_tacos_listed.csv
# /datasets/datacos/npyfolder
DATACOS_CSV_PATH ="/mnt/d/WimuProj/CoverDetectionHub/datasets/datacos/da_tacos_listed.csv"  
DATACOS_FEATURE_DIR = "/mnt/d/WimuProj/CoverDetectionHub/datasets/datacos/npyfolder"  




def gather_datacos_files_from_csv(csv_path: str, feature_dir: str):
    df = pd.read_csv(csv_path)
    files_and_labels = []

    for _, row in df.iterrows():
        track_id = row["id"]
        clique = row["clique"]
        feature_path = os.path.join(feature_dir, f"{track_id}.npy")
        if not os.path.exists(feature_path):
            logging.warning(f"Missing feature file: {feature_path}")
            continue
        files_and_labels.append((feature_path, clique))

    return files_and_labels



def load_embeddings(files_and_labels, progress=gr.Progress()):
    progress(0, desc="Loading feature embeddings")
    embeddings = {}
    total_files = len(files_and_labels)

    for i, (feat_path, _) in enumerate(tqdm(files_and_labels, desc="Loading features")):
        if feat_path not in embeddings:
            embeddings[feat_path] = np.load(feat_path,allow_pickle=True).item()
        progress((i + 1) / total_files, desc="Loading feature embeddings")

    return embeddings


# ranking
def compute_rankings_per_song(files_and_labels, model: ModelBase, progress=gr.Progress()):
    embeddings = load_embeddings(files_and_labels, progress)
    rankings_per_query = []
    total_files = len(files_and_labels)
    progress(0, desc="Computing rankings")

    for i, (query_path, query_label) in enumerate(tqdm(files_and_labels, desc="Ranking")):
        query_embedding = embeddings[query_path]
        comparisons = []

        for j, (cand_path, cand_label) in enumerate(files_and_labels):
            if i == j:
                continue
            cand_embedding = embeddings[cand_path]
            sim = model.compute_similarity(query_embedding, cand_embedding)
            ground_truth = (query_label == cand_label)
            comparisons.append({
                "candidate_path": cand_path,
                "similarity": sim,
                "ground_truth": ground_truth
            })

        comparisons.sort(key=lambda x: x["similarity"], reverse=True)

        rankings_per_query.append({
            "query_path": query_path,
            "query_label": query_label,
            "ranking": comparisons
        })

        progress((i + 1) / total_files, desc="Comparing query to all others")

    return rankings_per_query


# "main" eval function
# I did the evaluation on 1000 files
# its way less than the max amount of potential files
# few reasons for that:
# 1) low disk space - max that i could save were +4000 npy files (no more space on disk D)
# 2) low disk space II - with +4000 npy files, my disk C was burning as there is the WSL installed
# and during the ranking it uses its space, for the disk C initially it was breaking at 20%, i could increase that to ~~30%
# getting the remaining 70% was impossible for me, thats why i reduced the amount of files to just 1000
# 
def evaluate_on_datacos(model_name: str, k=10):
    logging.info(f"Evaluating '{model_name}' on Da-TACOS benchmark with k={k}")
    files_and_labels = gather_datacos_files_from_csv(DATACOS_CSV_PATH, DATACOS_FEATURE_DIR)

    model_mapping = {
        "ByteCover": ByteCoverModel,
        "CoverHunter": CoverHunterModel,
        "Lyricover": LyricoverModel,
        "MFCC": MFCCModel,
        "Spectral Centroid": SpectralCentroidModel,
        "Remove": RemoveModel,
        "DaTonal": DaTonalModel
    }

    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model_mapping[model_name]()
    rankings = compute_rankings_per_song(files_and_labels, model)
    metrics = compute_mean_metrics_for_rankings(rankings, k=k)

    return {
        "Model": model_name,
        "Dataset": "Da-TACOS",
        "Mean Average Precision (mAP)": metrics["mAP"],
        f"Precision at {k} (mP@{k})": metrics["mP@k"],
        "Mean Rank of First Correct Cover (mMR1)": metrics["mMR1"]
    }
