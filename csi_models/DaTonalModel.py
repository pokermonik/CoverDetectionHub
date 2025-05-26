import torch
import torch.nn.functional as F
import numpy as np
from csi_models.DaTonalCover.model import *
from csi_models.ModelBase import ModelBase
from csi_models.DaTonalCover import *

class DaTonalModel(ModelBase):
    # 8 input features (hpcp, crema, tempos, key, etc.)
    def __init__(self, model_path="csi_models/checkpoints/DaTonal/DaTonalCover.pth", input_size=8, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        self.model = DaTonalCover(input_size=self.input_size)
        self.model.nn_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.nn_model.to(self.device)
        self.model.nn_model.eval()

    def compute_embedding(self, audio_file_path):
        from csi_models.DaTonalCover.compare_two_songs import extract_all_features_from_mp3
        
        return extract_all_features_from_mp3(audio_file_path)
       

    def compute_similarity(self, embedding1, embedding2):
        from csi_models.DaTonalCover.compare_two_songs import build_feature_vector
        feature_vector = build_feature_vector(embedding1, embedding2)
        input_tensor = torch.tensor(feature_vector, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            prediction = self.model.nn_model(input_tensor).item()

        return prediction

    def compute_similarity_between_files(self, file1, file2):
        try:
            emb1 = self.compute_embedding(file1)
            emb2 = self.compute_embedding(file2)
            return self.compute_similarity(emb1, emb2)
        except Exception as e:
            print(f"[ERROR] Failed to compute similarity: {e}")
            return 0.0
