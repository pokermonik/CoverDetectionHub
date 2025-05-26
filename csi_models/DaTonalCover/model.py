import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import pandas as pd
import random
from itertools import combinations
def to_scalar(val):
    #print(val)
    arr = np.asarray(val)
    if arr.ndim == 0:
        return float(arr)
    elif arr.size == 1:
        return float(arr[0])
    else:
        raise ValueError(f"Expected scalar or length-1 array, got shape {arr.shape}")
def safe_scalar(x):
    try:
        return float(np.asscalar(x)) if hasattr(x, 'shape') and x.shape != () else float(x)
    except Exception:
        return float(x[0]) if hasattr(x, '__getitem__') else float(x)

def normalize_similarity_vector(sims_raw):
    
    sims = np.array(sims_raw, dtype=np.float32)

    if sims.shape[0] != 7:
        raise ValueError(f"Expected 8 similarity features, got shape {sims.shape}")

    # Normalize strength difference to similarity (1 - diff)
    sims[5] = 1.0 - np.clip(sims[5], 0.0, 1.0)

    # Normalize tempo difference to similarity (1 - diff/100)
    sims[6] = 1.0 - np.clip(sims[6] / 100.0, 0.0, 1.0)

    # remaining errors (NaNs, infs)
    sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)

    return sims.reshape(1, -1)


class DaTonalCoverNN(nn.Module):
    def __init__(self, input_size=7):
        super(DaTonalCoverNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

    
    
    

class DaTonalCover:
    # 7 input features
    # before it was 8 but i combine scale and key into one sim now 
    # 1) hpcp 2) chroma_cens 3) mfcc_htk 4) crema KEY features:[5  combined key and scale  6strength] 7) madmom features *only* tempos,
    def __init__(self, instrumental_threshold=8, input_size=7):
        self.instrumental_threshold = instrumental_threshold
        self.nn_model = DaTonalCoverNN(input_size=input_size)
        self.is_model_loaded = False

    def compute_similarity_features(self, f1, f2):
        # Compute cosine similarity between mean vectors of each feature
        return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
    


    # i take key and scale to make one feature out of it
    # before it was just straight binary
    # key2==key1 1:0 etc
    # this is better as it gives more depth now
    #
    def key_similarity(self, key1, scale1, key2, scale2):

        KEY_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        RELATIVE_MAJOR_MINOR = {
            'C': 'A', 'G': 'E', 'D': 'B', 'A': 'F#', 'E': 'C#', 'B': 'G#',
            'F#': 'D#', 'C#': 'A#', 'F': 'D', 'Bb': 'G', 'Eb': 'C', 'Ab': 'F'
        }
        # Invert for minor → major
        RELATIVE_MAJOR_MINOR.update({v: k for k, v in RELATIVE_MAJOR_MINOR.items()})
        """Graded similarity score between two keys and scales."""
        key1, key2 = key1.upper(), key2.upper()
        scale1, scale2 = scale1.lower(), scale2.lower()

        if key1 == key2 and scale1 == scale2:
            return 1.0

        # Relative major/minor
        if (RELATIVE_MAJOR_MINOR.get(key1) == key2 and scale1 != scale2) or \
        (RELATIVE_MAJOR_MINOR.get(key2) == key1 and scale1 != scale2):
            return 0.7

        try:
            idx1 = KEY_CLASSES.index(key1)
            idx2 = KEY_CLASSES.index(key2)
        except ValueError:
            return 0.0  # Unknown key

        semitone_distance = abs(idx1 - idx2) % 12

        if semitone_distance == 1:
            return 0.6  # half-step
        elif semitone_distance == 7:
            return 0.8  # perfect fifth
        elif semitone_distance == 6:
            return 0.4  # tritone

        # Same pitch but different scale
        if key1 == key2 and scale1 != scale2:
            return 0.6

        return 0.2  # weakly related

    def extract_combined_features(self, pair_list, feature_base_path):
        features, labels = [], []

        for pid1, pid2, label in pair_list:
            f1_path = self.find_h5_file(pid1, feature_base_path)
            f2_path = self.find_h5_file(pid2, feature_base_path)
            if not f1_path or not f2_path:
                print(f"[SKIP] Missing file for pair: {pid1}, {pid2}")
                continue

            try:
                with h5py.File(f1_path, 'r') as f1, h5py.File(f2_path, 'r') as f2:
                    sims = []

                    # 1. Vector features
                    for feat_name in ["hpcp", "chroma_cens", "mfcc_htk", "crema"]:
                        if feat_name in f1 and feat_name in f2:
                            vec1 = np.array(f1[feat_name])
                            vec2 = np.array(f2[feat_name])
                            mean1 = np.mean(vec1, axis=0)
                            mean2 = np.mean(vec2, axis=0)

                            sim = self.compute_similarity_features(mean1, mean2) if mean1.shape == mean2.shape else 0.0
                        else:
                            sim = 0.0
                        sims.append(float(sim))  # float

                    # 2. Key extractor
                    try:
                        ke1 = f1["key_extractor"].attrs
                        ke2 = f2["key_extractor"].attrs

                        key1 = ke1["key"]
                        key2 = ke2["key"]
                        scale1 = ke1["scale"]
                        scale2 = ke2["scale"]

                        key1 = key1.decode() if isinstance(key1, (bytes, np.bytes_)) else str(key1)
                        key2 = key2.decode() if isinstance(key2, (bytes, np.bytes_)) else str(key2)
                        scale1 = scale1.decode() if isinstance(scale1, (bytes, np.bytes_)) else str(scale1)
                        scale2 = scale2.decode() if isinstance(scale2, (bytes, np.bytes_)) else str(scale2)

                        strength1 = safe_scalar(ke1["strength"])
                        #print("str1",strength1)
                        strength2 = safe_scalar(ke2["strength"])
                        #print("str2",strength2)
                        key_sim = self.key_similarity(key1,scale1,key2,scale2)
                        #scale_sim = 1.0 if scale1 == scale2 else 0.0
                        strength_diff = abs(strength1 - strength2)
                    except Exception as e:
                        print(f"[WARN] key_extractor error in {pid1}, {pid2}: {e}")
                        key_sim, scale_sim, strength_diff = 0.0, 0.0, 1.0

                    sims.extend([float(key_sim), float(strength_diff)])
                    # #float(scale_sim)
                    # 3. Madmom tempos
                    try:
                        tempos1 = np.array(f1["madmom_features"]["tempos"])
                        tempos2 = np.array(f2["madmom_features"]["tempos"])
                        #print("tempos1",tempos1)
                        #print("tempos2",tempos2)
                        tempo_diff = abs(tempos1[0, 0] - tempos2[0, 0]) if len(tempos1) > 0 and len(tempos2) > 0 else 100.0
                        #print("tempod_diff",tempo_diff)
                    except Exception as e:
                        print(f"[WARN] madmom tempos error in {pid1}, {pid2}: {e}")
                        tempo_diff = 100.0

                    sims.append(float(tempo_diff))
                    sims = normalize_similarity_vector(sims)
                    # Final strict check
                    features.append(sims)
                    labels.append(label)
                    #if isinstance(sims, list) and len(sims) == 8 and all(isinstance(s, float) for s in sims):
                    #3    features.append(sims)
                    #    labels.append(label)
                    #else:
                     #   print(f"[SKIP] Bad sims vector for pair {pid1}, {pid2}: {sims} LABEL {label}")

            except Exception as e:
                print(f"[ERROR] Reading features for {pid1}, {pid2}: {e}")
                continue

        return np.array(features, dtype=np.float32), np.array(labels)



    def compute_tonal_similarity(self, hpcp_a, hpcp_b):
    
        mean_a = np.mean(hpcp_a, axis=0)
        mean_b = np.mean(hpcp_b, axis=0)

        # calculate cosine similarity
        similarity = np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b))
        return similarity

    def load_hpcp_feature(self,file_path):
        with h5py.File(file_path, 'r') as f:
            return np.array(f['hpcp'])

    def extract_similarity_features(self, pair_list, feature_base_path):
        """
        pair_list: List of tuples [(pid1, pid2, label), ...]
        feature_base_path: path to folder with W_*/P_*.h5
        """
        features, labels = [], []
        import time

        start = time.time()



        for pid1, pid2, label in pair_list:
            h1 = self.find_h5_file(pid1, feature_base_path)
            h2 = self.find_h5_file(pid2, feature_base_path)
            if not h1 or not h2:
                continue  # skip if file not found

            f1 = self.load_hpcp_feature(h1)
            f2 = self.load_hpcp_feature(h2)
            sim = self.compute_tonal_similarity(f1, f2)
            #print([sim])
            features.append([sim])  # must be 2D: [[0.87]]
            labels.append(label)
        end = time.time()
        print(end - start)

        return np.array(features), np.array(labels)

    def find_h5_file(self,pid, base_path):
        for root, _, files in os.walk(base_path):
            for f in files:
                if f.startswith(pid) and f.endswith(".h5"):
                    return os.path.join(root, f)
        return None

    def train_tonal_model(self, X_train, y_train,epoch_number=10,learning_rate=0.001, model_path="DaTonalCover.pth"):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn_model = self.nn_model.to(device)
        print("training started")

        optimizer = optim.Adam(self.nn_model.parameters(),lr=learning_rate)
        criterion = nn.BCELoss()

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(epoch_number):
            self.nn_model.train()
            running_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = self.nn_model(x_batch).view(-1)


                if not torch.all((preds >= 0) & (preds <= 1)):
                    print("Bad prediction range detected:")
                    print(f"Min: {preds.min().item()}, Max: {preds.max().item()}")
                    print("Sample preds:", preds[:10])
                    print("Sample labels:", y_batch[:10])
                    continue  # skip "ugly pairs"
                                # some pairs were giving errors (for example missing features) or wrong file structure
                                # i just skip them as a fast way to overcome this problem
                                # the amount of these "ugly pairs" is not big enough - its less than 10%
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"[{epoch+1}] Loss: {running_loss/len(loader):.4f}")

        torch.save(self.nn_model.state_dict(), model_path)
        return self.nn_model

    def generate_pairs_from_csv(self, csv_path, pairs_limit=1000, num_negative_pairs_per_clique=1):
        df = pd.read_csv(csv_path)
        grouped = df.groupby("clique")

        positive_pairs = []
        negative_pairs = []

        # Generate all positive pairs
        for clique_id, group in grouped:
            pids = group['id'].tolist()
            for pid1, pid2 in combinations(pids, 2):
                positive_pairs.append((pid1, pid2, 1))
        ## 
        # Generate all negative pairs
        all_cliques = list(grouped.groups.keys())
        for clique_id, group in grouped:
            this_pids = group['id'].tolist()
            other_cliques = [cid for cid in all_cliques if cid != clique_id]
            for pid in this_pids:
                for _ in range(num_negative_pairs_per_clique):
                    rand_clique = random.choice(other_cliques)
                    rand_pid = df[df['clique'] == rand_clique].sample(1)['id'].values[0]
                    negative_pairs.append((pid, rand_pid, 0))

        # Combine and shuffle 
        # shuffle is essential so its randomized
        # and not just all positive then all negative
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        if(pairs_limit==0):
            return all_pairs
        else:
            return all_pairs[:pairs_limit]

    