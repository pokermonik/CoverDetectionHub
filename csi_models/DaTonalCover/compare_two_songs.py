import torch
import numpy as np
import librosa
from model import DaTonalCover
# to run evaluate.py, the import needs to be ' from csi_models.DaTonalCover.model '
# but if you want to just check 2 songs directly from this file, the import needs to be ' from model '

def extract_all_features_from_mp3(mp3_path):
    y, sr = librosa.load(mp3_path, sr=44100)
    
    # HPCP 
    hpcp = librosa.feature.chroma_cqt(y=y, sr=sr)
    hpcp = hpcp / (np.max(hpcp, axis=0, keepdims=True) + 1e-6)

    # chroma_cens
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens = chroma_cens / (np.max(chroma_cens, axis=0, keepdims=True) + 1e-6)

    # mfcc_htk
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = mfcc / (np.max(mfcc, axis=0, keepdims=True) + 1e-6)
    
    # crema 
    crema = librosa.feature.spectral_contrast(y=y, sr=sr)
    crema = crema / (np.max(crema) + 1e-6)


    # madmom tempos
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    #print("tempo",tempo)
   
    tempos = tempo 
    #print("tmps",tempos)

    # key
    key = estimate_key_librosa(y, sr) 

    return {
        "hpcp": hpcp.T,
        "chroma_cens": chroma_cens.T,
        "mfcc_htk": mfcc.T,
        "crema": crema.T,
        "key_extractor": key,
        "madmom_features": {"tempos": tempos},

    }

def estimate_key_librosa(y, sr):
    # You can approximate key detection by pitch class profile and major/minor detection here
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    # finding peak pitch klass to identify as key root
    key_idx = np.argmax(chroma_mean)
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = keys[key_idx]
    scale = 'major'  # Simplify: assume major
    strength = float(np.max(chroma_mean) / np.sum(chroma_mean))  # proxy strength
    return {"key": key, "scale": scale, "strength": strength}

import librosa



def compute_similarity_features(vec1, vec2):
    # cosine sim as similarity between 2 vec
    


    D, wp = librosa.sequence.dtw(vec1.T, vec2.T, metric='euclidean')
    return np.exp(-D[-1, -1] / 100)

def tempo_similarity(t1, t2):
    diff = abs(t1 - t2)
    return np.exp(-diff**2 / (2 * 20**2))

def normalize_sims(sims):
    sims = [float(s[0]) if isinstance(s, np.ndarray) and s.shape == (1,) else float(s) for s in sims]
    sims = np.array(sims, dtype=np.float32)
    # Strength difference → similarity
    sims[6] = 1.0 - np.clip(sims[6], 0, 1)

    # Tempo difference → similarity
    #sims[7] = 1.0 - np.clip(sims[7] / 100.0, 0, 1)

    return sims.reshape(1, -1)

def build_feature_vector(f1, f2, model):
    sims = []
    for feat_name in ["hpcp", "chroma_cens", "mfcc_htk", "crema"]:
        sim = 0.0
        if feat_name in f1 and feat_name in f2:
            sim = compute_similarity_features(f1[feat_name], f2[feat_name])
        sims.append(sim)

    # key extractor similarity
    #print(f"type(f1): {type(f1)}")
    #print(f"f1: {f1}")
    #print(f"type(f2): {type(f2)}")
    #print(f"f1: {f2}")
    model=DaTonalCover()
    key1=f1["key_extractor"]["key"]
    key2=f2["key_extractor"]["key"]
    scale1=f1["key_extractor"]["scale"]
    scale2=f2["key_extractor"]["scale"]
    key_sim=model.key_similarity(key1,scale1,key2,scale2)
    
    #key_sim = 1.0 if f1["key_extractor"]["key"] == f2["key_extractor"]["key"] else 0.0
    #scale_sim = 1.0 if f1["key_extractor"]["scale"] == f2["key_extractor"]["scale"] else 0.0
    strength_diff = abs(f1["key_extractor"]["strength"] - f2["key_extractor"]["strength"])
    sims.extend([key_sim, strength_diff])

    # madmom tempos diff
    tempos1 = f1["madmom_features"]["tempos"]
    tempos2 = f2["madmom_features"]["tempos"]
    tempo_diff=0
    if tempos1.size > 0 and tempos2.size > 0:
        tempo_diff=tempo_similarity(tempos1,tempos2)
        
    else:
        tempo_diff = -1
        
    sims.append(tempo_diff)
    print("sims",sims)
    sims=normalize_sims(sims)
    print("sims2",sims)
    
    return np.array(sims, dtype=np.float32).reshape(1, -1)

def main(song1_path, song2_path, model_path="csi_models/checkpoints/DaTonal/DaTonalCover.pth"):
    model = DaTonalCover()
    model.nn_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"),weights_only=True))
    model.nn_model.eval()

    f1 = extract_all_features_from_mp3(song1_path)
    f2 = extract_all_features_from_mp3(song2_path)
    print(f"type(f1): {type(f1)}")
    print(f"f1: {f1}")
    print(f"type(f2): {type(f2)}")
    print(f"f1: {f2}")
    feature_vector = build_feature_vector(f1, f2, model)
    #print(f"Feature vector: {feature_vector}")

    with torch.no_grad():
        pred = model.nn_model(torch.tensor(feature_vector, dtype=torch.float32)).item()

    print(f"Cover prediction score: {pred:.4f}")
    if pred >= 0.75:
        print("Likely a cover!")
    else:
        print("Probably not a cover.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("song1", help="Path to first song (.mp3)")
    parser.add_argument("song2", help="Path to second song (.mp3)")
    parser.add_argument("--model", default="csi_models/checkpoints/DaTonal/DaTonalCover.pth", help="Path to trained model file")
    args = parser.parse_args()

    main(args.song1, args.song2, args.model)
