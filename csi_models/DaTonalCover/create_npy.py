import os
import h5py
import numpy as np

def find_all_h5_files(root):
    h5_files = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".h5"):
                h5_files.append(os.path.join(dirpath, filename))
    return h5_files

def extract_features(h5_file):
    feature_dict = {}
    try:
        with h5py.File(h5_file, 'r') as f:
            # vector features
            for name in ["hpcp", "chroma_cens", "mfcc_htk", "crema"]:
                if name in f:
                    feature_dict[name] = np.array(f[name])
                else:
                    print(f"[SKIP] {name} not found in {h5_file}")

            # key extractor
            if "key_extractor" in f:
                attrs = f["key_extractor"].attrs
                try:
                    feature_dict["key_extractor"] = {
                        "key": attrs.get("key", "").decode() if isinstance(attrs.get("key", ""), bytes) else attrs.get("key", ""),
                        "scale": attrs.get("scale", "").decode() if isinstance(attrs.get("scale", ""), bytes) else attrs.get("scale", ""),
                        "strength": float(attrs.get("strength", 0.0))
                    }
                except Exception as e:
                    print(f"[WARN] Could not read key_extractor in {h5_file}: {e}")
            else:
                print(f"[SKIP] key_extractor not found in {h5_file}")

            # Madmom tempos
            if "madmom_features" in f and "tempos" in f["madmom_features"]:
                feature_dict["madmom_features"] = {
                    "tempos": np.array(f["madmom_features"]["tempos"])
                }
            else:
                print(f"[SKIP] madmom_features/tempos not found in {h5_file}")

        return feature_dict if feature_dict else None
    except Exception as e:
        print(f"[ERROR] Could not process {h5_file}: {e}")
        return None

def main():
    source_dir = r"D:\WimuProj\da-tacos_benchmark_subset_single_files"
    # source dir is where you have your da tacos files stored
    # i have them outside the main project folder
    # just copy the absolute path of that folder
    output_dir = "datasets/datacos/npyfolder"
    os.makedirs(output_dir, exist_ok=True)

    h5_files = find_all_h5_files(source_dir)
    print(f"Found {len(h5_files)} .h5 files")

    for h5_path in h5_files:
        pid = os.path.splitext(os.path.basename(h5_path))[0]
        features = extract_features(h5_path)
        if features is not None:
            out_path = os.path.join(output_dir, f"{pid}.npy")
            np.save(out_path, features, allow_pickle=True)
            print(f"[OK] Saved {out_path}")

if __name__ == "__main__":
    main()
