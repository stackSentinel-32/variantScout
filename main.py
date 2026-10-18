import pandas as pd, os

ref = pd.read_csv("data/raw/DMS_substitutions.csv")
print(ref.shape)
print(ref.columns.tolist())

paths = {
    "dms":        "data/raw/dms/DMS_ProteinGym_substitutions",
    "msa":        "data/raw/msa/DMS_msa_files",
    "weights":    "data/raw/weights/DMS_msa_weights",
    "structures": "data/raw/structures/ProteinGym_AF2_structures",
}
for name, p in paths.items():
    files = os.listdir(p)
    print(name, len(files), files[:3])
    