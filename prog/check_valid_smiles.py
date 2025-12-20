import pandas as pd
from rdkit import Chem

# Definizione dei dataset con percorsi e colonne SMILES
datasets = {
    'BACE': ('data/dataset_finetuning/bace/raw/bace.csv', 'mol'),
    'BBBP': ('data/dataset_finetuning/bbbp/raw/BBBP.csv', 'smiles'),
    'CEP': ('data/dataset_finetuning/cep/raw/cep.csv', 'smiles'),
    'HIV': ('data/dataset_finetuning/hiv/raw/HIV.csv', 'smiles'),
    'Malaria': ('data/dataset_finetuning/malaria/raw/malaria.csv', 'smiles'),
    'Lipophilicity': ('data/dataset_finetuning/lipophilicity/raw/Lipophilicity.csv', 'smiles')
}

print("Dataset Summary - Valid SMILES Rows")
print("=" * 40)

for name, (path, smiles_col) in datasets.items():
    df = pd.read_csv(path)
    valid_count = 0
    for smiles in df[smiles_col]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                Chem.SanitizeMol(mol)
                valid_count += 1
        except:
            pass
    print(f"{name} - {valid_count}")