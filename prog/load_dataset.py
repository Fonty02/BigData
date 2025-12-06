import os
import pandas as pd

# List dei dataset trattati come regressione
REGRESSION_DATASETS = {"cep", "malaria", "lipophilicity"}


def _find_label_column(raw_df: pd.DataFrame) -> str:
    """Prova a determinare la colonna target nel raw dataframe.
    Strategie:
    - se esiste 'Class' o 'label' o 'activity' o 'PCE' o 'exp' scegli quella
    - altrimenti scegli la prima colonna numerica diversa da 'smiles' o identificatori
    """
    candidates = [c for c in ["Class", "class", "label", "activity", "PCE", "exp", "value"] if c in raw_df.columns]
    if candidates:
        return candidates[0]

    # preferisci colonne numeriche non 'smiles'
    numeric_cols = [c for c in raw_df.columns if pd.api.types.is_numeric_dtype(raw_df[c]) and c.lower() not in ("smiles", "mol", "cmpd_chemblid")]
    if numeric_cols:
        return numeric_cols[0]

    # fallback: se il file ha una sola colonna numerica oltre allo smiles
    possible = [c for c in raw_df.columns if c.lower() not in ("smiles", "mol", "cmpd_chemblid")]
    if possible:
        return possible[0]

    raise ValueError("Non sono riuscito a determinare la colonna target nel raw CSV.")


def load_for_transformers(name: str):
    """Unisce gli SMILES processati con le etichette (classificazione o regressione).
    - Cerca label nel file processed (se presente come secondo campo)
    - Altrimenti la ricava dal raw CSV (cerca colonna numerica o 'Class')
    """
    base_dir = os.path.join(os.path.dirname(__file__), "data", "dataset_finetuning", name)
    processed_csv = os.path.join(base_dir, "processed", "smiles.csv")
    raw_csv = os.path.join(base_dir, "raw", f"{name}.csv")

    # Leggi processed; alcune volte ha solo SMILES, altre volte ha label già presenti
    proc_df = pd.read_csv(processed_csv, header=None)
    if proc_df.shape[1] == 1:
        proc_df.columns = ["smiles"]
    else:
        # Se processed ha più colonne, assumiamo che la prima sia smiles e la seconda label
        cols = [f"col{i}" for i in range(proc_df.shape[1])]
        proc_df.columns = cols
        # rinominiamo le prime due colonne utili in smiles e label
        proc_df = proc_df.rename(columns={cols[0]: "smiles", cols[1]: "label"})

    raw_df = pd.read_csv(raw_csv)
    # normalizza il nome della colonna smiles
    if "mol" in raw_df.columns and "smiles" not in raw_df.columns:
        raw_df = raw_df.rename(columns={"mol": "smiles"})

    # se processed ha già la label, usala
    if "label" in proc_df.columns:
        merged = proc_df
    else:
        # individua la colonna target nel raw e la usiamo
        label_col = _find_label_column(raw_df)
        raw_target_df = raw_df[[c for c in ("smiles", label_col) if c in raw_df.columns]]
        raw_target_df = raw_target_df.rename(columns={label_col: "label"})
        merged = proc_df.merge(raw_target_df, on="smiles", how="left")

    missing = merged["label"].isna().sum()
    if missing:
        raise ValueError(f"Etichette mancanti per {missing} molecole. Controlla i file raw/processed per '{name}'.")

    # coercizione tipo in base al dataset: regressione -> float, classificazione -> int
    if name in REGRESSION_DATASETS:
        merged["label"] = merged["label"].astype(float)
    else:
        merged["label"] = merged["label"].astype(int)
    return merged


def load_for_GNN():
    pass  # Implementazione specifica per il caricamento dei dati per GNN

def load_dataset(type:str,name:str):
    if type == "transformers":
        return load_for_transformers(name)
    elif type == "gnn":
        return load_for_GNN(name)
    else:
        raise ValueError(f"Tipo di dataset sconosciuto: {type}")
    
if __name__ == "__main__":
    exit(0)