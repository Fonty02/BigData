import os
import pandas as pd

# List dei dataset trattati come regressione
REGRESSION_DATASETS = {"cep", "malaria", "lipophilicity", "delaney"}


def _find_label_column(raw_df: pd.DataFrame, dataset_name: str) -> str:
    candidates = [c for c in ["Class", "class", "label", "activity", "PCE", "exp", "value", "p_np", "HIV_active"] 
                  if c in raw_df.columns]
    if candidates:
        return candidates[0]

    numeric_cols = [c for c in raw_df.columns 
                   if pd.api.types.is_numeric_dtype(raw_df[c]) 
                   and c.lower() not in ("smiles", "mol", "cmpd_chemblid", "mol_id")]
    if numeric_cols:
        return numeric_cols[0]

    possible = [c for c in raw_df.columns 
               if c.lower() not in ("smiles", "mol", "cmpd_chemblid", "mol_id")]
    if possible:
        return possible[0]

    raise ValueError(f"Non sono riuscito a determinare la colonna target per '{dataset_name}'.")


def load_for_transformers(name: str):
    """
    Unisce gli SMILES processati con le etichette (classificazione o regressione).
    FIX: Migliore gestione dei nomi delle colonne e conversione label.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "data", "dataset_finetuning", name)
    processed_csv = os.path.join(base_dir, "processed", "smiles.csv")
    raw_csv = os.path.join(base_dir, "raw", f"{name}.csv")

    # Leggi processed
    proc_df = pd.read_csv(processed_csv, header=None)
    if proc_df.shape[1] == 1:
        proc_df.columns = ["smiles"]
    else:
        cols = [f"col{i}" for i in range(proc_df.shape[1])]
        proc_df.columns = cols
        proc_df = proc_df.rename(columns={cols[0]: "smiles", cols[1]: "label"})

    raw_df = pd.read_csv(raw_csv)
    
    # Normalizza il nome della colonna smiles
    if "mol" in raw_df.columns and "smiles" not in raw_df.columns:
        raw_df = raw_df.rename(columns={"mol": "smiles"})

    # Se processed ha già la label, usala
    if "label" in proc_df.columns:
        merged = proc_df
    else:
        # Individua la colonna target nel raw
        label_col = _find_label_column(raw_df, name)
        
        # FIX: Gestisci valori non numerici (es. 'CI' in HIV)
        raw_target_df = raw_df[["smiles", label_col]].copy()
        
        # Converti a numerico, forzando errori a NaN
        raw_target_df[label_col] = pd.to_numeric(raw_target_df[label_col], errors='coerce')
        
        # Rimuovi righe con label NaN
        raw_target_df = raw_target_df.dropna(subset=[label_col])
        
        raw_target_df = raw_target_df.rename(columns={label_col: "label"})
        merged = proc_df.merge(raw_target_df, on="smiles", how="inner")

    missing = merged["label"].isna().sum()
    if missing:
        raise ValueError(f"Etichette mancanti per {missing} molecole nel dataset '{name}'. "
                        f"Verifica che i file raw/processed siano allineati.")

    # FIX CRITICO: Converti label da {-1, 1} a {0, 1} per classificazione
    if name.lower() not in REGRESSION_DATASETS:
        unique_labels = merged["label"].unique()
        
        # Se le label sono {-1, 1}, convertile a {0, 1}
        if set(unique_labels) == {-1, 1} or set(unique_labels) == {-1.0, 1.0}:
            merged["label"] = (merged["label"] + 1) / 2
            merged["label"] = merged["label"].astype(int)
        else:
            # Assicurati che siano int
            merged["label"] = merged["label"].astype(int)
    else:
        # Regressione: converti a float
        merged["label"] = merged["label"].astype(float)
    
    return merged


def load_for_GNN(name: str):
    """
    Carica i dati geometrici processati per GNN.
    FIX: Patch in-memory per correggere le label da {-1, 1} a {0, 1} senza rigenerare il file.
    """
    import torch
    from torch_geometric.data import InMemoryDataset
    
    base_dir = os.path.join(os.path.dirname(__file__), "data", "dataset_finetuning", name)
    processed_path = os.path.join(base_dir, "processed", "geometric_data_processed.pt")
    
    # Controllo esistenza file (non possiamo rigenerarlo, quindi deve esistere)
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"File non trovato: {processed_path}\n"
            f"ERRORE: Non cancellare questo file perché manca lo script per rigenerarlo dai RAW.\n"
            f"Se il file è corrotto o assente, devi recuperare 'geometric_data_processed.pt' originale."
        )
        
    class TempDataset(InMemoryDataset):
        def __init__(self, path):
            super().__init__(root=None)
            # Carica i dati dal disco (anche se hanno label sbagliate)
            self.data, self.slices = torch.load(path, weights_only=False)
            
            # --- FIX CRITICO IN MEMORIA ---
            if hasattr(self.data, 'y') and self.data.y is not None:
                # Se il dataset non è regressione, controlliamo le label
                if name.lower() not in REGRESSION_DATASETS:
                    # Controlla se ci sono valori uguali a -1
                    mask_neg = (self.data.y == -1)
                    
                    if mask_neg.any():
                        self.data.y[mask_neg] = 0

    # Istanzia il dataset (applica il fix nel costruttore)
    dataset = TempDataset(processed_path)
    return dataset


def load_dataset(type: str, name: str):
    """
    Carica dataset per Transformers o GNN.
    
    Args:
        type: "transformers" o "gnn"
        name: nome del dataset (es. "bace", "bbbp")
    """
    if type == "transformers":
        return load_for_transformers(name)
    elif type == "gnn":
        return load_for_GNN(name)
    else:
        raise ValueError(f"Tipo di dataset sconosciuto: {type}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        
        try:
            df = load_for_transformers(dataset_name)
        except Exception as e:
            print(f"❌ Errore Transformers: {e}")
        
        try:
            data_list = load_for_GNN(dataset_name)
        except Exception as e:
            print(f"❌ Errore GNN: {e}")
    else:
        print("Usage: python load_dataset.py <dataset_name>")
        print("Example: python load_dataset.py bace")