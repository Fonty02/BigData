import os
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset

# Definiamo una classe wrapper per il dataset GNN
# Questo permette a PyTorch Geometric di gestire il file .pt correttamente
class BaceGNNDataset(InMemoryDataset):
    def __init__(self, root, file_name, transform=None, pre_transform=None):
        self.file_name = file_name
        super().__init__(root, transform, pre_transform)
        # Carica il file .pt specifico (data.pt o geometric...)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Non ci servono i file raw perché abbiamo già quelli processati
        return []

    @property
    def processed_file_names(self):
        return [self.file_name]

    def download(self):
        # Nessun download necessario
        pass

    def process(self):
        # Nessun processing necessario, i file esistono già
        pass

def get_bace_dataset(root_dir='./bace', data_type='graph'):
    """
    Carica il dataset BACE in base al tipo di modello che devi usare.
    
    Args:
        root_dir (str): Il percorso alla cartella 'bace' (che deve contenere 'processed').
        data_type (str): Il tipo di dati richiesto. 
                         Opzioni: 'graph', 'text', 'geometric'.
    
    Returns:
        Un oggetto Dataset di PyG (per graph/geometric) o un DataFrame pandas (per text).
    """
    
    processed_dir = os.path.join(root_dir, 'processed')
    
    # 1. MODALITÀ GNN (Carica data.pt)
    if data_type == 'graph':
        print(f"Caricamento dati GNN da: {processed_dir}/data.pt")
        if not os.path.exists(os.path.join(processed_dir, 'data.pt')):
            raise FileNotFoundError("Il file data.pt non esiste. Controlla il percorso.")
            
        return BaceGNNDataset(root=root_dir, file_name='data.pt')

    # 2. MODALITÀ BERT (Carica smiles.csv)
    elif data_type == 'text':
        csv_path = os.path.join(processed_dir, 'smiles.csv')
        print(f"Caricamento dati Testuali da: {csv_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError("Il file smiles.csv non esiste. Controlla il percorso.")
            
        # Restituisce un DataFrame Pandas (pronto per HuggingFace Dataset)
        df = pd.read_csv(csv_path)
        return df

    # 3. MODALITÀ GEOMETRICA (Carica geometric_data_processed.pt)
    elif data_type == 'geometric':
        print(f"Caricamento dati 3D da: {processed_dir}/geometric_data_processed.pt")
        # Nota: PyG cerca i file processati automaticamente, ma specifichiamo il nome file custom
        return BaceGNNDataset(root=root_dir, file_name='geometric_data_processed.pt')

    else:
        raise ValueError("data_type non valido. Usa 'graph', 'text' o 'geometric'.")

# --- ESEMPIO DI UTILIZZO (Test) ---
if __name__ == "__main__":
    # Esempio per GNN
    try:
        dataset_gnn = get_bace_dataset(data_type='graph')
        print(f"Dataset GNN caricato! Numero molecole: {len(dataset_gnn)}")
        print(f"Esempio prima molecola: {dataset_gnn[0]}")
    except Exception as e:
        print(f"Errore caricamento GNN: {e}")

    print("-" * 20)

    # Esempio per BERT
    try:
        dataset_text = get_bace_dataset(data_type='text')
        print(f"Dataset Testo caricato! Righe: {len(dataset_text)}")
        print(f"Prime 2 righe:\n{dataset_text.head(2)}")
    except Exception as e:
        print(f"Errore caricamento Testo: {e}")