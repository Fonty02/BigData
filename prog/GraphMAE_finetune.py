import sys
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from collections import defaultdict
from codecarbon import EmissionsTracker
from scipy.special import expit
import logging

# --- Configurazione Logging CodeCarbon ---
cc_level = os.getenv('CODECARBON_LOG_LEVEL', 'CRITICAL').upper()
try:
    logging.getLogger('codecarbon').setLevel(getattr(logging, cc_level, logging.CRITICAL))
except Exception:
    logging.getLogger('codecarbon').setLevel(logging.CRITICAL)

cc_logfile = os.getenv('CODECARBON_LOG_FILE', None)
if cc_logfile:
    try:
        fh = logging.FileHandler(cc_logfile)
        fh.setLevel(getattr(logging, cc_level, logging.CRITICAL))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger = logging.getLogger('codecarbon')
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.addHandler(fh)
    except Exception:
        pass
# -------------------------------------------------------------------------------

# Aggiunge GraphMAE al path
sys.path.append(os.path.join(os.path.dirname(__file__), "third_party", "GraphMAE"))
sys.path.append(os.path.join(os.path.dirname(__file__), "third_party", "GraphMAE", "chem"))

try:
    from third_party.GraphMAE.chem.model import GNN_graphpred
except ImportError:
    raise ImportError("Impossibile importare GNN_graphpred. Verifica che la cartella 'third_party/GraphMAE' esista.")

from load_dataset import load_dataset
from EarlyStopping import EmissionsEarlyStoppingCallback

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    raise ImportError("RDKit non è installato. Esegui 'pip install rdkit' per utilizzare lo Scaffold Splitting.")

# Dataset configurazioni
REGRESSION_DATASETS = {"cep", "malaria", "lipophilicity", "delaney"}
DATASETS_WITH_MISSING = {"muv", "toxcast", "tox21"}
DATASET_NUM_TASKS = {
    "tox21": 12, "toxcast": 617, "sider": 27, "muv": 17, "clintox": 2,
    "bace": 1, "bbbp": 1, "hiv": 1, "cep": 1, "malaria": 1, 
    "lipophilicity": 1, "delaney": 1
}

def scaffold_split_indices(df, smiles_col='smiles', label_col='label', sizes=(0.8, 0.1, 0.1), seed=42):
    """
    Ritorna GLI INDICI per train, val, test basandosi sugli scaffold.
    Assicura che val e test abbiano rappresentanza di classi.
    """
    rng = np.random.RandomState(seed)
    
    # 1. Raggruppa molecole per scaffold (uso itertuples per velocità)
    valid_indices = []
    scaffolds = defaultdict(list)
    
    # Ottimizzazione: pre-calcolo indici e smiles
    print("Generazione Scaffolds...")
    for row in df.itertuples():
        smiles = getattr(row, smiles_col)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds[scaffold].append(row.Index)
            valid_indices.append(row.Index)
            
    print(f"Molecole valide: {len(valid_indices)}/{len(df)}")

    # 2. Crea metadati
    scaffold_info = []
    all_valid_labels = df.loc[valid_indices, label_col].unique()
    
    for sc, idxs in scaffolds.items():
        labels = df.loc[idxs, label_col].values
        # Usiamo set per verifica rapida delle classi presenti nello scaffold
        unique_lbls = set(labels)
        
        scaffold_info.append({
            'scaffold': sc,
            'indices': idxs,
            'len': len(idxs),
            'unique_labels': unique_lbls # Salviamo il set delle label presenti
        })

    # 3. Ordina (Shuffle + Sort)
    rng.shuffle(scaffold_info)
    scaffold_info.sort(key=lambda x: x['len'], reverse=True)

    # 4. Assegnazione Iniziale
    train_cutoff = sizes[0] * len(valid_indices)
    val_cutoff = (sizes[0] + sizes[1]) * len(valid_indices)
    
    train_idxs = []
    val_idxs = []
    test_idxs = []
    
    # Teniamo traccia degli scaffold nel train per poterli spostare dopo
    train_scaffolds_list = [] 

    for info in scaffold_info:
        if len(train_idxs) + info['len'] <= train_cutoff:
            train_idxs.extend(info['indices'])
            train_scaffolds_list.append(info)
        elif len(train_idxs) + len(val_idxs) + info['len'] <= val_cutoff:
            val_idxs.extend(info['indices'])
        else:
            test_idxs.extend(info['indices'])

    # 5. Funzione di Bilanciamento (Applicabile sia a Val che a Test)
    def balance_dataset(target_idxs, dataset_name):
        # Convertiamo a set per velocità di calcolo
        current_labels = set(df.loc[target_idxs, label_col].unique())
        
        # Se abbiamo meno di 2 classi (e nel dataset originale ce ne sono almeno 2)
        if len(current_labels) < 2 and len(all_valid_labels) >= 2:
            # Trova quale label manca
            missing = list(set(all_valid_labels) - current_labels)
            target_label = missing[0] # Prendiamo la prima mancante
            
            # Cerca candidati nel TRAIN che hanno la label mancante
            # Ordiniamo per lunghezza crescente (smallest first) per minimizzare impatto
            candidates = [s for s in train_scaffolds_list if target_label in s['unique_labels']]
            candidates.sort(key=lambda x: x['len'])
            
            if candidates:
                swap_scaffold = candidates[0]
                
                # Rimuovi da Train (logica ottimizzata)
                # Usiamo set removal che è O(1) invece di list remove O(N)
                train_scaffolds_list.remove(swap_scaffold)
                
                # Per rimuovere gli indici velocemente, ricostruiamo la lista o usiamo set
                # Qui usiamo un trick veloce con i set
                idxs_to_remove = set(swap_scaffold['indices'])
                # Nota: modificare train_idxs qui richiede attenzione. 
                # Poiché train_idxs è una lista di interi, la ricostruzione è sicura:
                # Modifichiamo la lista train_idxs "in place" filtrandola
                train_idxs[:] = [i for i in train_idxs if i not in idxs_to_remove]
                
                # Aggiungi al target (Val o Test)
                target_idxs.extend(swap_scaffold['indices'])
                print(f"   -> Bilanciato {dataset_name}: aggiunto scaffold con label {target_label} ({swap_scaffold['len']} mol)")
            else:
                print(f"   Warning: Impossibile trovare scaffold nel train per bilanciare {dataset_name}")

    # Applica bilanciamento a entrambi
    balance_dataset(test_idxs, "TEST")
    balance_dataset(val_idxs, "VAL")

    return train_idxs, val_idxs, test_idxs



def compute_metrics(preds, labels, is_regression, train_labels_mean=None, has_missing=False):
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    
    if has_missing:
        valid_mask = ~(np.isnan(preds_flat) | np.isnan(labels_flat))
        preds_clean = preds_flat[valid_mask]
        labels_clean = labels_flat[valid_mask]
    else:
        preds_clean = preds_flat
        labels_clean = labels_flat
    
    if len(labels_clean) == 0:
        return {"rse": float('nan'), "roc_auc": 0.5, "accuracy": 0.0}
    
    if is_regression:
        numerator = np.sum((labels_clean - preds_clean) ** 2)
        if train_labels_mean is not None:
            denominator = np.sum((labels_clean - train_labels_mean) ** 2)
        else:
            denominator = np.sum((labels_clean - np.mean(labels_clean)) ** 2)
        
        rse = float(numerator / denominator) if denominator > 0 else 0.0
        mse = np.mean((preds_clean - labels_clean) ** 2)
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(preds_clean - labels_clean)))
        
        return {"rse": rse, "rmse": rmse, "mae": mae}
    else:
        unique_labels = np.unique(labels_clean)
        if len(unique_labels) < 2:
            raise ValueError("Impossibile calcolare ROC-AUC con una sola classe presente nelle etichette.")
        else:
            try:
                roc_auc = roc_auc_score(labels_clean, preds_clean)
            except ValueError:
                raise ValueError("Errore nel calcolo del ROC-AUC. Verifica le etichette e le predizioni.")
            
        preds_binary = (preds_clean > 0.5).astype(int)
        acc = accuracy_score(labels_clean, preds_binary)
        
        return {"roc_auc": roc_auc, "accuracy": acc}


def train(model, device, loader, optimizer, criterion, is_regression, has_missing, epoch_num):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for step, batch in enumerate(tqdm(loader, desc=f"Train E{epoch_num}", leave=False)):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        if not is_regression:
            mask_valid = ~torch.isnan(y)
            if mask_valid.any() and (y[mask_valid] < 0).any():
                y = (y + 1) / 2

        if has_missing:
            is_valid = ~torch.isnan(y)
            if is_valid.sum() == 0:
                continue
            loss = criterion(pred.double()[is_valid], y[is_valid])
        else:
            loss = criterion(pred.double(), y)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / num_batches if num_batches > 0 else 0.0


def eval(model, device, loader, is_regression, train_labels_mean=None, has_missing=False):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Eval", leave=False)):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y = batch.y.view(pred.shape)
        
        if not is_regression:
            mask_valid = ~torch.isnan(y)
            if mask_valid.any() and (y[mask_valid] < 0).any():
                y = (y + 1) / 2

        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    
    if not is_regression:
        y_scores = expit(y_scores)
    
    return compute_metrics(y_scores, y_true, is_regression, train_labels_mean, has_missing)


def main(dataset, epochs, batch_size, lr, model_path, use_early_stopping, alpha=0.9, warmup_epochs=10):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    num_tasks = DATASET_NUM_TASKS.get(dataset.lower(), 1)
    has_missing = dataset.lower() in DATASETS_WITH_MISSING
    is_regression = dataset.lower() in REGRESSION_DATASETS
    
    data_list = load_dataset("gnn", dataset)
    df = load_dataset("transformers", dataset)
    
    if len(data_list) != len(df):
        min_len = min(len(data_list), len(df))
        data_list = data_list[:min_len]
        df = df.iloc[:min_len].reset_index(drop=True)

    train_idxs, val_idxs, test_idxs = scaffold_split_indices(df, smiles_col="smiles")
    
    train_dataset = [data_list[i] for i in train_idxs]
    val_dataset = [data_list[i] for i in val_idxs]
    test_dataset = [data_list[i] for i in test_idxs]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    
    
    train_labels_mean = None
    if is_regression:
        train_labels = [data.y.item() for data in train_dataset if not torch.isnan(data.y).any()]
        if train_labels:
            train_labels_mean = np.mean(train_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GNN_graphpred(
        num_layer=5,
        emb_dim=300,
        num_tasks=num_tasks,
        JK="last",
        drop_ratio=0.5,
        graph_pooling="mean",
        gnn_type="gin"
    )
    
    if os.path.exists(model_path):
        try:
            model.from_pretrained(model_path)
        except Exception as e:
            pass
    else:
        pass
        
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    
    criterion = nn.MSELoss() if is_regression else nn.BCEWithLogitsLoss()
    
    project_name = f"graphmae_early_{warmup_epochs}" if use_early_stopping else f"graphmae_classic_{warmup_epochs}"
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=f"./emissions_{dataset}",
        save_to_file=True,
        log_level="critical"
    )
    tracker.start()
    
    early_stopping_callback = EmissionsEarlyStoppingCallback(
        tracker, alpha=alpha, beta=0.2,
        warmup_epochs=warmup_epochs, is_regression=is_regression, classic=not use_early_stopping
    )
    
    epochs_completed = epochs
    
    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, criterion, 
                          is_regression, has_missing, epoch+1)
        
        val_metrics = eval(model, device, val_loader, is_regression, 
                          train_labels_mean, has_missing)
        
        
        metric_name = "RSE" if is_regression else "ROC-AUC"
        val_perf = val_metrics["rse"] if is_regression else val_metrics["roc_auc"]
        current_lr = optimizer.param_groups[0]['lr']
        
        if early_stopping_callback:
            if early_stopping_callback.check_early_stopping(val_metrics, epoch):
                epochs_completed = epoch + 1
                break

    test_metrics = eval(model, device, test_loader, is_regression, 
                       train_labels_mean, has_missing)
    
    emissions = tracker.stop()

    if is_regression:
        auroc = float('nan')
        rse = test_metrics.get("rse", float('nan'))
        rmse = test_metrics.get("rmse", float('nan'))
    else:
        auroc = test_metrics.get("roc_auc", float('nan'))
        rse = float('nan')
        rmse = float('nan')
    
    results_df = pd.DataFrame([{
        "experiment": project_name,
        "emissions": emissions,
        "auroc": auroc,
        "rse": rse,
        "rmse": rmse,
        "Epoche fornite": epochs,
        "epoche usate": int(epochs_completed),
        "warmup": warmup_epochs
    }])
    
    csv_file = f"experiments_{dataset}.csv"
    results_df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning GraphMAE")
    parser.add_argument("--dataset", type=str, default="bace", 
                       help="Nome del dataset (es. bace, tox21)")
    parser.add_argument("--epochs", type=int, default=100, help="Max epoche")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_path", type=str, 
                       default="third_party/GraphMAE/chem/init_weights/pretrained.pth",
                       help="Path ai pesi pre-addestrati")
    parser.add_argument("--alpha", type=float, default=0.9, help="EMA alpha per early stopping")
    parser.add_argument("--warmup_epochs", type=int, default=30, help="Warmup epoche")
    
    args = parser.parse_args()
    
    for use_early in [False, True]:
        main(
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_path=args.model_path,
            use_early_stopping=use_early,
            alpha=args.alpha,
            warmup_epochs=args.warmup_epochs
        )