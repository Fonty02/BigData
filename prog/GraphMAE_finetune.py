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

# Import personalizzati
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


def scaffold_split_indices(df, smiles_col='smiles', sizes=(0.8, 0.1, 0.1), seed=42):
    """
    Divide il dataset in train/val/test basandosi sugli scaffold molecolari.
    FIX: Include il meccanismo di ribilanciamento (preso dalla versione Transformers)
    per garantire che il Test Set abbia entrambe le classi.
    """
    import numpy as np
    from collections import defaultdict
    
    # 1. Raggruppa molecole per scaffold
    valid_indices = []
    scaffolds = defaultdict(list)
    
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds[scaffold].append(idx)
            valid_indices.append(idx)
    
    print(f"Molecole valide per split: {len(valid_indices)}/{len(df)}")

    # 2. Crea metadati per ogni scaffold (indici e label presenti)
    scaffold_info = []
    for sc, idxs in scaffolds.items():
        # Ottieni le label per questo scaffold usando il dataframe
        labels = df.loc[idxs, 'label'].values
        unique_labels = np.unique(labels)
        scaffold_info.append({
            'scaffold': sc,
            'indices': idxs,
            'len': len(idxs),
            'has_class_0': 0 in unique_labels or 0.0 in unique_labels,
            'has_class_1': 1 in unique_labels or 1.0 in unique_labels
        })

    # 3. Ordina scaffold per dimensione (dal più grande al più piccolo)
    # Mescoliamo prima per rompere l'ordine deterministico degli scaffold di pari dimensione
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_info)
    scaffold_info.sort(key=lambda x: x['len'], reverse=True)

    # 4. Assegnazione Iniziale
    train_idxs, val_idxs, test_idxs = [], [], []
    train_scaffolds_list = [] # Teniamo traccia degli oggetti scaffold nel train per il fix
    
    train_cutoff = sizes[0] * len(valid_indices)
    val_cutoff = (sizes[0] + sizes[1]) * len(valid_indices)

    for info in scaffold_info:
        if len(train_idxs) + info['len'] <= train_cutoff:
            train_idxs.extend(info['indices'])
            train_scaffolds_list.append(info)
        elif len(train_idxs) + len(val_idxs) + info['len'] <= val_cutoff:
            val_idxs.extend(info['indices'])
        else:
            test_idxs.extend(info['indices'])
            
    # 5. FIX: Logica di ribilanciamento (Adattata dalla versione Transformers)
    # Controlliamo se nel test mancano classi
    if not test_idxs:
         return train_idxs, val_idxs, test_idxs
         
    test_labels = df.loc[test_idxs, 'label'].unique()
    
    # Se manca la classe 1 o la classe 0
    if len(test_labels) < 2:
        # Identifica quale label manca
        missing_label = 1 if (0 in test_labels or 0.0 in test_labels) else 0
        print(f"⚠️ Warning: Test set sbilanciato (Manca classe {missing_label}). Applicazione FIX...")
        
        # Cerca uno scaffold nel TRAIN che contenga la classe mancante
        # Ordiniamo i candidati dal più piccolo al più grande per minimizzare l'impatto sulle dimensioni
        candidates = [s for s in train_scaffolds_list if (s['has_class_1'] if missing_label == 1 else s['has_class_0'])]
        candidates.sort(key=lambda x: x['len'])
        
        if candidates:
            swap_scaffold = candidates[0] # Prendi il più piccolo utile
            
            # Rimuovi dal train
            for idx in swap_scaffold['indices']:
                if idx in train_idxs: train_idxs.remove(idx)
            
            # Aggiungi al test
            test_idxs.extend(swap_scaffold['indices'])
            
            print(f"   🔄 Spostato scaffold (size={swap_scaffold['len']}) dal Train al Test per aggiungere classe {missing_label}")
            
            # Verifica finale
            final_test_labels = df.loc[test_idxs, 'label'].unique()
            print(f"   Classi nel Test Set dopo il fix: {final_test_labels}")
        else:
            print("   ❌ Impossibile bilanciare: nessun scaffold adatto trovato nel Train.")

    print(f"Scaffold Split Result: Train={len(train_idxs)}, Val={len(val_idxs)}, Test={len(test_idxs)}")
    
    return train_idxs, val_idxs, test_idxs



def compute_metrics(preds, labels, is_regression, train_labels_mean=None, has_missing=False):
    """
    Calcola le metriche di valutazione.
    Per classificazione: gestisce il caso di classe singola restituendo 0.5 e loggando un warning.
    """
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    
    # Rimuovi NaN (per dataset multitask o con missing values)
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
        # Calcolo RSE (Relative Squared Error)
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
        # Calcolo ROC-AUC per classificazione
        # Controlliamo esplicitamente se c'è una sola classe per evitare crash o risultati silenti
        unique_labels = np.unique(labels_clean)
        if len(unique_labels) < 2:
            # Fallback standard per AUC non definita
            roc_auc = 0.5 
        else:
            # Calcolo normale
            try:
                roc_auc = roc_auc_score(labels_clean, preds_clean)
            except ValueError:
                roc_auc = 0.5
            
        preds_binary = (preds_clean > 0.5).astype(int)
        acc = accuracy_score(labels_clean, preds_binary)
        
        return {"roc_auc": roc_auc, "accuracy": acc}


def train(model, device, loader, optimizer, criterion, is_regression, has_missing, epoch_num):
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Usa tqdm per la barra di progresso
    for step, batch in enumerate(tqdm(loader, desc=f"Train E{epoch_num}", leave=False)):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        if not is_regression:
            # Conversione di sicurezza: se troviamo -1, portiamo a 0
            # Questo serve se il fix in load_dataset non fosse attivo
            mask_valid = ~torch.isnan(y)
            if mask_valid.any() and (y[mask_valid] < 0).any():
                y = (y + 1) / 2

        # Calcolo Loss
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
        y_scores = expit(y_scores)  # Sigmoide per ottenere probabilità
    
    # Dentro eval(), prima di chiamare compute_metrics:
    unique_test_labels = np.unique(y_true)
    print(f"DEBUG LABEL TEST: {unique_test_labels}")
    print(f"DEBUG PREDIZIONI (primi 10): {y_scores[:10].flatten()}")

    return compute_metrics(y_scores, y_true, is_regression, train_labels_mean, has_missing)


def main(dataset, epochs, batch_size, lr, model_path, use_early_stopping, alpha=0.9, warmup_epochs=3):
    print(f"\n{'='*80}")
    print(f"Fine-tuning GraphMAE sul dataset {dataset.upper()}")
    print(f"Configurazione: Early Stopping={use_early_stopping}, Epochs={epochs}")
    print(f"{'='*80}\n")
    
    # Imposta seed per riproducibilità
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Parametri dataset
    num_tasks = DATASET_NUM_TASKS.get(dataset.lower(), 1)
    has_missing = dataset.lower() in DATASETS_WITH_MISSING
    is_regression = dataset.lower() in REGRESSION_DATASETS
    
    # Caricamento Dati
    print("Caricamento dataset...")
    data_list = load_dataset("gnn", dataset)  # Qui entra in gioco il fix di load_dataset.py
    df = load_dataset("transformers", dataset)
    
    # Allineamento lunghezze
    if len(data_list) != len(df):
        print(f"⚠️ Warning: Mismatch dataset lengths (GNN={len(data_list)}, CSV={len(df)}). Truncating.")
        min_len = min(len(data_list), len(df))
        data_list = data_list[:min_len]
        df = df.iloc[:min_len].reset_index(drop=True)

    # Split
    train_idxs, val_idxs, test_idxs = scaffold_split_indices(df, smiles_col="smiles")
    
    train_dataset = [data_list[i] for i in train_idxs]
    val_dataset = [data_list[i] for i in val_idxs]
    test_dataset = [data_list[i] for i in test_idxs]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    
    
    # Calcolo media label per RSE (solo regressione)
    train_labels_mean = None
    if is_regression:
        train_labels = [data.y.item() for data in train_dataset if not torch.isnan(data.y).any()]
        if train_labels:
            train_labels_mean = np.mean(train_labels)
            print(f"   Media label training (per RSE): {train_labels_mean:.4f}")

    # Setup Modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in uso: {device}")
    
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
        print(f"   ✅ Pesi pre-addestrati caricati da: {model_path}")
        try:
            model.from_pretrained(model_path)
        except Exception as e:
            print(f"   ⚠️  Attenzione: Errore nel caricamento pesi ({e}). Si procede con init random.")
    else:
        print(f"   ⚠️  File pesi non trovato ({model_path}). Si procede con init random.")
        
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    
    criterion = nn.MSELoss() if is_regression else nn.BCEWithLogitsLoss()
    
    # Tracking Emissioni
    project_name = f"graphmae_early" if use_early_stopping else f"graphmae_classic"
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=f"./emissions_{dataset}",
        save_to_file=True,
        log_level="critical"
    )
    tracker.start()
    
    # Callback Early Stopping
    early_stopping_callback = None
    if use_early_stopping:
        early_stopping_callback = EmissionsEarlyStoppingCallback(
            tracker, alpha=alpha, beta=0.2,
            warmup_epochs=warmup_epochs, is_regression=is_regression
        )
        print(f"Early Stopping: ATTIVO (alpha={alpha}, warmup={warmup_epochs})")
    
    # Loop di Training
    print(f"\nAvvio training ({epochs} epoche)...")
    epochs_completed = epochs
    
    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, criterion, 
                          is_regression, has_missing, epoch+1)
        
        val_metrics = eval(model, device, val_loader, is_regression, 
                          train_labels_mean, has_missing)
        
        
        metric_name = "RSE" if is_regression else "ROC-AUC"
        val_perf = val_metrics["rse"] if is_regression else val_metrics["roc_auc"]
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.4f} | "
              f"Val {metric_name}: {val_perf:.4f} | LR: {current_lr:.6f}")
        
        if early_stopping_callback:
            if early_stopping_callback.check_early_stopping(val_metrics, epoch):
                print(f"🛑 Early stopping triggerato all'epoca {epoch+1}")
                epochs_completed = epoch + 1
                break

    # Test Finale
    print(f"\nValutazione finale su Test Set...")
    test_metrics = eval(model, device, test_loader, is_regression, 
                       train_labels_mean, has_missing)
    print(f"   Test Metrics: {test_metrics}")
    
    emissions = tracker.stop()
    print(f"Emissioni totali: {emissions:.6f} kg CO₂eq")

    # Salvataggio CSV
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
        "epochs_completed": int(epochs_completed)
    }])
    
    csv_file = f"experiments_{dataset}.csv"
    results_df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
    print(f"Risultati salvati in {csv_file}")
    print(f"{'='*80}\n")


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
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epoche")
    
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