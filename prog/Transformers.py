import argparse
import os
import inspect
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, accuracy_score
import random
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
from codecarbon import EmissionsTracker
from collections import defaultdict
import logging
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    raise ImportError("RDKit non è installato. Esegui 'pip install rdkit' per utilizzare lo Scaffold Splitting.")

# --- CodeCarbon logging control -------------------------------------------------
cc_level = os.getenv('CODECARBON_LOG_LEVEL', 'CRITICAL').upper()
try:
    logging.getLogger('codecarbon').setLevel(getattr(logging, cc_level, logging.WARNING))
except Exception:
    logging.getLogger('codecarbon').setLevel(logging.WARNING)

cc_logfile = os.getenv('CODECARBON_LOG_FILE', None)
if cc_logfile:
    try:
        fh = logging.FileHandler(cc_logfile)
        fh.setLevel(getattr(logging, cc_level, logging.WARNING))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger = logging.getLogger('codecarbon')
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.addHandler(fh)
    except Exception:
        pass
# -------------------------------------------------------------------------------

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, *args, **kwargs):
        return x

try:
    from prog.load_dataset import load_dataset
    from prog.EarlyStopping import EmissionsEarlyStoppingCallback
except Exception:
    from load_dataset import load_dataset
    from EarlyStopping import EmissionsEarlyStoppingCallback


MODEL_PATHS={
    "chemberta": "seyonec/ChemBERTa-zinc-base-v1",
    "chemberta2":"DeepChem/ChemBERTa-77M-MLM",
    "selformer":"HUBioDataLab/SELFormer",
    "smilesbert":"JuIm/SMILES_BERT"
}

REGRESSION_DATASETS = {"cep", "malaria", "lipophilicity"}

# Parametri dal PDF (Pagina 8)
HYPERPARAMS = {
    "lr": 1e-4,
    "batch_size": 32,
    "epochs": 30
}


def scaffold_split(df, smiles_col='smiles', sizes=(0.8, 0.1, 0.1), seed=42):
    # Setup random seeds
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    
    valid_indices = []
    scaffolds = defaultdict(list)
    
    # 1. Generazione Scaffold (Ottimizzato con itertuples)
    # Rileviamo automaticamente se usare 'label' o 'labels' (il tuo codice usa 'label' prima della rinomina)
    label_col = 'label' if 'label' in df.columns else 'labels'
    
    for row in df.itertuples():
        # getattr è sicuro e veloce
        smiles = getattr(row, smiles_col)
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds[scaffold].append(row.Index)
            valid_indices.append(row.Index)
        else:
            # Opzionale: print(f"Warning: Eliminata molecola indice {row.Index}")
            pass

    # 2. Creazione Metadati
    scaffold_metadata = []
    # Pre-calcoliamo tutte le label valide per sapere quali classi esistono globalmente
    all_valid_labels = set(df.loc[valid_indices, label_col].unique())
    
    for scaffold, indices in scaffolds.items():
        labels = df.loc[indices, label_col].values
        unique_labels = set(labels)
        
        scaffold_metadata.append({
            'scaffold': scaffold,
            'indices': indices,
            'len': len(indices),
            'unique_labels': unique_labels # Salviamo il set per lookup rapido O(1)
        })
    
    # Ordina: prima mescola (per rompere parità), poi ordina per dimensione
    rng.shuffle(scaffold_metadata)
    scaffold_metadata.sort(key=lambda x: x['len'], reverse=True)
    
    train_idxs, val_idxs, test_idxs = [], [], []
    train_scaffolds_list = [] # Serve per tenere traccia di cosa è nel train per il bilanciamento
    
    train_cutoff = sizes[0] * len(valid_indices)
    val_cutoff = (sizes[0] + sizes[1]) * len(valid_indices)

    # 3. Assegnazione iniziale (Greedy)
    for info in scaffold_metadata:
        if len(train_idxs) + info['len'] <= train_cutoff:
            train_idxs.extend(info['indices'])
            train_scaffolds_list.append(info)
        elif len(train_idxs) + len(val_idxs) + info['len'] <= val_cutoff:
            val_idxs.extend(info['indices'])
        else:
            test_idxs.extend(info['indices'])
    
    # 4. Funzione di Ribilanciamento Robusta
    # Questa funzione sposta scaffold dal Train al target (Val o Test) se mancano classi
    def balance_dataset(target_idxs, dataset_name):
        current_labels = set(df.loc[target_idxs, label_col].unique())
        
        # Se il target ha meno di 2 classi, ma globalmente ce ne sono almeno 2
        if len(current_labels) < 2 and len(all_valid_labels) >= 2:
            # Trova la label mancante
            missing_set = all_valid_labels - current_labels
            if not missing_set: return
            target_label = list(missing_set)[0]
            
            # Cerca nel TRAIN uno scaffold che contenga la label mancante
            candidates = [s for s in train_scaffolds_list if target_label in s['unique_labels']]
            
            # Ordina per lunghezza crescente (Smallest First) per minimizzare impatto sulle dimensioni
            candidates.sort(key=lambda x: x['len'])
            
            if candidates:
                swap_scaffold = candidates[0]
                
                # Rimuovi da Train (usiamo la rimozione logica veloce)
                train_scaffolds_list.remove(swap_scaffold)
                
                # Rimuovi gli indici dal train (ricostruzione lista veloce)
                idxs_to_remove = set(swap_scaffold['indices'])
                train_idxs[:] = [i for i in train_idxs if i not in idxs_to_remove]
                
                # Aggiungi al target
                target_idxs.extend(swap_scaffold['indices'])
                print(f"   -> Bilanciato {dataset_name}: spostato scaffold ({swap_scaffold['len']} mol) con label {target_label}")
            else:
                print(f"   Warning: Impossibile bilanciare {dataset_name} (nessuno scaffold adatto nel train)")

    # Bilancia sia Validation che Test (ordine importante)
    if len(all_valid_labels) >= 2: # Solo se è un task di classificazione
        balance_dataset(test_idxs, "TEST")
        balance_dataset(val_idxs, "VAL")
    
    # 5. RESTITUZIONE DATAFRAME (Compatibilità con il tuo codice originale)
    return df.loc[train_idxs], df.loc[val_idxs], df.loc[test_idxs]


class TransformerClassifier(nn.Module):
    def __init__(self, checkpoint: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint, add_pooling_layer=False)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.is_regression = (num_labels == 1)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = encoder_outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if self.is_regression:
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.squeeze(-1), labels.float())
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels.long())

        return SequenceClassifierOutput(loss=loss, logits=logits)

def compute_metrics(eval_pred, is_regression=False, train_labels_mean=None):
    logits, labels = eval_pred
    
    if is_regression:
        preds = np.squeeze(logits)
        
        numerator = np.sum((labels - preds) ** 2)
        
        if train_labels_mean is not None:
            denominator = np.sum((labels - train_labels_mean) ** 2)
        else:
            denominator = np.sum((labels - np.mean(labels)) ** 2)
        
        rse = float(numerator / denominator) if denominator > 0 else 0.0
        
        mse = np.mean((preds - labels) ** 2)
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(preds - labels)))
        
        return {
            "rse": rse,
            "eval_rse": rse,
            "rmse": rmse,
            "mae": mae
        }
    else:
        predictions = np.argmax(logits, axis=-1)

        try:
            logits_np = np.array(logits)
            exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            if probs.shape[1] > 1:
                pos_probs = probs[:, 1]
            else:
                pos_probs = probs[:, 0]
        except Exception:
            try:
                pos_probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
            except Exception:
                pos_probs = np.zeros(len(predictions))

        labels_np = np.array(labels).reshape(-1)
        try:
            labels_int = labels_np.astype(int)
        except Exception:
            labels_int = labels_np

        unique_classes = np.unique(labels_int)
        if len(unique_classes) < 2:
            roc_auc = float('nan')
        else:
            roc_auc = float(roc_auc_score(labels_int, pos_probs))

        return {
            "accuracy": float(accuracy_score(labels_int, predictions)),
            "roc_auc": roc_auc,
            "eval_roc_auc": roc_auc
        }

def train_model(model, train_loader, eval_loader, device, optimizer, tracker, emissions_callback, 
                epochs, is_regression, train_labels_mean, use_tqdm=True, show_batch_logs=False, quiet=False):
    epoch_iter = range(epochs) if not use_tqdm else tqdm(range(epochs), desc="Epochs")
    for epoch in epoch_iter:
        model.train()
        train_iter = train_loader if not use_tqdm else tqdm(train_loader, desc=f"Train E{epoch+1}", leave=False)
        for i, batch in enumerate(train_iter):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if show_batch_logs and (i + 1) % 10 == 0:
                try:
                    if use_tqdm:
                        tqdm.write(f"Epoch {epoch+1}/{epochs} - batch {i+1} - loss: {loss.item():.4f}")
                    else:
                        print(f"Epoch {epoch+1}/{epochs} - batch {i+1} - loss: {loss.item():.4f}")
                except Exception:
                    pass
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            eval_iter = eval_loader if not use_tqdm else tqdm(eval_loader, desc=f"Eval E{epoch+1}", leave=False)
            for batch in eval_iter:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                all_logits.append(outputs.logits.cpu())
                all_labels.append(batch['labels'].cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics((all_logits.numpy(), all_labels.numpy()), 
                                 is_regression=is_regression, 
                                 train_labels_mean=train_labels_mean)
        if use_tqdm:
            tqdm.write(f"Epoch {epoch+1}: {metrics}")
        else:
            print(f"Epoch {epoch+1}: {metrics}")

        if tracker:
            current_emissions = getattr(tracker, '_total_emissions', tracker._total_energy.kWh * getattr(tracker, '_country_iso_code_intensity', 0.5))
        else:
            current_emissions = 0

        if emissions_callback:
            if emissions_callback.check_early_stopping(metrics, current_emissions):
                return metrics, (epoch + 1)
    return metrics, epochs


def main(dataset, alpha, warmup_epochs, model_name, progress_mode='epoch', show_batch_logs=False, quiet=False):
    df = load_dataset("transformers", dataset)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    is_regression = dataset in REGRESSION_DATASETS
    
    df = df.reset_index(drop=True)
    train_df, eval_df, test_df = scaffold_split(df, smiles_col="smiles", sizes=(0.8, 0.1, 0.1))

    train_labels_mean = train_df["label"].mean() if is_regression else None

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    eval_dataset = Dataset.from_pandas(eval_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
    
    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Modello sconosciuto: {model_name}")
    model_checkpoint = MODEL_PATHS.get(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        return tokenizer(examples["smiles"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["smiles"])
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["smiles"])
    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["smiles"])

    tokenized_train.set_format("torch")
    tokenized_eval.set_format("torch")
    tokenized_test.set_format("torch")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_labels = 1 if is_regression else int(df["label"].nunique())

    for use_early in [False, True]:
        model = TransformerClassifier(model_checkpoint, num_labels)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=HYPERPARAMS["lr"])
        
        train_loader = DataLoader(tokenized_train, batch_size=HYPERPARAMS["batch_size"], shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
        eval_loader = DataLoader(tokenized_eval, batch_size=HYPERPARAMS["batch_size"], num_workers=2, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(tokenized_test, batch_size=HYPERPARAMS["batch_size"], num_workers=2, pin_memory=True, persistent_workers=True)

        project_name = f"{model_name}_early" if use_early else f"{model_name}_classic"
        tracker = EmissionsTracker(project_name=project_name, output_dir=f"./emissions_{dataset}", tracking_mode="process", log_level="critical")
        tracker.start()

        # Passa is_regression al callback
        emissions_callback = EmissionsEarlyStoppingCallback(tracker, alpha=alpha, warmup_epochs=warmup_epochs, is_regression=is_regression, classic=not use_early)

        use_tqdm = (progress_mode == 'bar')
        eval_results, epochs_completed = train_model(model, train_loader, eval_loader, device, optimizer, tracker, 
                      emissions_callback, HYPERPARAMS["epochs"], 
                      is_regression=is_regression, train_labels_mean=train_labels_mean,
                      use_tqdm=use_tqdm, show_batch_logs=show_batch_logs, quiet=quiet)

        emissions = tracker.stop()

        # Valutazione finale su test set
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                all_logits.append(outputs.logits.cpu())
                all_labels.append(batch['labels'].cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        final_metrics = compute_metrics((all_logits.numpy(), all_labels.numpy()), 
                           is_regression=is_regression,
                           train_labels_mean=train_labels_mean)
        print(f"Metriche finali su test set: {final_metrics}")

        if is_regression:
            auroc = float('nan')
            rse = final_metrics.get("rse", float('nan'))
            rmse = final_metrics.get("rmse", float('nan'))
        else:
            auroc = final_metrics.get("roc_auc", final_metrics.get("eval_roc_auc", float('nan')))
            rse = float('nan')
            rmse = float('nan')
        
        emissions_value = emissions
        results_df = pd.DataFrame([{
            "experiment": project_name,
            "emissions": emissions_value,
            "auroc": auroc,
            "rse": rse,
            "rmse": rmse,
            "Epoche fornite": HYPERPARAMS["epochs"],
            "epoche usate": int(epochs_completed),
            "warmup": warmup_epochs
        }])
        
        results_df.to_csv(f"experiments_{dataset}.csv", mode='a', header=not os.path.exists(f"experiments_{dataset}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="bace",
        help="Nome del dataset da utilizzare (default: bace).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Smoothing factor per EMA.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Epoche di warm-up per l'early stopping.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="chemberta",
        help="Modello da utilizzare (default: chemberta).",
    )
    parser.add_argument(
        "--progress",
        choices=["bar", "epoch", "none"],
        default="bar",
        help="Tipo di avanzamento: 'bar' per tqdm, 'epoch' per stampare solo epoche, 'none' per niente",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Se impostato, non stampa i riepiloghi di epoca",
    )
    parser.add_argument(
        "--show-batch-logs",
        action="store_true",
        help="Stampa log per batch ogni 10 batch (utile per debug)",
    )
    args = parser.parse_args()

    main(args.dataset, args.alpha, args.warmup_epochs, args.model, progress_mode=args.progress, show_batch_logs=args.show_batch_logs, quiet=args.quiet)