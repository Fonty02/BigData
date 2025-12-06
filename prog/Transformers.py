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
import logging
from collections import defaultdict

# --- RDKit Imports per Scaffold Splitting ---
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
    "epochs": 10
}

# --- FUNZIONE PER SCAFFOLD SPLITTING ---
def scaffold_split(df, smiles_col='smiles', sizes=(0.8, 0.1, 0.1), seed=42):
    """
    Divide il dataset basandosi sugli scaffold di Murcko (struttura chimica).
    Le molecole con lo stesso scaffold finiscono nello stesso set.
    Molecole con SMILES invalidi vengono eliminate (come da slide 9).
    """
    # 1. Filtra molecole invalide prima di processare
    valid_indices = []
    scaffolds = defaultdict(list)
    
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is not None:
            # Calcola lo scaffold di Murcko
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds[scaffold].append(idx)
            valid_indices.append(idx)
        else:
            # Elimina molecole non valide (come da slide 9: "Eliminazione di molecole con feature non valide")
            print(f"Warning: Eliminata molecola con SMILES invalido all'indice {idx}")

    # Usa solo le molecole valide
    df_valid = df.loc[valid_indices]
    print(f"Molecole valide: {len(df_valid)}/{len(df)} ({len(df)-len(df_valid)} eliminate)")

    # 2. Ordina gli scaffold dal più grande al più piccolo
    scaffold_sets = list(scaffolds.values())
    scaffold_sets.sort(key=len, reverse=True)

    train_idxs, val_idxs, test_idxs = [], [], []
    train_cutoff = sizes[0] * len(df_valid)
    val_cutoff = (sizes[0] + sizes[1]) * len(df_valid)

    # 3. Distribuisci i gruppi nei set
    for group in scaffold_sets:
        if len(train_idxs) + len(group) <= train_cutoff:
            train_idxs.extend(group)
        elif len(train_idxs) + len(val_idxs) + len(group) <= val_cutoff:
            val_idxs.extend(group)
        else:
            test_idxs.extend(group)
    # Garantire che ciascuna partizione contenga almeno due classi (se il dataset ne contiene almeno 2)
    all_labels = df.loc[valid_indices, 'label'].unique()
    n_classes_overall = len(np.unique(all_labels))

    def ensure_min_two_classes(receiver_idxs, receiver_name):
        receiver_labels = np.unique(df.loc[receiver_idxs, 'label'].values) if len(receiver_idxs) > 0 else np.array([])
        if n_classes_overall >= 2 and len(receiver_labels) < 2:
            missing = [c for c in np.unique(all_labels) if c not in receiver_labels]
            # Cerchiamo esempi da spostare preferibilmente dal train
            for m in missing:
                donor_found = False
                # Priorità: train -> test -> val (non spostare dall'holder stesso)
                for donor_list in (train_idxs, test_idxs, val_idxs):
                    if donor_list is receiver_idxs:
                        continue
                    # cerca un indice nel donor_list che abbia label == m
                    for j, idx in enumerate(list(donor_list)):
                        try:
                            if df.at[idx, 'label'] == m:
                                # sposta questo indice nel receiver
                                donor_list.pop(j)
                                receiver_idxs.append(idx)
                                print(f"Bilanciamento split: spostato indice {idx} (label={m}) in {receiver_name} da donor")
                                donor_found = True
                                break
                        except Exception:
                            continue
                    if donor_found:
                        break

    # Applichiamo il bilanciamento per val e test, preferendo prendere esempi dal train
    ensure_min_two_classes(val_idxs, 'Val')
    ensure_min_two_classes(test_idxs, 'Test')

    print(f"Scaffold Split Result: Train {len(train_idxs)}, Val {len(val_idxs)}, Test {len(test_idxs)}")
    
    return df.loc[train_idxs], df.loc[val_idxs], df.loc[test_idxs]
# ---------------------------------------------


class TransformerClassifier(nn.Module):
    """ChemBERTa encoder with a single fully connected classification head."""

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
    """
    Calcola le metriche di VALUTAZIONE secondo la slide 10:
    - Classificazione: ROC-AUC
    - Regressione: RSE (Relative Squared Error)
    
    Nota: La loss function (slide 8) è diversa:
    - Classificazione usa BCE durante il training
    - Regressione usa MSE durante il training
    """
    logits, labels = eval_pred
    
    if is_regression:
        # Regressione: calcola RSE come METRICA DI VALUTAZIONE (slide 10)
        preds = np.squeeze(logits)
        
        # RSE = sum((y_test - y_pred)^2) / sum((y_test - mean(y_train))^2)
        numerator = np.sum((labels - preds) ** 2)
        
        if train_labels_mean is not None:
            denominator = np.sum((labels - train_labels_mean) ** 2)
        else:
            # Fallback: usa la media dei labels di test (meno accurato)
            denominator = np.sum((labels - np.mean(labels)) ** 2)
        
        rse = float(numerator / denominator) if denominator > 0 else 0.0
        
        # Calcola anche RMSE per riferimento (la loss usa MSE durante training)
        mse = np.mean((preds - labels) ** 2)
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(preds - labels)))
        
        return {
            "rse": rse,
            "eval_rse": rse,  # Per compatibilità con early stopping
            "rmse": rmse,     # Radice quadrata della loss MSE
            "mae": mae
        }
    else:
        # Classificazione: calcola ROC-AUC come da slide 10
        # Predizioni discrete
        predictions = np.argmax(logits, axis=-1)

        # Calcolo probabilità classe positiva in modo numericamente stabile
        try:
            # logits potrebbe essere numpy array
            logits_np = np.array(logits)
            # Softmax numericamente stabile
            exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            # prendiamo la probabilità della classe positiva (indice 1 se presente)
            if probs.shape[1] > 1:
                pos_probs = probs[:, 1]
            else:
                # se il modello ha una sola uscita, assumiamo regressione-binarizzata
                pos_probs = probs[:, 0]
        except Exception:
            # Fallback: usa torch softmax se possibile
            try:
                pos_probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
            except Exception:
                pos_probs = np.zeros(len(predictions))

        # Assicuriamoci che le labels siano array 1D di interi
        labels_np = np.array(labels).reshape(-1)
        try:
            labels_int = labels_np.astype(int)
        except Exception:
            labels_int = labels_np

        # Se nel batch/test set è presente una sola classe, roc_auc_score non è definita
        unique_classes = np.unique(labels_int)
        if len(unique_classes) < 2:
            # Meglio ritornare NaN e loggare la situazione piuttosto che 0.0 (che è fuorviante)
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
                # epochs_completed = epoch+1
                return metrics, (epoch + 1)
    # completed all epochs
    return metrics, epochs


def main(dataset, alpha, warmup_epochs, model_name, progress_mode='epoch', show_batch_logs=False, quiet=False):
    print(f"--- Fine-tuning {model_name} sul dataset {dataset} ---")
    df = load_dataset("transformers", dataset)
    # Rendiamo l'esperimento riproducibile
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Determina se regressione
    is_regression = dataset in REGRESSION_DATASETS
    
    # Scaffold Splitting con eliminazione molecole invalide
    print(f"Eseguendo Scaffold Splitting (80/10/10) su {len(df)} molecole...")
    df = df.reset_index(drop=True)
    train_df, eval_df, test_df = scaffold_split(df, smiles_col="smiles", sizes=(0.8, 0.1, 0.1))

    # Calcola la media delle label di training per RSE (necessario per regressione)
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
        tracker = EmissionsTracker(project_name=project_name, output_dir=f"./emissions_{dataset}", tracking_mode="process")
        tracker.start()

        # Passa is_regression al callback
        emissions_callback = EmissionsEarlyStoppingCallback(tracker, alpha=alpha, warmup_epochs=warmup_epochs, is_regression=is_regression) if use_early else None

        print(f"Inizio training {'con' if use_early else 'senza'} early stopping...")
        use_tqdm = (progress_mode == 'bar')
        eval_results, epochs_completed = train_model(model, train_loader, eval_loader, device, optimizer, tracker, 
                      emissions_callback, HYPERPARAMS["epochs"], 
                      is_regression=is_regression, train_labels_mean=train_labels_mean,
                      use_tqdm=use_tqdm, show_batch_logs=show_batch_logs, quiet=quiet)

        emissions = tracker.stop()
        print(f"Emissioni stimate: {emissions:.6f} kg CO2eq")
        print(f"Risultati Finali (val): {eval_results} -- Epoche eseguite: {epochs_completed}")

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

        # Salva risultati su CSV
        if is_regression:
            auroc = float('nan')
            rse = final_metrics.get("rse", float('nan'))
            rmse = final_metrics.get("rmse", float('nan'))
        else:
            auroc = final_metrics.get("roc_auc", final_metrics.get("eval_roc_auc", float('nan')))
            rse = float('nan')
            rmse = float('nan')
        # Avvisa se AUROC non è definita (es. solo una classe nel test set)
        try:
            if not is_regression and (auroc is None or (isinstance(auroc, float) and np.isnan(auroc))):
                print("Warning: AUROC non definita sul test set (probabilmente presente una sola classe). Verificare lo split dei dati.")
        except Exception:
            pass
        emissions_value = emissions
        results_df = pd.DataFrame([{
            "experiment": project_name,
            "emissions": emissions_value,
            "auroc": auroc,
            "rse": rse,
            "rmse": rmse,
            "epochs_completed": int(epochs_completed)
        }])
        
        print(f"Saving experiment: {project_name}, emissions={emissions_value:.6f}, auroc={auroc}, rse={rse}, rmse={rmse}")
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
        default=1,
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
        default="epoch",
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