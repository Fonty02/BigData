import argparse
import os
import inspect
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from codecarbon import EmissionsTracker
except ImportError:
    EmissionsTracker = None

from transformers import TrainerCallback


HYPERPARAMS = {
    "lr": 1e-4,
    "batch_size": 32,
    "epochs": 10,
}


class EmissionsEarlyStoppingCallback(TrainerCallback):
    """
    Early stopping basato su Adaptive Accuracy-Emission Ratio (AN-GES dal report.pdf).
    Ferma il training se AER_current < beta * EMA_AER.
    """

    def __init__(self, tracker, alpha=0.9, beta=0.2, warmup_epochs=3):
        self.tracker = tracker
        self.alpha = alpha  # Smoothing per EMA
        self.beta = beta    # Soglia moltiplicativa
        self.warmup_epochs = warmup_epochs
        self.prev_performance = None
        self.prev_emissions = None
        self.ema_aer = 1e-6  # Inizializzazione piccola
        self.epoch = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.epoch += 1
        if self.tracker is None or metrics is None:
            return

        # Ottieni emissioni cumulative attuali
        current_emissions = getattr(self.tracker, '_total_emissions', self.tracker._total_energy.kWh * getattr(self.tracker, '_country_iso_code_intensity', 0.5))  # Fallback

        # Usa ROC-AUC come performance (dal compute_metrics)
        current_performance = metrics.get("eval_roc_auc", 0)

        if self.prev_performance is not None and self.prev_emissions is not None:
            delta_perf = (current_performance - self.prev_performance) / abs(self.prev_performance) * 100
            delta_emiss = (current_emissions - self.prev_emissions) / abs(self.prev_emissions) * 100 if self.prev_emissions > 0 else 0

            if delta_emiss != 0:
                aer_current = delta_perf / delta_emiss
            else:
                aer_current = 0  # Evita divisione per zero

            if self.epoch >= self.warmup_epochs:
                self.ema_aer = self.alpha * aer_current + (1 - self.alpha) * self.ema_aer
                if aer_current < self.beta * self.ema_aer:
                    print(f"Early stopping at epoch {self.epoch}: AER {aer_current:.4f} < {self.beta} * EMA {self.ema_aer:.4f}")
                    control.should_training_stop = True
            else:
                # Accumula per inizializzare EMA
                self.ema_aer = (self.ema_aer + aer_current) / 2

        self.prev_performance = current_performance
        self.prev_emissions = current_emissions

# Parametri dal PDF (Pagina 8) per Encoder pre-addestrati
# Learning Rate: 1e-4, Batch Size: 32, Epoche: 10
HYPERPARAMS = {
    "lr": 1e-4,
    "batch_size": 32,
    "epochs": 10
}


class ChemBERTaClassifier(nn.Module):
    """ChemBERTa encoder with a single fully connected classification head."""

    def __init__(self, checkpoint: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint, add_pooling_layer=False)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = encoder_outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "roc_auc": roc_auc_score(labels, probs) # Metrica principale per BACE (Pagina 10)
    }


def load_bace_dataframe(dataset_path: str) -> pd.DataFrame:
    """Unisce gli SMILES processati con le etichette di classe dal file raw."""

    processed_csv = os.path.join(dataset_path, "processed", "smiles.csv")
    raw_csv = os.path.join(dataset_path, "raw", "bace.csv")

    smiles_df = pd.read_csv(processed_csv, header=None, names=["smiles"])
    raw_df = pd.read_csv(raw_csv, usecols=["mol", "Class"])
    raw_df = raw_df.rename(columns={"mol": "smiles", "Class": "label"})

    merged = smiles_df.merge(raw_df, on="smiles", how="left")
    missing = merged["label"].isna().sum()
    if missing:
        raise ValueError(f"Etichette mancanti per {missing} molecole. Controlla i file raw/processed.")

    merged["label"] = merged["label"].astype(int)
    return merged

def main(dataset_path, early_stopping=False, alpha=0.9, beta=0.2, warmup_epochs=3):
    print(f"--- Fine-tuning ChemBERTa-2 su {dataset_path} ---")
    
    try:
        df = load_bace_dataframe(dataset_path)
        print(f"Dataset caricato: {len(df)} molecole con etichetta.")
    except FileNotFoundError as exc:
        print(f"Errore: {exc}")
        return

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    # 2. Tokenizzazione
    model_checkpoint = "DeepChem/ChemBERTa-77M-MLM" 
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        return tokenizer(examples["smiles"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["smiles"])
    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["smiles"])

    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")

    # 3. Inizializzazione Modello (1 FC per la classificazione)
    model = ChemBERTaClassifier(model_checkpoint, num_labels=2)

    # 4. Configurazione Training (Parametri dal PDF)
    all_kwargs = dict(
        output_dir="./results_chemberta",
        learning_rate=HYPERPARAMS["lr"],
        per_device_train_batch_size=HYPERPARAMS["batch_size"],
        per_device_eval_batch_size=HYPERPARAMS["batch_size"],
        num_train_epochs=HYPERPARAMS["epochs"],
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir='./logs',
    )

    supported_args = set(inspect.signature(TrainingArguments).parameters.keys())
    filtered_kwargs = {k: v for k, v in all_kwargs.items() if k in supported_args}

    # Handle evaluation and save strategies for compatibility
    if "eval_strategy" in supported_args:
        filtered_kwargs["eval_strategy"] = "epoch"
        filtered_kwargs["save_strategy"] = "epoch"
    elif "evaluation_strategy" in supported_args:
        filtered_kwargs["evaluation_strategy"] = "epoch"
        filtered_kwargs["save_strategy"] = "epoch"
    else:
        # Fallback for very old versions
        if "evaluate_during_training" in supported_args:
            filtered_kwargs["evaluate_during_training"] = True
        if "save_steps" in supported_args:
            filtered_kwargs["save_steps"] = len(tokenized_train) // HYPERPARAMS["batch_size"]  # Save every epoch approx

    training_args = TrainingArguments(**filtered_kwargs)

    tracker = None
    project_name = "chemberta2_bace_early" if early_stopping else "chemberta2_bace_classic"
    if EmissionsTracker:
        tracker = EmissionsTracker(project_name=project_name, output_dir="./emissions")
        tracker.start()
    else:
        print("Avviso: codecarbon non installato, emissioni non tracciate.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    if tracker and early_stopping:
        emissions_callback = EmissionsEarlyStoppingCallback(tracker, alpha=alpha, beta=beta, warmup_epochs=warmup_epochs)
        trainer.add_callback(emissions_callback)

    print("Inizio training...")
    trainer.train()

    if tracker:
        emissions = tracker.stop()
        print(f"Emissioni stimate: {emissions:.6f} kg CO2eq")

    eval_results = trainer.evaluate()
    print(f"Risultati Finali: {eval_results}")

    # Salva risultati su CSV
    auroc = eval_results.get("eval_roc_auc", 0)
    emissions_value = emissions if tracker else 0
    results_df = pd.DataFrame([{
        "experiment": project_name,
        "emissions": emissions_value,
        "auroc": auroc
    }])
    results_df.to_csv("experiments.csv", mode='a', header=not os.path.exists("experiments.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/dataset_finetuning/bace",
        help="Percorso alla cartella del dataset (contiene raw/ e processed/)",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Abilita early stopping basato su emissioni (AN-GES).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Smoothing factor per EMA in AN-GES.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.2,
        help="Soglia moltiplicativa per AN-GES.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=3,
        help="Epoche di warm-up per AN-GES.",
    )
    args = parser.parse_args()

    main(args.dataset, early_stopping=args.early_stopping, alpha=args.alpha, beta=args.beta, warmup_epochs=args.warmup_epochs)