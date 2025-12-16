import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
import os
import sys
import glob
from transformers import AutoModel, AutoConfig

warnings.filterwarnings('ignore')

# ============================================================================
# UTILITIES
# ============================================================================

def get_model_metadata(model_name):
    MODEL_PATHS = {
        "chemberta": "seyonec/ChemBERTa-zinc-base-v1",
        "chemberta2": "DeepChem/ChemBERTa-77M-MLM",
        "selformer": "HUBioDataLab/SELFormer",
        "smilesbert": "JuIm/SMILES_BERT"
    }
    
    metadata = {
        'model_type': model_name,
        'model_family': 'unknown',
        'num_parameters': None,
        'hidden_size': None,
        'num_layers': None
    }
    
    if 'graphmae' in model_name.lower():
        metadata['model_family'] = 'gnn'
        metadata['num_layers'] = 5
        metadata['hidden_size'] = 300
        metadata['num_parameters'] = 5 * (300 * 300 * 3 + 300 * 2)
        return metadata
    
    if model_name in MODEL_PATHS:
        try:
            config = AutoConfig.from_pretrained(MODEL_PATHS[model_name])
            metadata['model_family'] = 'transformer'
            metadata['hidden_size'] = config.hidden_size
            metadata['num_layers'] = config.num_hidden_layers
            
            try:
                model = AutoModel.from_pretrained(MODEL_PATHS[model_name])
                metadata['num_parameters'] = sum(p.numel() for p in model.parameters())
                del model
            except Exception:
                H = metadata['hidden_size']
                L = metadata['num_layers']
                metadata['num_parameters'] = L * (12 * H * H + 13 * H)
        except Exception: pass
    
    return metadata

def get_dataset_metadata(dataset_name, data_dir="./data/dataset_finetuning"):
    metadata = {
        'dataset_size': None,
        'num_tasks': None,          # Verrà rimosso dopo
        'is_regression': None,
        'is_multitask': None,       # Verrà rimosso dopo
        'avg_molecular_weight': None,
        'dataset_complexity': None  # Verrà rimosso dopo
    }
    
    REGRESSION_DATASETS = {"cep", "malaria", "lipophilicity", "delaney", "esol", "freesolv"}
    metadata['is_regression'] = 1 if dataset_name.lower() in REGRESSION_DATASETS else 0
    
    base_dir = os.path.join(data_dir, dataset_name)
    raw_csv = os.path.join(base_dir, "raw", f"{dataset_name}.csv")
    processed_csv = os.path.join(base_dir, "processed", "smiles.csv")
    
    if os.path.exists(processed_csv):
        try:
            df_proc = pd.read_csv(processed_csv, header=None)
            metadata['dataset_size'] = len(df_proc)
        except: metadata['dataset_size'] = 5000
    elif os.path.exists(raw_csv):
        try:
            df_raw = pd.read_csv(raw_csv)
            metadata['dataset_size'] = len(df_raw)
        except: metadata['dataset_size'] = 5000
    else:
        metadata['dataset_size'] = 5000
    
    if os.path.exists(raw_csv):
        try:
            df_raw = pd.read_csv(raw_csv)
            metadata['num_tasks'] = 1 
            
            if ('smiles' in df_raw.columns or 'mol' in df_raw.columns) and metadata['avg_molecular_weight'] is None:
                from rdkit import Chem
                from rdkit.Chem import Descriptors
                smiles_col = 'smiles' if 'smiles' in df_raw.columns else 'mol'
                weights = []
                for smi in df_raw[smiles_col].dropna().head(200):
                    mol = Chem.MolFromSmiles(smi)
                    if mol: weights.append(Descriptors.MolWt(mol))
                if weights:
                    metadata['avg_molecular_weight'] = np.mean(weights)
        except Exception: pass
    
    if metadata['avg_molecular_weight'] is None: metadata['avg_molecular_weight'] = 300.0
    
    return metadata

def extract_model_features_from_name(experiment_name, model_cache={}):
    features = {}
    model_name = None
    for candidate in ['chemberta2', 'chemberta', 'selformer', 'smilesbert', 'graphmae']:
        if candidate in experiment_name.lower():
            model_name = candidate
            break
    if model_name is None: model_name = 'unknown'
    if model_name not in model_cache:
        model_cache[model_name] = get_model_metadata(model_name)
    
    metadata = model_cache[model_name]
    features.update(metadata)
    features['uses_early_stopping'] = 1 if 'early' in experiment_name.lower() else 0
    return features

def build_feature_matrix(csv_path, dataset_name, data_dir="./data/dataset_finetuning"):
    df = pd.read_csv(csv_path)
    dataset_metadata = get_dataset_metadata(dataset_name, data_dir)
    
    feature_rows = []
    model_cache = {}
    
    for idx, row in df.iterrows():
        features = {}
        model_features = extract_model_features_from_name(row['experiment'], model_cache)
        features.update(model_features)
        features.update(dataset_metadata)
        
        features['epochs_provided'] = row.get('Epoche fornite', row.get('epochs_provided', 100))
        features['epochs_used'] = row.get('epoche usate', row.get('epochs_used', features['epochs_provided']))
        features['warmup_epochs'] = row.get('warmup', 30)
        
        if features['epochs_provided'] > 0:
            features['epoch_efficiency'] = features['epochs_used'] / features['epochs_provided']
        else:
            features['epoch_efficiency'] = 1.0
        
        features['emissions'] = row['emissions']
        
        # Performance standardization
        if dataset_metadata['is_regression']:
            rse = row.get('rse', row.get('eval_rse', np.nan))
            rmse = row.get('rmse', row.get('eval_rmse', np.nan))
            if not np.isnan(rse) and rse > 0: features['performance'] = 1.0 / rse
            elif not np.isnan(rmse) and rmse > 0: features['performance'] = 1.0 / rmse
            else: features['performance'] = np.nan
        else:
            auroc = row.get('auroc', row.get('roc_auc', row.get('eval_roc_auc', np.nan)))
            acc = row.get('accuracy', row.get('eval_accuracy', np.nan))
            if not np.isnan(auroc): features['performance'] = auroc
            elif not np.isnan(acc): features['performance'] = acc
            else: features['performance'] = np.nan

        feature_rows.append(features)
    
    return pd.DataFrame(feature_rows).dropna(subset=['emissions', 'performance'])

# ============================================================================
# STATISTICA: Correlazione Parziale
# ============================================================================

def get_residuals(x, y, covariates):
    """Calcola i residui di X e Y rispetto alle covariate."""
    if covariates.shape[1] == 0:
        return x, y
        
    # Residui X
    model_x = LinearRegression().fit(covariates, x)
    res_x = x - model_x.predict(covariates)
    
    # Residui Y
    model_y = LinearRegression().fit(covariates, y)
    res_y = y - model_y.predict(covariates)
    
    return res_x, res_y

def calculate_feature_contributions_partial(df_features):
    """Calcola l'importanza usando Correlazione Parziale."""
    FORBIDDEN_COLS = ['num_tasks', 'is_multitask', 'dataset_complexity']
    IGNORE_COLS = ['emissions', 'performance', 'efficiency_score', 'model_type', 'model_family', 'dataset_origin', 'is_regression']
    
    feature_cols = [c for c in df_features.columns 
                   if c not in IGNORE_COLS 
                   and c not in FORBIDDEN_COLS
                   and pd.api.types.is_numeric_dtype(df_features[c])]
    
    valid_features = [c for c in feature_cols if df_features[c].std() > 1e-10]
    
    print(f"   ℹ️  Features incluse nell'analisi parziale: {len(valid_features)}")
    print(f"       {valid_features}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features[valid_features].fillna(0))
    df_scaled = pd.DataFrame(X_scaled, columns=valid_features)
    
    y_emissions = df_features['emissions'].values
    y_performance = df_features['performance'].values 
    
    p_corr_emissions = {}
    p_corr_performance = {}
    
    print("\n   ⏳ Calcolo correlazioni parziali...")
    
    for feat in valid_features:
        other_feats = [f for f in valid_features if f != feat]
        Z = df_scaled[other_feats].values
        x_vec = df_scaled[feat].values
        
        # Calcolo residui e correlazione
        res_x_em, res_y_em = get_residuals(x_vec, y_emissions, Z)
        p_corr_emissions[feat] = np.corrcoef(res_x_em, res_y_em)[0, 1]
        
        res_x_perf, res_y_perf = get_residuals(x_vec, y_performance, Z)
        p_corr_performance[feat] = np.corrcoef(res_x_perf, res_y_perf)[0, 1]
        
    p_corr_emissions = pd.Series(p_corr_emissions).fillna(0)
    p_corr_performance = pd.Series(p_corr_performance).fillna(0)
    
    importance_emissions = p_corr_emissions.abs()
    importance_performance = p_corr_performance.abs()
    
    # Normalizzazione per visualizzazione
    importance_emissions = importance_emissions / (importance_emissions.sum() + 1e-10)
    importance_performance = importance_performance / (importance_performance.sum() + 1e-10)
    
    return {
        'features': valid_features,
        'corr_emissions': p_corr_emissions,
        'corr_performance': p_corr_performance,
        'importance_emissions': importance_emissions,
        'importance_performance': importance_performance,
        'scaled_data': df_scaled,
        'raw_emissions': y_emissions,
        'raw_performance': y_performance
    }

def create_plots(analysis, output_dir="."):
    # --- 1. BAR CHARTS (Importanza) ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    imp_em = analysis['importance_emissions'].sort_values()
    axes[0].barh(imp_em.index, imp_em.values, color='orangered')
    axes[0].set_title('Importanza (Partial Correlation) su Emissioni', fontweight='bold')
    axes[0].set_xlabel('Contributo Unico Relativo')
    for i, (feat, val) in enumerate(imp_em.items()):
        raw_corr = analysis['corr_emissions'][feat]
        axes[0].text(val, i, f" $\\rho$={raw_corr:.2f}", va='center', fontsize=9)

    imp_perf = analysis['importance_performance'].sort_values()
    axes[1].barh(imp_perf.index, imp_perf.values, color='green')
    axes[1].set_title('Importanza (Partial Correlation) su Performance', fontweight='bold')
    axes[1].set_xlabel('Contributo Unico Relativo')
    for i, (feat, val) in enumerate(imp_perf.items()):
        raw_corr = analysis['corr_performance'][feat]
        axes[1].text(val, i, f" $\\rho$={raw_corr:.2f}", va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'partial_correlation_importance.png'), dpi=300)
    plt.close()
    
    # --- 2. SCATTER PLOTS (Partial Regression Plots) ---
    # Mostriamo solo le top 6 features per chiarezza
    top_features = analysis['importance_emissions'].sort_values(ascending=False).head(6).index.tolist()
    n_feats = len(top_features)
    
    if n_feats > 0:
        fig, axes = plt.subplots(2, n_feats, figsize=(4*n_feats, 8))
        if n_feats == 1: axes = np.array([axes]).T # Gestione caso 1 feature
        
        df_scaled = analysis['scaled_data']
        valid_features = analysis['features']
        
        # Riga 1: Emissioni
        for i, feat in enumerate(top_features):
            other_feats = [f for f in valid_features if f != feat]
            Z = df_scaled[other_feats].values
            x_vec = df_scaled[feat].values
            y_vec = analysis['raw_emissions']
            
            # Ricalcolo residui per il plot
            res_x, res_y = get_residuals(x_vec, y_vec, Z)
            
            ax = axes[0, i]
            ax.scatter(res_x, res_y, alpha=0.5, color='orangered')
            
            # Trendline
            z = np.polyfit(res_x, res_y, 1)
            p = np.poly1d(z)
            rng = np.linspace(res_x.min(), res_x.max(), 100)
            ax.plot(rng, p(rng), "--", color='black', alpha=0.5)
            
            ax.set_title(f"{feat}\n(vs Emissioni)", fontsize=10, fontweight='bold')
            ax.set_xlabel(f"Residui {feat}")
            if i == 0: ax.set_ylabel("Residui Emissioni")
            ax.grid(True, alpha=0.3)

        # Riga 2: Performance
        top_features_perf = analysis['importance_performance'].sort_values(ascending=False).head(n_feats).index.tolist()
        
        for i, feat in enumerate(top_features_perf):
            other_feats = [f for f in valid_features if f != feat]
            Z = df_scaled[other_feats].values
            x_vec = df_scaled[feat].values
            y_vec = analysis['raw_performance']
            
            res_x, res_y = get_residuals(x_vec, y_vec, Z)
            
            ax = axes[1, i]
            ax.scatter(res_x, res_y, alpha=0.5, color='green')
            
            z = np.polyfit(res_x, res_y, 1)
            p = np.poly1d(z)
            rng = np.linspace(res_x.min(), res_x.max(), 100)
            ax.plot(rng, p(rng), "--", color='black', alpha=0.5)
            
            ax.set_title(f"{feat}\n(vs Performance)", fontsize=10, fontweight='bold')
            ax.set_xlabel(f"Residui {feat}")
            if i == 0: ax.set_ylabel("Residui Performance")
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'partial_correlation_scatter.png'), dpi=300)
        plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main(data_dir="./data/dataset_finetuning", output_dir="."):
    files = glob.glob("experiments_*.csv")
    if not files:
        print("❌ Nessun file experiments_*.csv trovato.")
        return

    print("="*70)
    print(f"🔬 PARTIAL CORRELATION ANALYSIS - {len(files)} Datasets")
    print("="*70)
    
    all_dfs = []
    for f in files:
        dname = os.path.basename(f).replace("experiments_", "").replace(".csv", "")
        try:
            all_dfs.append(build_feature_matrix(f, dname, data_dir))
        except Exception: pass
        
    global_df = pd.concat(all_dfs, ignore_index=True)
    print(f"📊 Dataset Globale: {len(global_df)} righe")
    
    # Analisi
    analysis = calculate_feature_contributions_partial(global_df)
    create_plots(analysis, output_dir)
    
    print("\n✅ Analisi completata.")
    print("   📁 partial_correlation_importance.png (Bar Chart)")
    print("   📁 partial_correlation_scatter.png    (Residual Plots - Added Variable Plots)")

if __name__ == "__main__":
    main()