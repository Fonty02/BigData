# shap_analysis.py - DA CREARE
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_all_experiments():
    """Carica tutti i CSV esperimenti"""
    datasets = ['bace', 'cep', 'lipophilicity', 'malaria']
    all_data = []
    
    for ds in datasets:
        df = pd.read_csv(f'experiments_{ds}.csv')
        df['dataset'] = ds
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def engineer_features(df):
    """Feature engineering per SHAP"""
    df['model_family'] = df['experiment'].str.extract(r'([a-z]+)_')[0]
    df['uses_early'] = df['experiment'].str.contains('early').astype(int)
    df['efficiency'] = df['auroc'] / (df['emissions'] + 1e-8)  # Per classificazione
    df['efficiency_reg'] = 1 / (df['rse'] * df['emissions'] + 1e-8)  # Per regressione
    df['epoch_ratio'] = df['epoche usate'] / df['Epoche fornite']
    df['is_regression'] = df['dataset'].isin(['cep', 'malaria', 'lipophilicity']).astype(int)
    
    return df

# 1. SHAP per Early Stopping Decision
def explain_early_stopping(df):
    """Spiega quando l'early stopping è vantaggioso"""
    features = ['warmup', 'epoch_ratio', 'emissions', 'is_regression']
    
    # Rimuovi righe con NaN nelle features o target
    df_clean = df[features + ['uses_early']].dropna()
    
    if len(df_clean) < 10:
        print("⚠️  Troppo pochi dati per analisi early stopping, skip.")
        return None, None
    
    X = df_clean[features].astype(float)
    y = df_clean['uses_early']
    
    if y.nunique() < 2:
        print("⚠️  Solo una classe per early stopping, skip.")
        return None, None
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Migliora leggibilità: dimensioni, font, dpi
    plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12})

    # Summary plot (dot) - mostra distribuzione SHAP per feature
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP: When is Early Stopping Beneficial?\n(Distribuzione dei contributi per feature)")
    plt.tight_layout()
    plt.savefig("shap_early_stopping.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Bar plot: mean(|SHAP|) per feature per evidenziare importanza assoluta
    # Gestione dei diversi formati di shap_values:
    # - regressori -> shap_values.values.shape == (n_samples, n_features)
    # - classificatori -> shap_values.values.shape == (n_samples, n_classes, n_features)
    vals = shap_values.values
    if vals.ndim == 3:
        # vals shape: (n_samples, n_features, n_classes) -> aggrega su samples e classes per ottenere importanza per feature
        mean_abs = np.abs(vals).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(vals).mean(axis=0)

    # Debug shapes in caso di mismatch
    try:
        print(f"Debug vals.shape = {getattr(vals, 'shape', None)}")
        print(f"Debug mean_abs.shape = {getattr(mean_abs, 'shape', None)}, X.columns len = {len(X.columns)}")
    except Exception:
        pass
    feat_imp = pd.DataFrame({'feature': X.columns, 'mean_abs_shap': mean_abs})
    feat_imp = feat_imp.sort_values('mean_abs_shap', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feat_imp['feature'], feat_imp['mean_abs_shap'], color='#2b8cbe')
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Feature importance (mean |SHAP|) - Early Stopping')
    plt.tight_layout()
    fig.savefig('shap_early_stopping_bar.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return explainer, shap_values

# 2. SHAP per Efficiency Trade-off
def explain_efficiency_tradeoff(df):
    """Spiega cosa determina alta efficienza (performance/emissions)"""
    # Filtra classificazione
    df_class = df[df['is_regression'] == 0].copy()
    
    if len(df_class) < 10:
        print("⚠️  Troppo pochi dati classificazione per analisi efficiency, skip.")
        return None, None, None
    
    features = ['warmup', 'epoch_ratio', 'uses_early']
    # One-hot encoding per model_family
    df_encoded = pd.get_dummies(df_class, columns=['model_family', 'dataset'])
    feature_cols = [c for c in df_encoded.columns if c.startswith(('model_family_', 'dataset_')) or c in features]
    
    # Rimuovi NaN
    df_encoded = df_encoded[feature_cols + ['efficiency']].dropna()
    
    X = df_encoded[feature_cols].astype(float)
    y = df_encoded['efficiency']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Migliora leggibilità
    plt.rcParams.update({'figure.figsize': (12, 7), 'font.size': 12})

    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP: Performance/Emissions Efficiency Drivers\n(Distribuzione dei contributi per feature)")
    plt.tight_layout()
    plt.savefig("shap_efficiency.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Feature importance assoluta (bar plot)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values.values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top Features per Efficiency:")
    print(feature_importance.head(10))
    
    return explainer, shap_values, feature_importance

# 3. SHAP per Single Prediction (Force Plot)
def explain_single_experiment(df, experiment_name):
    """Spiega perché un esperimento specifico ha certe emissioni"""
    df_enc = pd.get_dummies(df, columns=['model_family', 'dataset'])
    features = ['warmup', 'epoch_ratio', 'uses_early'] + \
               [c for c in df_enc.columns if c.startswith(('model_family_', 'dataset_'))]
    
    if experiment_name not in df['experiment'].values:
        print(f"⚠️  Esperimento '{experiment_name}' non trovato, skip.")
        return
    
    # Rimuovi NaN
    df_enc = df_enc[features + ['emissions', 'experiment']].dropna()
    
    X = df_enc[features].astype(float)
    y = df_enc['emissions']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    explainer = shap.Explainer(model)

    # Trova l'esperimento
    idx = df_enc[df_enc['experiment'] == experiment_name].index[0]
    shap_values = explainer(X)

    # Migliora leggibilità
    plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12})

    # Bar plot dei contributi (valori SHAP) per il singolo campione - ordina top contributori
    sample_vals = shap_values.values[idx]
    sample_feats = X.columns
    sample_df = pd.DataFrame({'feature': sample_feats, 'shap_value': sample_vals})
    sample_df['abs'] = np.abs(sample_df['shap_value'])
    sample_df = sample_df.sort_values('abs', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sample_df['feature'], sample_df['shap_value'], color=np.where(sample_df['shap_value']>=0, '#2ca25f','#de2d26'))
    ax.set_xlabel('SHAP value (positivo aumenta emissioni, negativo le riduce)')
    ax.set_title(f'SHAP contributions - {experiment_name} (sorted by |SHAP|)')
    plt.tight_layout()
    fig.savefig(f'shap_force_{experiment_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Waterfall / detailed view (se disponibile)
    try:
        plt.rcParams.update({'figure.figsize': (8, 5), 'font.size': 11})
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.title(f'Waterfall SHAP - {experiment_name}')
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_{experiment_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception:
        # ignore if waterfall not supported in this env
        pass

# Main execution
if __name__ == "__main__":
    df = load_all_experiments()
    df = engineer_features(df)
    
    print("🔍 Analisi SHAP in corso...\n")
    
    # Analisi 1
    print("1️⃣ Early Stopping Decision...")
    result1 = explain_early_stopping(df)
    if result1[0] is None:
        print("   Skipped.\n")
    else:
        print("   ✅ Completata.\n")
    
    # Analisi 2
    print("2️⃣ Efficiency Trade-off...")
    result2 = explain_efficiency_tradeoff(df)
    if result2[0] is None:
        print("   Skipped.\n")
    else:
        print("   ✅ Completata.\n")
    
    # Analisi 3 - Esempio
    print("3️⃣ Single Experiment Explanation...")
    explain_single_experiment(df, 'chemberta2_early')
    print("   ✅ Completata.\n")
    
    print("\n✅ Analisi completata! Controlla i file PNG generati.")