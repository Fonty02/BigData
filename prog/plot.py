import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Lista dei file da plottare (uno per dataset richiesto)
FILES = [
    "experiments_bace.csv",
    "experiments_cep.csv",
    "experiments_lipophilicity.csv",
    "experiments_malaria.csv",
]

def plot_tradeoff(df, metric, out_name, title):
    if df is None or df.empty:
        print(f"Nessun dato per {out_name}, salto il plot.")
        return

    # Preprocessing: separa base_model e variant (split dall'ultimo underscore)
    def split_name(x):
        # Normalizza i nomi dei modelli. Per i modelli GraphMAE vogliamo
        # che la base sia `graphmae` e la variant includa sia il tipo
        # (classic/early) che la dimensione (10/25/50), es. 'classic_10'.
        if not isinstance(x, str) or '_' not in x:
            return x, ''
        x_lower = x.lower()
        if x_lower.startswith('graphmae'):
            parts = x.split('_', 1)
            if len(parts) == 2:
                return parts[0], parts[1]
            return parts[0], ''
        # comportamento preesistente: split dall'ultimo underscore
        parts = x.rsplit('_', 1)
        return parts[0], parts[1]

    df[['base_model', 'variant']] = df['experiment'].apply(lambda x: pd.Series(split_name(x)))

    # Creiamo un gruppo di plotting. Per GraphMAE vogliamo distinguere
    # le curve in base al valore di `warmup` (es. graphmae_10, graphmae_25).
    df['plot_group'] = df['base_model']
    if 'warmup' in df.columns:
        try:
            df['warmup_str'] = df['warmup'].astype(int).astype(str)
        except Exception:
            df['warmup_str'] = df['warmup'].astype(str)
        mask = df['base_model'].str.lower() == 'graphmae'
        df.loc[mask, 'plot_group'] = df.loc[mask, 'base_model'] + '_' + df.loc[mask, 'warmup_str']

    plt.figure(figsize=(10, 8))

    colors = {'early': 'green', 'classic': 'red'}

    unique_models = df['plot_group'].unique()

    for model in unique_models:
        subset = df[df['plot_group'] == model].copy()
        subset = subset.sort_values(by='emissions')

        # disegna linea se ci sono più punti
        if len(subset) >= 2:
            plt.plot(subset['emissions'], subset[metric], color='gray', linestyle=':', zorder=1)
            mid_x = subset['emissions'].mean()
            mid_y = subset[metric].mean()
            plt.text(mid_x, mid_y, model, fontsize=9, ha='center', va='bottom', fontweight='bold')
        else:
            # etichetta vicino al punto singolo
            try:
                plt.text(subset['emissions'].iloc[0], subset[metric].iloc[0], model, fontsize=9)
            except Exception:
                pass

        for _, row in subset.iterrows():
            variant = str(row.get('variant', '') or '').lower()
            # se la variant contiene sottoparti come 'early_10', prendiamo
            # la prima parte ('early') per decidere il colore
            var_key = variant.split('_')[0] if variant else ''
            c = colors.get(var_key, 'blue')
            plt.scatter(row['emissions'], row[metric], color=c, s=100, zorder=2)

    plt.xlabel("Emissioni (kg CO2eq)")
    ylabel = metric.upper()
    plt.ylabel(ylabel)
    plt.title(f"Trade-off {ylabel} vs Emissioni — {title}")
    plt.xscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Emission Early Stopping', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Classic Early Stopping', markerfacecolor='red', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_name)
    plt.close()
    print(f"Plot salvato come {out_name}")


def main():
    cwd = os.path.dirname(__file__) or '.'

    for f in FILES:
        path = os.path.join(cwd, f)
        if not os.path.exists(path):
            print(f"File non trovato: {path} -> salto")
            continue

        df = pd.read_csv(path)

        # Determina la metrica da plottare: se presente 'rse' con valori non NaN usiamo RSE (task regressione), altrimenti se c'è 'auroc' usiamo AUROC
        metric = None
        if 'rse' in df.columns and df['rse'].notna().any():
            metric = 'rse'
        elif 'auroc' in df.columns and df['auroc'].notna().any():
            metric = 'auroc'
        else:
            print(f"Nessuna metrica valida trovata in {f}, colonne: {list(df.columns)} -> salto")
            continue

        # Rinomina colonne se necessario (assicurarsi che ci siano colonne numeriche)
        # Assumiamo che 'emissions' e la metrica siano già presenti e numeriche

        base = os.path.splitext(f)[0]
        out_name = f"tradeoff_{base}.png"
        title = base.replace('experiments_', '').capitalize()
        plot_tradeoff(df, metric, out_name, title)


if __name__ == '__main__':
    main()