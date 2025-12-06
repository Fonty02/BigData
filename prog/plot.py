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
        if isinstance(x, str) and '_' in x:
            parts = x.rsplit('_', 1)
            return parts[0], parts[1]
        return x, ''

    df[['base_model', 'variant']] = df['experiment'].apply(lambda x: pd.Series(split_name(x)))

    plt.figure(figsize=(10, 8))

    colors = {'early': 'green', 'classic': 'red'}

    unique_models = df['base_model'].unique()

    for model in unique_models:
        subset = df[df['base_model'] == model].copy()
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
            c = colors.get(str(row.get('variant', '')).lower(), 'blue')
            plt.scatter(row['emissions'], row[metric], color=c, s=100, zorder=2)

    plt.xlabel("Emissioni (kg CO2eq)")
    ylabel = metric.upper()
    plt.ylabel(ylabel)
    plt.title(f"Trade-off {ylabel} vs Emissioni — {title}")
    plt.xscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='GES (Early)', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='No ES (Classic)', markerfacecolor='red', markersize=10)
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