import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Genera plot se ci sono dati sufficienti
if os.path.exists("experiments.csv"):
    df_plot = pd.read_csv("experiments.csv")
    
    if len(df_plot) >= 1:
        plt.figure(figsize=(10, 8))
        
        # 1. Preprocessing: Separiamo il nome base dalla variante
        # Assumiamo che il formato sia "nome_modello_variante" (es. _classic, _early)
        df_plot['base_model'] = df_plot['experiment'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        df_plot['variant'] = df_plot['experiment'].apply(lambda x: x.split('_')[-1])
        
        # Mappa dei colori e stili
        colors = {'early': 'green', 'classic': 'red'}
        labels_legend = {'early': 'Early Stopping (GES)', 'classic': 'Classic (no ES)'}
        
        # Otteniamo la lista dei modelli unici
        unique_models = df_plot['base_model'].unique()
        
        for model in unique_models:
            # Estraiamo i dati per questo specifico modello
            subset = df_plot[df_plot['base_model'] == model]
            
            # Se abbiamo almeno 2 punti (coppia classic/early), disegniamo la linea di connessione
            if len(subset) >= 2:
                subset = subset.sort_values(by='emissions') # Ordina per disegnare la linea correttamente
                plt.plot(subset["emissions"], subset["auroc"], 
                         color='gray', linestyle=':', zorder=1) # Linea tratteggiata (zorder basso per stare sotto i punti)
                
                # Aggiungiamo il testo (nome del modello) vicino al punto medio o al primo punto
                mid_x = subset["emissions"].mean()
                mid_y = subset["auroc"].mean()
                plt.text(mid_x, mid_y, model, fontsize=9, ha='center', va='bottom', fontweight='bold')
            else:
                # Se è un punto singolo, mettiamo l'etichetta direttamente sul punto
                plt.text(subset["emissions"].iloc[0], subset["auroc"].iloc[0], model, fontsize=9)

            # Disegniamo i punti (Scatter)
            for _, row in subset.iterrows():
                c = colors.get(row['variant'], 'blue') # Verde o Rosso, Blu se la variante non è riconosciuta
                plt.scatter(row["emissions"], row["auroc"], color=c, s=100, zorder=2)

        # Configurazione Assi e Griglia
        plt.xlabel("Emissioni (kg CO2eq)")
        plt.ylabel("AUROC")
        plt.title("Trade-off AUROC vs Emissioni")
        
        # Scala logaritmica sull'asse X come nell'immagine di riferimento
        plt.xscale('log') 
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        # Creazione Legenda personalizzata
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='GES (Early)', markerfacecolor='green', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='No ES (Classic)', markerfacecolor='red', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig("tradeoff_plot_style.png")
        print("Plot salvato come tradeoff_plot_style.png")