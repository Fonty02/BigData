"""
Script to analyze data distribution in MoleculeNet datasets
Generates distribution diagrams for each dataset used in the project
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from collections import Counter

# Configure matplotlib for high-quality graphics
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_bace_dataset():
    """BACE dataset analysis"""
    print("=== BACE Dataset Analysis ===")
    
    # Load the dataset
    bace_path = "data/dataset_finetuning/bace/raw/bace.csv"
    df = pd.read_csv(bace_path)
    
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("BACE Dataset - Target Distribution & SMILES Length", fontsize=16)
    
    # Target distribution (classification)
    class_dist = df['Class'].value_counts()
    axes[0].bar(class_dist.index, class_dist.values, alpha=0.7, edgecolor='black')
    axes[0].set_title("Class Distribution")
    axes[0].set_xlabel("Class (0=Inactive, 1=Active)")
    axes[0].set_ylabel("Count")
    
    # SMILES length distribution
    df['smiles_length'] = df['mol'].str.len()
    axes[1].hist(df['smiles_length'], bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_title("SMILES Length Distribution")
    axes[1].set_xlabel("SMILES Length")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("bace_analysis.png", dpi=300, bbox_inches='tight')
    
    
    return df

def analyze_bbbp_dataset():
    """BBBP dataset analysis"""
    print("\n=== BBBP Dataset Analysis ===")
    
    bbbp_path = "data/dataset_finetuning/bbbp/raw/BBBP.csv"
    df = pd.read_csv(bbbp_path)
    
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("BBBP Dataset - Target Distribution & SMILES Length", fontsize=16)
    
    # Target distribution (classification)
    perm_dist = df['p_np'].value_counts()
    axes[0].bar(perm_dist.index, perm_dist.values, alpha=0.7, edgecolor='black')
    axes[0].set_title("BBB Permeability Distribution")
    axes[0].set_xlabel("Permeability (0=Non-permeable, 1=Permeable)")
    axes[0].set_ylabel("Count")
    
    # SMILES length distribution
    df['smiles_length'] = df['smiles'].str.len()
    axes[1].hist(df['smiles_length'], bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_title("SMILES Length Distribution")
    axes[1].set_xlabel("SMILES Length")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("bbbp_analysis.png", dpi=300, bbox_inches='tight')
    
    
    return df

def analyze_cep_dataset():
    """CEP dataset analysis"""
    print("\n=== CEP Dataset Analysis ===")
    
    cep_path = "data/dataset_finetuning/cep/raw/cep.csv"
    df = pd.read_csv(cep_path)
    
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"PCE range: {df['PCE'].min():.2f} - {df['PCE'].max():.2f}")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("CEP Dataset - PCE vs SMILES Length & SMILES Length Distribution", fontsize=16)
    
    # Target vs SMILES length (regression)
    df['smiles_length'] = df['smiles'].str.len()
    axes[0].scatter(df['smiles_length'], df['PCE'], alpha=0.5, s=10)
    axes[0].set_title("PCE vs SMILES Length")
    axes[0].set_xlabel("SMILES Length")
    axes[0].set_ylabel("Power Conversion Efficiency (%)")
    
    # SMILES length distribution
    axes[1].hist(df['smiles_length'], bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_title("SMILES Length Distribution")
    axes[1].set_xlabel("SMILES Length")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("cep_analysis.png", dpi=300, bbox_inches='tight')
    
    
    return df

def analyze_hiv_dataset():
    """HIV dataset analysis"""
    print("\n=== HIV Dataset Analysis ===")
    
    hiv_path = "data/dataset_finetuning/hiv/raw/HIV.csv"
    df = pd.read_csv(hiv_path)
    
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("HIV Dataset - Target Distribution & SMILES Length", fontsize=16)
    
    # Target distribution (classification)
    hiv_dist = df['HIV_active'].value_counts()
    axes[0].bar(hiv_dist.index, hiv_dist.values, alpha=0.7, edgecolor='black')
    axes[0].set_title("Anti-HIV Activity Distribution")
    axes[0].set_xlabel("Activity (0=Inactive, 1=Active)")
    axes[0].set_ylabel("Count")
    
    # SMILES length distribution
    df['smiles_length'] = df['smiles'].str.len()
    axes[1].hist(df['smiles_length'], bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_title("SMILES Length Distribution")
    axes[1].set_xlabel("SMILES Length")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("hiv_analysis.png", dpi=300, bbox_inches='tight')
    
    
    return df

def analyze_malaria_dataset():
    """Malaria dataset analysis"""
    print("\n=== Malaria Dataset Analysis ===")
    
    malaria_path = "data/dataset_finetuning/malaria/raw/malaria.csv"
    df = pd.read_csv(malaria_path)
    
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Activity range: {df['activity'].min():.2f} - {df['activity'].max():.2f}")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Malaria Dataset - Activity vs SMILES Length & SMILES Length Distribution", fontsize=16)
    
    # Target vs SMILES length (regression)
    df['smiles_length'] = df['smiles'].str.len()
    axes[0].scatter(df['smiles_length'], df['activity'], alpha=0.5, s=10)
    axes[0].set_title("Anti-Malarial Activity vs SMILES Length")
    axes[0].set_xlabel("SMILES Length")
    axes[0].set_ylabel("Activity Value")
    
    # SMILES length distribution
    axes[1].hist(df['smiles_length'], bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_title("SMILES Length Distribution")
    axes[1].set_xlabel("SMILES Length")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("malaria_analysis.png", dpi=300, bbox_inches='tight')
    
    
    return df

def analyze_lipophilicity_dataset():
    """Lipophilicity dataset analysis"""
    print("\n=== Lipophilicity Dataset Analysis ===")
    
    lipo_path = "data/dataset_finetuning/lipophilicity/raw/Lipophilicity.csv"
    df = pd.read_csv(lipo_path)
    
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Lipophilicity range: {df['exp'].min():.2f} - {df['exp'].max():.2f}")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Lipophilicity Dataset - Lipophilicity vs SMILES Length & SMILES Length Distribution", fontsize=16)
    
    # Target vs SMILES length (regression)
    df['smiles_length'] = df['smiles'].str.len()
    axes[0].scatter(df['smiles_length'], df['exp'], alpha=0.5, s=10)
    axes[0].set_title("Lipophilicity vs SMILES Length")
    axes[0].set_xlabel("SMILES Length")
    axes[0].set_ylabel("Experimental Lipophilicity (log D)")
    
    # SMILES length distribution
    axes[1].hist(df['smiles_length'], bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_title("SMILES Length Distribution")
    axes[1].set_xlabel("SMILES Length")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("lipophilicity_analysis.png", dpi=300, bbox_inches='tight')
    
    
    return df

def generate_summary_report(datasets):
    """Generate a summary report of all datasets"""
    print("\n" + "="*50)
    print("DATASET SUMMARY REPORT")
    print("="*50)
    
    summary_data = []
    for name, df in datasets.items():
        summary_data.append({
            'Dataset': name,
            'Rows': len(df),
            'Columns': len(df.columns),
            'Task': get_task_type(name, df)
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Summary chart
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("MoleculeNet Dataset Comparison", fontsize=16)
    
    # Number of samples per dataset
    axes[0].bar(summary_df['Dataset'], summary_df['Rows'])
    axes[0].set_title("Number of Samples per Dataset")
    axes[0].set_xlabel("Dataset")
    axes[0].set_ylabel("Number of Samples")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Task types
    task_counts = summary_df['Task'].value_counts()
    axes[1].pie(task_counts.values, labels=task_counts.index, autopct='%1.0f', startangle=90)
    axes[1].set_title("Task Type Distribution")
    
    plt.tight_layout()
    #plt.savefig("dataset_summary.png", dpi=300, bbox_inches='tight')
    

def get_task_type(name, df):
    """Determine the task type based on the dataset"""
    if name.upper() in ['BACE', 'BBBP', 'HIV']:
        return 'Classification'
    elif name.upper() in ['CEP', 'MALARIA', 'LIPOPHILICITY']:
        return 'Regression'
    else:
        return 'Unknown'

def main():
    """Main function"""
    print("Data Distribution Analysis - MoleculeNet Datasets")
    print("="*60)
    
    # Change working directory
    os.chdir(Path(__file__).parent)
    
    # Analyze all datasets
    datasets = {}
    
    try:
        datasets['BACE'] = analyze_bace_dataset()
    except Exception as e:
        print(f"Error in BACE analysis: {e}")
    
    try:
        datasets['BBBP'] = analyze_bbbp_dataset()
    except Exception as e:
        print(f"Error in BBBP analysis: {e}")
    
    try:
        datasets['CEP'] = analyze_cep_dataset()
    except Exception as e:
        print(f"Error in CEP analysis: {e}")
    
    try:
        datasets['HIV'] = analyze_hiv_dataset()
    except Exception as e:
        print(f"Error in HIV analysis: {e}")
    
    try:
        datasets['MALARIA'] = analyze_malaria_dataset()
    except Exception as e:
        print(f"Error in MALARIA analysis: {e}")
    
    try:
        datasets['LIPOPHILICITY'] = analyze_lipophilicity_dataset()
    except Exception as e:
        print(f"Error in LIPOPHILICITY analysis: {e}")
    
    # Generate summary report
    if datasets:
        generate_summary_report(datasets)
    
    print("\nAnalysis completed! Charts have been saved as PNG files.")

if __name__ == "__main__":
    main()