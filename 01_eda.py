import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    data_dir = "data"
    output_dir = "eda_results"
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]
    
    for filename in datasets:
        ds_name = filename.replace(".csv", "")
        filepath = os.path.join(data_dir, filename)
        print(f"\n======================================")
        print(f"Processing EDA for {ds_name}...")
        print(f"======================================")
        
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        
        print(f"Shape: {df.shape}")
        
        # 1. Plot entire OT waveform
        plt.figure(figsize=(15, 5))
        plt.plot(df.index, df['OT'], label=f'{ds_name} OT', color='blue', alpha=0.7)
        plt.title(f"{ds_name}: Oil Temperature Over Time (Full Dataset)")
        plt.xlabel("Date")
        plt.ylabel("OT")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{ds_name}_01_OT_Full_Trend.png"))
        plt.close()

        # 2. Plot 1-month subset to see daily periodicity
        subset = df.loc['2016-08-01':'2016-08-31']
        if not subset.empty:
            plt.figure(figsize=(15, 5))
            plt.plot(subset.index, subset['OT'], label='OT', color='orange')
            plt.title(f"{ds_name}: Oil Temperature (August 2016 - 1 Month Subset)")
            plt.xlabel("Date")
            plt.ylabel("OT")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{ds_name}_02_OT_1Month_Subset.png"))
            plt.close()

        # 3. Correlation Heatmap
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f"{ds_name}: Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{ds_name}_03_Correlation_Heatmap.png"))
        plt.close()

        # 4. Scatter plot: Highest correlated feature vs OT
        correlations_with_OT = corr['OT'].drop('OT')
        best_feature = correlations_with_OT.abs().idxmax()
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[best_feature], y=df['OT'], alpha=0.3)
        plt.title(f"{ds_name} Scatter Plot: {best_feature} vs OT")
        plt.xlabel(best_feature)
        plt.ylabel("OT")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{ds_name}_04_Scatter_{best_feature}_vs_OT.png"))
        plt.close()
        
    print(f"\nEDA completed for all datasets. Plots are saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
