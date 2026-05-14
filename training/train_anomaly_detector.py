import sys
import os
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Path setup
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.environ['PROJECT_ROOT'] = str(project_root)

from satellite_module.config.settings import (
    TRAINING_DATA_DIR, MODEL_DIR, ANOMALY_CONFIG
)


class AnomalyDetectorTrainer:
    """
    Trains unsupervised anomaly detection model
    """
    
    def __init__(self):
        self.training_data_path = TRAINING_DATA_DIR / "training_dataset_final.csv"
        self.results_dir = MODEL_DIR / f"anomaly_{timestamp}"
        self.results_dir.mkdir(exist_ok=True)
        self.model_path = self.results_dir / f"anomaly_detector_{timestamp}.pkl"
        
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.X_scaled = None
        
        print("="*70)
        print("ANOMALY DETECTOR TRAINING PIPELINE")
        print("="*70)
        print(f"Training data: {self.training_data_path}")
        print(f"Results will be saved to: {self.results_dir}")
        print(f"Model will be saved to: {self.model_path}")
    
    def load_and_prepare_data(self):
        """
        Load data and prepare features
        """
        print("\n" + "-"*50)
        print("STEP 1: Loading & Preparing Data")
        print("-"*50)
        
        # Load data
        df = pd.read_csv(self.training_data_path)
        print(f"Loaded {len(df)} samples")
        
        # Get feature columns (all s2_* columns)
        self.feature_cols = [col for col in df.columns if col.startswith('s2_')]
        print(f"Features: {len(self.feature_cols)}")
        
        # Prepare features
        X = df[self.feature_cols].values
        print(f"Feature matrix shape: {X.shape}")
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        print(f"Data standardized: mean≈0, std≈1")
        
        return self.X_scaled
    
    def train_isolation_forest(self, X):
        """
        Train Isolation Forest model
        """
        print("\n" + "-"*50)
        print("STEP 2: Training Isolation Forest")
        print("-"*50)
        
        # Get parameters
        params = ANOMALY_CONFIG["isolation_forest_params"].copy()
        
        print("Training with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Initialize and train
        model = IsolationForest(**params)
        model.fit(X)
        
        # Predict on training data
        y_pred = model.predict(X)
        scores = model.score_samples(X)
        
        # Statistics
        n_anomalies = np.sum(y_pred == -1)
        n_normal = np.sum(y_pred == 1)
        anomaly_rate = n_anomalies / len(X) * 100
        
        print(f"\nResults:")
        print(f"  Normal samples: {n_normal} ({100-anomaly_rate:.1f}%)")
        print(f"  Anomalies detected: {n_anomalies} ({anomaly_rate:.1f}%)")
        print(f"  Score range: {scores.min():.4f} to {scores.max():.4f}")
        print(f"  Threshold: {model.offset_:.4f}")
        
        self.model = model
        return model
    
    def test_multiple_random_states(self, X, n_tests=10):
        """Test anomaly detector stability with different random states"""
        print("\n" + "-"*50)
        print("STEP 3: Testing Multiple Random States")
        print("-"*50)
        
        results = []
        for i in range(n_tests):
            random_state = i * 10
            params = ANOMALY_CONFIG["isolation_forest_params"].copy()
            params['random_state'] = random_state
            
            model = IsolationForest(**params)
            model.fit(X)
            
            y_pred = model.predict(X)
            n_anomalies = np.sum(y_pred == -1)
            anomaly_rate = n_anomalies / len(X) * 100
            
            results.append({
                'random_state': random_state,
                'anomaly_rate': anomaly_rate,
                'n_anomalies': n_anomalies
            })
            
            print(f"  RS={random_state:3d} | Anomalies: {n_anomalies:4d} ({anomaly_rate:.2f}%)")
        
        df_results = pd.DataFrame(results)
        
        # Save results
        csv_path = self.results_dir / 'random_state_test.csv'
        df_results.to_csv(csv_path, index=False)
        
        print(f"\nSummary:")
        print(f"  Mean anomaly rate: {df_results['anomaly_rate'].mean():.2f}%")
        print(f"  Std anomaly rate: {df_results['anomaly_rate'].std():.2f}%")
        print(f"  Min anomaly rate: {df_results['anomaly_rate'].min():.2f}%")
        print(f"  Max anomaly rate: {df_results['anomaly_rate'].max():.2f}%")
        
        # Plot stability
        plt.figure(figsize=(10, 5))
        plt.plot(df_results['random_state'], df_results['anomaly_rate'], 'o-', color='red', linewidth=2, markersize=8)
        plt.axhline(y=df_results['anomaly_rate'].mean(), color='blue', linestyle='--', 
                   label=f"Mean: {df_results['anomaly_rate'].mean():.2f}%")
        plt.fill_between(df_results['random_state'],
                         df_results['anomaly_rate'].mean() - df_results['anomaly_rate'].std(),
                         df_results['anomaly_rate'].mean() + df_results['anomaly_rate'].std(),
                         alpha=0.2, color='gray', label=f"±1 Std: {df_results['anomaly_rate'].std():.2f}%")
        plt.xlabel('Random State')
        plt.ylabel('Anomaly Rate (%)')
        plt.title('Anomaly Detector Stability Across Random States')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = self.results_dir / 'random_state_stability.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Stability plot saved to: {plot_path}")
        
        return df_results
    
    def analyze_feature_contribution(self, X):
        """Analyze which features contribute most to anomalies"""
        print("\n" + "-"*50)
        print("STEP 4: Feature Contribution Analysis")
        print("-"*50)
        
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Calculate mean values for normal vs anomalous
        normal_mean = X[y_pred == 1].mean(axis=0)
        anomaly_mean = X[y_pred == -1].mean(axis=0)
        
        # Calculate difference
        diff = anomaly_mean - normal_mean
        
        # Get top features with biggest absolute difference
        top_idx = np.argsort(np.abs(diff))[::-1][:15]
        
        print("\nTop features distinguishing anomalies:")
        print("-"*40)
        for i, idx in enumerate(top_idx[:10]):
            direction = "higher" if diff[idx] > 0 else "lower"
            print(f"  {i+1}. {self.feature_cols[idx]}: {diff[idx]:+.3f} ({direction} in anomalies)")
        
        # Plot
        plt.figure(figsize=(12, 8))
        colors = ['red' if x > 0 else 'blue' for x in diff[top_idx]]
        plt.barh(range(len(top_idx)), diff[top_idx][::-1], color=colors[::-1], alpha=0.7)
        plt.yticks(range(len(top_idx)), [self.feature_cols[i] for i in top_idx][::-1])
        plt.xlabel('Mean Difference (Anomaly - Normal)')
        plt.title('Feature Contributions to Anomaly Detection')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        contrib_path = self.results_dir / 'feature_contribution.png'
        plt.savefig(contrib_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature contribution plot saved to: {contrib_path}")
    
    def visualize_anomalies(self, X):
        """
        Visualize anomalies using PCA
        """
        print("\n" + "-"*50)
        print("STEP 5: Visualizing Anomalies")
        print("-"*50)
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Predict anomalies
        y_pred = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: PCA scatter
        ax = axes[0]
        ax.scatter(X_pca[y_pred == 1, 0], X_pca[y_pred == 1, 1], 
                   c='blue', alpha=0.3, s=20, label='Normal')
        ax.scatter(X_pca[y_pred == -1, 0], X_pca[y_pred == -1, 1], 
                   c='red', alpha=0.8, s=40, label='Anomaly', edgecolors='black', linewidth=0.5)
        ax.set_title(f"Anomaly Detection Results (PCA)\n{np.sum(y_pred==-1)} anomalies detected ({np.sum(y_pred==-1)/len(X)*100:.2f}%)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Score distribution
        ax = axes[1]
        ax.hist(scores[y_pred == 1], bins=30, alpha=0.5, label='Normal', color='blue', edgecolor='black')
        ax.hist(scores[y_pred == -1], bins=30, alpha=0.7, label='Anomaly', color='red', edgecolor='black')
        ax.axvline(x=self.model.offset_, color='black', linestyle='--', linewidth=2, label='Threshold')
        ax.set_title("Anomaly Score Distribution")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        viz_path = self.results_dir / 'anomaly_visualization.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {viz_path}")
        
        # Also save explained variance
        explained_var = pca.explained_variance_ratio_
        print(f"PCA explained variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}, Total={explained_var.sum():.2%}")
    
    def save_model(self):
        """
        Save model and scaler
        """
        print("\n" + "-"*50)
        print("STEP 6: Saving Model")
        print("-"*50)
        
        # Save model (using self.model_path defined in __init__)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'config': ANOMALY_CONFIG["isolation_forest_params"],
            'timestamp': timestamp
        }, self.model_path)
        
        print(f"Model saved to: {self.model_path}")
    
    def save_summary(self, random_state_results):
        """Save summary text file"""
        summary_path = self.results_dir / 'anomaly_detector_summary.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ANOMALY DETECTOR TRAINING SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("DATASET SUMMARY:\n")
            f.write("-"*50 + "\n")
            f.write(f"Total samples: {len(self.X_scaled)}\n")
            f.write(f"Features: {len(self.feature_cols)}\n\n")
            
            f.write("MODEL PARAMETERS:\n")
            f.write("-"*50 + "\n")
            for key, value in ANOMALY_CONFIG["isolation_forest_params"].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Predict on full data
            y_pred = self.model.predict(self.X_scaled)
            n_anomalies = np.sum(y_pred == -1)
            anomaly_rate = n_anomalies / len(self.X_scaled) * 100
            
            f.write("TRAINING RESULTS:\n")
            f.write("-"*50 + "\n")
            f.write(f"Normal samples: {len(self.X_scaled) - n_anomalies} ({100-anomaly_rate:.2f}%)\n")
            f.write(f"Anomalies detected: {n_anomalies} ({anomaly_rate:.2f}%)\n")
            f.write(f"Threshold: {self.model.offset_:.4f}\n\n")
            
            f.write("RANDOM STATE TEST RESULTS:\n")
            f.write("-"*50 + "\n")
            f.write(f"Mean anomaly rate: {random_state_results['anomaly_rate'].mean():.2f}%\n")
            f.write(f"Std anomaly rate: {random_state_results['anomaly_rate'].std():.2f}%\n")
            f.write(f"Min anomaly rate: {random_state_results['anomaly_rate'].min():.2f}%\n")
            f.write(f"Max anomaly rate: {random_state_results['anomaly_rate'].max():.2f}%\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"Summary saved to: {summary_path}")
    
    def run_pipeline(self):
        """
        Run complete training pipeline
        """
        print("\n" + "="*70)
        print("STARTING ANOMALY DETECTOR TRAINING")
        print("="*70)
        
        # Step 1: Load and prepare data
        X = self.load_and_prepare_data()
        
        # Step 2: Train model
        self.train_isolation_forest(X)
        
        # Step 3: Test multiple random states
        random_state_results = self.test_multiple_random_states(X, n_tests=10)
        
        # Step 4: Feature contribution analysis
        self.analyze_feature_contribution(X)
        
        # Step 5: Visualize
        self.visualize_anomalies(X)
        
        # Step 6: Save model
        self.save_model()
        
        # Step 7: Save summary
        self.save_summary(random_state_results)
        
        print("\n" + "="*70)
        print("ANOMALY DETECTOR TRAINING COMPLETE!")
        print("="*70)
        print(f"All results saved to: {self.results_dir}")
        
        # List generated files
        print("\nGenerated files:")
        for file in sorted(self.results_dir.glob('*')):
            print(f"  - {file.name}")
        print(f"  - {self.model_path.name}")


if __name__ == "__main__":
    trainer = AnomalyDetectorTrainer()
    trainer.run_pipeline()