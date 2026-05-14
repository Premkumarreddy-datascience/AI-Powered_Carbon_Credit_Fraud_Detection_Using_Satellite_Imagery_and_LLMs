import sys
import os
from pathlib import Path
from datetime import datetime 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no display)
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

# Path setup
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.environ['PROJECT_ROOT'] = str(project_root)

from satellite_module.config.settings import (
    TRAINING_DATA_DIR, MODEL_DIR, BIOMASS_MODEL_CONFIG
)


class BiomassModelTrainer:
    """
    Trains, evaluates and saves both XGBoost and Random Forest models
    """
    
    def __init__(self):
        self.training_data_path = TRAINING_DATA_DIR / "training_dataset_final.csv"
        self.results_dir = MODEL_DIR / f"training_{timestamp}"
        self.results_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.feature_cols = None
        self.X = None
        self.y = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        
        print("BIOMASS MODEL TRAINING PIPELINE")
        print("="*70)
        print(f"Results will be saved to: {self.results_dir}")
        print("Graphs will be saved only (not displayed)")
    
    def load_and_prepare_data(self):
        """Load data and prepare features"""
        print("\n" + "-"*50)
        print("STEP 1: Loading & Preparing Data")
        print("-"*50)
        
        df = pd.read_csv(self.training_data_path)
        print(f"Total samples: {len(df)}")
        
        # Get feature columns
        exclude_cols = ['country', 'area_name', 'aoi_category', 'longitude', 
                       'latitude', 'year', 'biomass_mg_ha']
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"Features: {len(self.feature_cols)}")
        
        # Handle missing values
        df = df.dropna(subset=self.feature_cols + ['biomass_mg_ha'])
        
        self.X = df[self.feature_cols].values
        self.y = df['biomass_mg_ha'].values
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=BIOMASS_MODEL_CONFIG["test_size"],
            random_state=42
        )
        
        print(f"\nTraining set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Biomass range: {self.y.min():.1f} - {self.y.max():.1f} Mg/ha")
        print(f"Biomass mean: {self.y.mean():.1f} ± {self.y.std():.1f} Mg/ha")
    
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n" + "-"*50)
        print("Training XGBoost Model")
        print("-"*50)
        
        params = {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 5,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'random_state': 42,
            'verbosity': 0
        }
        
        model = XGBRegressor(**params)
        model.fit(self.X_train, self.y_train, verbose=False)
        
        return model
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "-"*50)
        print("Training Random Forest Model")
        print("-"*50)
        
        params = {
            'n_estimators': 200,
            'max_depth': 8,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        model.fit(self.X_train, self.y_train)
        
        return model
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate all regression metrics"""
        return {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'nrmse': np.sqrt(mean_squared_error(y_true, y_pred)) / (y_true.max() - y_true.min())
        }
    
    def evaluate_model(self, model, name):
        """Evaluate model and calculate all metrics"""
        print(f"\nEvaluating {name}...")
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(self.y_train, y_train_pred)
        test_metrics = self.calculate_metrics(self.y_test, y_test_pred)
        
        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='r2')
        
        # Store results
        results = {
            'train': train_metrics,
            'test': test_metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_test_pred,
            'actual': self.y_test,
            'residuals': self.y_test - y_test_pred
        }
        
        # Print results
        print(f"\n{name} Results:")
        print(f"  {'Metric':<10} {'Train':<10} {'Test':<10}")
        print(f"  {'-'*32}")
        print(f"  {'R²':<10} {train_metrics['r2']:.4f}    {test_metrics['r2']:.4f}")
        print(f"  {'RMSE':<10} {train_metrics['rmse']:.1f}    {test_metrics['rmse']:.1f}")
        print(f"  {'MAE':<10} {train_metrics['mae']:.1f}    {test_metrics['mae']:.1f}")
        print(f"  {'NRMSE':<10} {train_metrics['nrmse']:.3f}    {test_metrics['nrmse']:.3f}")
        print(f"\n  Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def test_multiple_random_states(self, n_tests=10):
        """Test both models with different random states"""
        print("\n" + "="*70)
        print("RANDOM STATE STABILITY TEST")
        print("="*70)
        
        results_xgb = []
        results_rf = []
        
        for i in range(n_tests):
            random_state = i * 10
            print(f"\nTest {i+1}/{n_tests} - Random State: {random_state}")
            
            # Split with different random state
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y,
                test_size=BIOMASS_MODEL_CONFIG["test_size"],
                random_state=random_state
            )
            
            # Train XGBoost
            xgb_model = XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=5,
                reg_alpha=1.0,
                reg_lambda=2.0,
                random_state=random_state,
                verbosity=0
            )
            xgb_model.fit(X_train, y_train)
            y_pred_xgb = xgb_model.predict(X_test)
            metrics_xgb = self.calculate_metrics(y_test, y_pred_xgb)
            
            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            metrics_rf = self.calculate_metrics(y_test, y_pred_rf)
            
            # Store results
            results_xgb.append({
                'random_state': random_state,
                'r2': metrics_xgb['r2'],
                'rmse': metrics_xgb['rmse'],
                'mae': metrics_xgb['mae'],
                'nrmse': metrics_xgb['nrmse']
            })
            
            results_rf.append({
                'random_state': random_state,
                'r2': metrics_rf['r2'],
                'rmse': metrics_rf['rmse'],
                'mae': metrics_rf['mae'],
                'nrmse': metrics_rf['nrmse']
            })
            
            print(f"  XGBoost: R²={metrics_xgb['r2']:.4f} | RMSE={metrics_xgb['rmse']:.2f}")
            print(f"  RF:      R²={metrics_rf['r2']:.4f} | RMSE={metrics_rf['rmse']:.2f}")
        
        # Create DataFrames
        df_xgb = pd.DataFrame(results_xgb)
        df_rf = pd.DataFrame(results_rf)
        
        # Save results
        df_xgb.to_csv(self.results_dir / 'xgboost_random_state_test.csv', index=False)
        df_rf.to_csv(self.results_dir / 'random_forest_random_state_test.csv', index=False)
        
        # Print summaries
        print("\n" + "-"*50)
        print("XGBOOST RANDOM STATE SUMMARY:")
        print("-"*50)
        for metric in ['r2', 'rmse', 'mae']:
            mean_val = df_xgb[metric].mean()
            std_val = df_xgb[metric].std()
            print(f"{metric.upper():<10} | Mean: {mean_val:.4f} | Std: {std_val:.4f}")
        
        print("\n" + "-"*50)
        print("RANDOM FOREST RANDOM STATE SUMMARY:")
        print("-"*50)
        for metric in ['r2', 'rmse', 'mae']:
            mean_val = df_rf[metric].mean()
            std_val = df_rf[metric].std()
            print(f"{metric.upper():<10} | Mean: {mean_val:.4f} | Std: {std_val:.4f}")
        
        # Create visualizations (save only, no display)
        self.plot_random_state_results(df_xgb, 'XGBoost')
        self.plot_random_state_results(df_rf, 'Random Forest')
        self.plot_random_state_comparison(df_xgb, df_rf)
        
        return df_xgb, df_rf
    
    def plot_random_state_results(self, df, model_name):
        """Plot random state stability for a single model (save only)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{model_name} - Stability Across Random States', fontsize=14)
        
        metrics = [
            ('r2', 'R² Score', '#ff7f0e'),
            ('rmse', 'RMSE (Mg/ha)', '#1f77b4'),
            ('mae', 'MAE (Mg/ha)', '#2ca02c'),
            ('nrmse', 'NRMSE', '#9467bd')
        ]
        
        for idx, (metric, ylabel, color) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Line plot
            ax.plot(df['random_state'], df[metric], 'o-', color=color, linewidth=2, markersize=8)
            
            # Mean line
            mean_val = df[metric].mean()
            ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, 
                      label=f'Mean: {mean_val:.3f}')
            
            # Std band
            std_val = df[metric].std()
            ax.fill_between(df['random_state'], 
                           mean_val - std_val, mean_val + std_val,
                           alpha=0.2, color='gray', label=f'±1 Std: {std_val:.3f}')
            
            ax.set_xlabel('Random State')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{metric.upper()} Stability')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = model_name.lower().replace(' ', '_') + '_random_state_stability.png'
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        print(f"  Saved: {filename}")
    
    def plot_random_state_comparison(self, df_xgb, df_rf):
        """Compare random state stability between models (save only)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Stability Comparison Across Random States', fontsize=14)
        
        metrics = [
            ('r2', 'R² Score'),
            ('rmse', 'RMSE (Mg/ha)'),
            ('mae', 'MAE (Mg/ha)'),
            ('nrmse', 'NRMSE')
        ]
        
        for idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Box plots for comparison
            data = [df_xgb[metric], df_rf[metric]]
            bp = ax.boxplot(data, patch_artist=True, labels=['XGBoost', 'Random Forest'])
            
            # Colors
            bp['boxes'][0].set_facecolor('#ff7f0e')
            bp['boxes'][1].set_facecolor('#1f77b4')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_alpha(0.7)
            
            ax.set_ylabel(ylabel)
            ax.set_title(f'{metric.upper()} Distribution')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add mean values
            means = [df_xgb[metric].mean(), df_rf[metric].mean()]
            for i, mean_val in enumerate(means):
                ax.text(i+1, mean_val, f'Mean: {mean_val:.3f}', 
                       ha='center', va='bottom', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'random_state_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: random_state_comparison.png")
    
    def plot_predictions(self, name):
        """Plot predicted vs actual for a single model (save only)"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{name} - Model Performance Analysis', fontsize=14)
        
        results = self.results[name]
        y_test = results['actual']
        y_pred = results['predictions']
        residuals = results['residuals']
        
        # 1. Predicted vs Actual
        ax = axes[0]
        ax.scatter(y_test, y_pred, alpha=0.5, c='blue', edgecolors='black', linewidth=0.5, s=30)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7, label='1:1 Line')
        ax.set_xlabel('Actual Biomass (Mg/ha)')
        ax.set_ylabel('Predicted Biomass (Mg/ha)')
        ax.set_title(f'Predicted vs Actual\nR² = {results["test"]["r2"]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Residuals vs Predicted
        ax = axes[1]
        ax.scatter(y_pred, residuals, alpha=0.5, c='green', edgecolors='black', linewidth=0.5, s=30)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Predicted Biomass (Mg/ha)')
        ax.set_ylabel('Residuals (Mg/ha)')
        ax.set_title(f'Residuals Plot\nMean = {residuals.mean():.2f}, Std = {residuals.std():.2f}')
        ax.grid(True, alpha=0.3)
        
        # 3. Residual Distribution
        ax = axes[2]
        ax.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, linewidth=2)
        ax.set_xlabel('Residuals (Mg/ha)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Residual Distribution\nRMSE = {results["test"]["rmse"]:.1f}, MAE = {results["test"]["mae"]:.1f}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = name.lower().replace(' ', '_') + '_analysis.png'
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {filename}")
    
    def plot_feature_importance(self, model, name):
        """Plot feature importance for a single model (save only)"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:15]
        top_features = [self.feature_cols[i] for i in indices]
        top_importance = importance[indices]
        
        ax.barh(range(len(top_features)), top_importance[::-1], color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1])
        ax.set_xlabel('Importance')
        ax.set_title(f'{name} - Top 15 Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(top_importance[::-1]):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        filename = name.lower().replace(' ', '_') + '_feature_importance.png'
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {filename}")
    
    def plot_comparison_summary(self):
        """Plot comparison of both models (save only)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('XGBoost vs Random Forest - Performance Comparison', fontsize=16)
        
        models = list(self.results.keys())
        x = np.arange(len(models))
        width = 0.4
        
        # R² Comparison
        ax = axes[0, 0]
        test_r2 = [self.results[m]['test']['r2'] for m in models]
        cv_r2 = [self.results[m]['cv_mean'] for m in models]
        
        ax.bar(x - width/2, test_r2, width, label='Test R²', color='#ff7f0e', alpha=0.8)
        ax.bar(x + width/2, cv_r2, width, label='CV R²', color='#1f77b4', alpha=0.8)
        ax.set_ylabel('R² Score')
        ax.set_title('R² Comparison (higher is better)')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        for i, v in enumerate(test_r2):
            ax.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
        for i, v in enumerate(cv_r2):
            ax.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
        
        # RMSE Comparison
        ax = axes[0, 1]
        test_rmse = [self.results[m]['test']['rmse'] for m in models]
        
        ax.bar(x, test_rmse, width, color=['#ff7f0e', '#1f77b4'], alpha=0.8)
        ax.set_ylabel('RMSE (Mg/ha)')
        ax.set_title('RMSE Comparison (lower is better)')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.grid(True, alpha=0.3)
        
        for i, v in enumerate(test_rmse):
            ax.text(i, v + 2, f'{v:.1f}', ha='center', fontsize=9)
        
        # MAE Comparison
        ax = axes[1, 0]
        test_mae = [self.results[m]['test']['mae'] for m in models]
        
        ax.bar(x, test_mae, width, color=['#ff7f0e', '#1f77b4'], alpha=0.8)
        ax.set_ylabel('MAE (Mg/ha)')
        ax.set_title('MAE Comparison (lower is better)')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.grid(True, alpha=0.3)
        
        for i, v in enumerate(test_mae):
            ax.text(i, v + 2, f'{v:.1f}', ha='center', fontsize=9)
        
        # Overfitting Comparison
        ax = axes[1, 1]
        overfitting = [self.results[m]['train']['r2'] - self.results[m]['test']['r2'] for m in models]
        
        ax.bar(x, overfitting, width, color=['#ff7f0e', '#1f77b4'], alpha=0.8)
        ax.set_ylabel('R² Gap (Train - Test)')
        ax.set_title('Overfitting Score (lower is better)')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.grid(True, alpha=0.3)
        
        for i, v in enumerate(overfitting):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'models_comparison_summary.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: models_comparison_summary.png")
    
    def save_model(self, model, name):
        """Save trained model with metadata"""
        model_path = self.results_dir / f'{name.lower().replace(" ", "_")}_model_{timestamp}.pkl'
        
        joblib.dump({
            'model': model,
            'name': name,
            'feature_cols': self.feature_cols,
            'metrics': self.results[name],
            'timestamp': timestamp
        }, model_path)
        
        print(f"\n{name} model saved to: {model_path}")
        return model_path
    
    def save_results_csv(self):
        """Save all results to CSV and create a summary text file"""
        rows = []
        for name in self.results:
            rows.append({
                'Model': name,
                'Train R²': self.results[name]['train']['r2'],
                'Test R²': self.results[name]['test']['r2'],
                'Train RMSE': self.results[name]['train']['rmse'],
                'Test RMSE': self.results[name]['test']['rmse'],
                'Train MAE': self.results[name]['train']['mae'],
                'Test MAE': self.results[name]['test']['mae'],
                'Train NRMSE': self.results[name]['train']['nrmse'],
                'Test NRMSE': self.results[name]['test']['nrmse'],
                'CV R² Mean': self.results[name]['cv_mean'],
                'CV R² Std': self.results[name]['cv_std'],
                'Overfitting Gap': self.results[name]['train']['r2'] - self.results[name]['test']['r2']
            })
        
        df = pd.DataFrame(rows)
        csv_path = self.results_dir / 'model_results_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Calculate composite scores
        composite_scores = {}
        max_rmse = max(df['Test RMSE'])
        max_mae = max(df['Test MAE'])
        
        for name in self.results:
            composite = (
                0.3 * self.results[name]['test']['r2'] +
                0.3 * (1 - self.results[name]['test']['rmse'] / max_rmse) +
                0.2 * (1 - self.results[name]['test']['mae'] / max_mae) +
                0.2 * self.results[name]['cv_mean']
            )
            composite_scores[name] = composite
        
        # Determine winners
        best_r2 = df.loc[df['Test R²'].idxmax(), 'Model']
        best_cv = df.loc[df['CV R² Mean'].idxmax(), 'Model']
        best_gen = df.loc[df['Overfitting Gap'].idxmin(), 'Model']
        best_composite = max(composite_scores, key=composite_scores.get)
        
        # Create summary text file
        summary_path = self.results_dir / 'model_selection_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BIOMASS MODEL TRAINING - FINAL RESULTS\n")
            f.write("="*70 + "\n\n")
            
            f.write("DATASET SUMMARY:\n")
            f.write("-"*50 + "\n")
            f.write(f"Total samples: {len(self.y)}\n")
            f.write(f"Training samples: {len(self.y_train)}\n")
            f.write(f"Test samples: {len(self.y_test)}\n")
            f.write(f"Features: {len(self.feature_cols)}\n")
            f.write(f"Biomass range: {self.y.min():.1f} - {self.y.max():.1f} Mg/ha\n")
            f.write(f"Biomass mean: {self.y.mean():.1f} ± {self.y.std():.1f} Mg/ha\n\n")
            
            f.write("MODEL PERFORMANCE COMPARISON:\n")
            f.write("-"*50 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            f.write("COMPOSITE SCORES (higher is better):\n")
            f.write("-"*50 + "\n")
            for name, score in composite_scores.items():
                f.write(f"{name}: {score:.4f}\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("BEST MODEL SELECTION RESULTS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Best Model by Test R²: {best_r2}\n")
            f.write(f"Best Model by CV R²: {best_cv}\n")
            f.write(f"Best Generalization (lowest overfitting): {best_gen}\n")
            f.write(f"BEST OVERALL MODEL (Composite Score): {best_composite}\n\n")
            
            f.write("WHY THIS MODEL WINS:\n")
            f.write("-"*50 + "\n")
            
            # Detailed explanation
            winner_metrics = self.results[best_composite]
            other_model = [m for m in self.results.keys() if m != best_composite][0]
            other_metrics = self.results[other_model]
            
            f.write(f"• Test R²: {winner_metrics['test']['r2']:.4f} vs {other_metrics['test']['r2']:.4f}\n")
            f.write(f"• Test RMSE: {winner_metrics['test']['rmse']:.2f} vs {other_metrics['test']['rmse']:.2f} Mg/ha\n")
            f.write(f"• Test MAE: {winner_metrics['test']['mae']:.2f} vs {other_metrics['test']['mae']:.2f} Mg/ha\n")
            f.write(f"• CV R²: {winner_metrics['cv_mean']:.4f} vs {other_metrics['cv_mean']:.4f}\n")
            f.write(f"• Overfitting Gap: {winner_metrics['train']['r2'] - winner_metrics['test']['r2']:.4f} vs {other_metrics['train']['r2'] - other_metrics['test']['r2']:.4f}\n\n")
            
            if best_composite == best_gen:
                f.write("This model generalizes much better (lower overfitting)\n")
            if best_composite == best_r2:
                f.write("This model has higher predictive power (better R²)\n")
            if best_composite == best_cv:
                f.write("This model is more stable across cross-validation folds\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"\nDetailed summary saved to: {summary_path}")
        
        # Print to console as well
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)
        print(df.to_string(index=False))
        
        print("\n" + "-"*50)
        print("BEST MODEL SELECTION:")
        print("-"*50)
        for name, score in composite_scores.items():
            print(f"{name} Composite Score: {score:.4f}")
        
        print(f"\nBest Model by Test R²: {best_r2}")
        print(f"Best Model by CV R²: {best_cv}")
        print(f"Best Generalization (lowest overfitting): {best_gen}")
        print(f"\nRECOMMENDED MODEL FOR PRODUCTION: {best_composite}")
        print(f"   (Based on weighted composite score)")
    
    def run(self):
        """Run complete training pipeline for both models"""
        
        # Step 1: Load data
        self.load_and_prepare_data()
        
        # Step 2: Train and evaluate XGBoost
        print("\n" + "="*70)
        print("PART 1: XGBOOST MODEL")
        print("="*70)
        self.models['XGBoost'] = self.train_xgboost()
        self.results['XGBoost'] = self.evaluate_model(self.models['XGBoost'], 'XGBoost')
        self.plot_predictions('XGBoost')
        self.plot_feature_importance(self.models['XGBoost'], 'XGBoost')
        self.save_model(self.models['XGBoost'], 'XGBoost')
        
        # Step 3: Train and evaluate Random Forest
        print("\n" + "="*70)
        print("PART 2: RANDOM FOREST MODEL")
        print("="*70)
        self.models['Random Forest'] = self.train_random_forest()
        self.results['Random Forest'] = self.evaluate_model(self.models['Random Forest'], 'Random Forest')
        self.plot_predictions('Random Forest')
        self.plot_feature_importance(self.models['Random Forest'], 'Random Forest')
        self.save_model(self.models['Random Forest'], 'Random Forest')
        
        # Step 4: Random state stability tests
        print("\n" + "="*70)
        print("PART 3: RANDOM STATE STABILITY TESTS")
        print("="*70)
        df_xgb, df_rf = self.test_multiple_random_states(n_tests=10)
        
        # Step 5: Compare and save results
        print("\n" + "="*70)
        print("PART 4: MODEL COMPARISON")
        print("="*70)
        self.plot_comparison_summary()
        self.save_results_csv()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"All results saved to: {self.results_dir}")
        print("\nGenerated files:")
        for file in self.results_dir.glob('*'):
            print(f"  - {file.name}")


if __name__ == "__main__":
    trainer = BiomassModelTrainer()
    trainer.run()