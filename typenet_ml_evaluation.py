import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class TypeNetMLEvaluator:
    """
    Evaluation system for TypeNet ML experiments
    Implements similarity search and top-k accuracy metrics
    """
    
    def __init__(self, experiment_dir: str = 'ml_experiments'):
        self.experiment_dir = Path(experiment_dir)
        self.results = []
        
    def load_experiment_data(self, imputation: str, experiment_name: str, 
                           dataset_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data for a specific experiment"""
        exp_path = self.experiment_dir / f'imputation_{imputation}' / experiment_name
        
        train_path = exp_path / f'{dataset_type}_train.csv'
        test_path = exp_path / f'{dataset_type}_test.csv'
        
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Dataset files not found for {experiment_name}/{dataset_type}")
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        return train_data, test_data
    
    def prepare_features(self, data: pd.DataFrame, dataset_type: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Prepare features and labels from dataset
        Returns: features, labels, and indices for tracking
        """
        # Identify ID columns based on dataset type
        if dataset_type == 'dataset_1':
            id_cols = ['user_id', 'platform_id']
        elif dataset_type == 'dataset_2':
            id_cols = ['user_id', 'platform_id', 'session_id']
        elif dataset_type == 'dataset_3':
            id_cols = ['user_id', 'platform_id', 'session_id', 'video_id']
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Extract features (all non-ID columns)
        feature_cols = [col for col in data.columns if col not in id_cols]
        X = data[feature_cols].values
        
        # Extract labels (user_id)
        y = data['user_id'].values
        
        # Create indices for tracking
        indices = list(range(len(data)))
        
        return X, y, indices
    
    def compute_similarity_matrix(self, X_train: np.ndarray, X_test: np.ndarray, 
                                 method: str = 'cosine') -> np.ndarray:
        """
        Compute similarity matrix between test and train samples
        Returns: similarity matrix of shape (n_test, n_train)
        """
        if method == 'cosine':
            # Cosine similarity (higher is more similar)
            similarity = cosine_similarity(X_test, X_train)
        elif method == 'euclidean':
            # Convert Euclidean distance to similarity (lower distance = higher similarity)
            distances = euclidean_distances(X_test, X_train)
            # Convert to similarity: 1 / (1 + distance)
            similarity = 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        return similarity
    
    def calculate_top_k_accuracy(self, similarity_matrix: np.ndarray, 
                               y_test: np.ndarray, y_train: np.ndarray,
                               k_values: List[int] = [1, 3, 5]) -> Dict[int, float]:
        """
        Calculate top-k accuracy for given k values
        
        For each test sample:
        1. Find k most similar training samples
        2. Check if true user_id appears in top-k predictions
        3. Count as hit if ANY of the top-k matches (no extra credit for multiple matches)
        """
        n_test = len(y_test)
        accuracies = {k: 0 for k in k_values}
        
        for i in range(n_test):
            # Get similarity scores for this test sample
            sim_scores = similarity_matrix[i]
            
            # Get indices of top-k most similar training samples
            top_k_indices = np.argsort(sim_scores)[::-1]  # Sort descending
            
            # Check for each k value
            true_user = y_test[i]
            
            for k in k_values:
                # Get top-k predicted users
                top_k_users = y_train[top_k_indices[:k]]
                
                # Check if true user appears in top-k (at least once)
                if true_user in top_k_users:
                    accuracies[k] += 1
        
        # Convert counts to percentages
        for k in k_values:
            accuracies[k] = (accuracies[k] / n_test) * 100
        
        return accuracies
    
    def evaluate_experiment(self, imputation: str, experiment_name: str, 
                          dataset_type: str, similarity_method: str = 'cosine',
                          normalize: bool = True) -> Dict:
        """
        Evaluate a single experiment configuration
        """
        print(f"\nEvaluating: {imputation}/{experiment_name}/{dataset_type}")
        
        # Load data
        train_data, test_data = self.load_experiment_data(imputation, experiment_name, dataset_type)
        
        # Prepare features
        X_train, y_train, _ = self.prepare_features(train_data, dataset_type)
        X_test, y_test, _ = self.prepare_features(test_data, dataset_type)
        
        print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"  Unique users - Train: {len(np.unique(y_train))}, Test: {len(np.unique(y_test))}")
        
        # Normalize features if requested
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(X_train, X_test, similarity_method)
        
        # Calculate top-k accuracies
        k_values = [1, 3, 5]
        accuracies = self.calculate_top_k_accuracy(similarity_matrix, y_test, y_train, k_values)
        
        # Store results
        result = {
            'imputation': imputation,
            'experiment': experiment_name,
            'dataset': dataset_type,
            'similarity_method': similarity_method,
            'normalized': normalize,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'n_users_train': len(np.unique(y_train)),
            'n_users_test': len(np.unique(y_test)),
            **{f'top_{k}_accuracy': acc for k, acc in accuracies.items()}
        }
        
        print(f"  Results: Top-1: {accuracies[1]:.2f}%, Top-3: {accuracies[3]:.2f}%, Top-5: {accuracies[5]:.2f}%")
        
        return result
    
    def run_all_experiments(self, similarity_method: str = 'cosine', normalize: bool = True):
        """
        Run evaluation for all experiment configurations
        """
        print("Running all TypeNet ML experiments...")
        print(f"Similarity method: {similarity_method}, Normalization: {normalize}")
        
        # Get all experiment directories
        for imputation in ['global', 'user']:
            imp_dir = self.experiment_dir / f'imputation_{imputation}'
            
            if not imp_dir.exists():
                print(f"Skipping {imputation} - directory not found")
                continue
            
            # Get all experiment subdirectories
            exp_dirs = [d for d in imp_dir.iterdir() if d.is_dir() and d.name != '__pycache__']
            
            for exp_dir in exp_dirs:
                experiment_name = exp_dir.name
                
                # Try all dataset types
                for dataset_type in ['dataset_1', 'dataset_2', 'dataset_3']:
                    try:
                        result = self.evaluate_experiment(
                            imputation, experiment_name, dataset_type, 
                            similarity_method, normalize
                        )
                        self.results.append(result)
                    except Exception as e:
                        print(f"  Error: {e}")
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(self.results)
        
        # Save results
        output_file = self.experiment_dir / 'evaluation_results.csv'
        self.results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        return self.results_df
    
    def plot_results_summary(self, output_file: str = 'ml_results_summary.png'):
        """Create comprehensive visualization of results"""
        if self.results_df is None or len(self.results_df) == 0:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Top-k accuracy by dataset type
        ax = axes[0, 0]
        dataset_summary = self.results_df.groupby('dataset')[['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy']].mean()
        dataset_summary.plot(kind='bar', ax=ax)
        ax.set_title('Average Top-k Accuracy by Dataset Type')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Dataset Type')
        ax.legend(['Top-1', 'Top-3', 'Top-5'])
        
        # 2. Imputation strategy comparison
        ax = axes[0, 1]
        imp_summary = self.results_df.groupby('imputation')[['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy']].mean()
        imp_summary.plot(kind='bar', ax=ax)
        ax.set_title('Average Top-k Accuracy by Imputation Strategy')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Imputation Strategy')
        
        # 3. Session experiment results
        ax = axes[0, 2]
        session_results = self.results_df[self.results_df['experiment'].str.contains('session')]
        if len(session_results) > 0:
            session_summary = session_results.groupby('dataset')[['top_1_accuracy', 'top_5_accuracy']].mean()
            session_summary.plot(kind='bar', ax=ax)
            ax.set_title('Session Experiments: Top-1 and Top-5 Accuracy')
            ax.set_ylabel('Accuracy (%)')
        
        # 4. Platform 3c2 experiments
        ax = axes[1, 0]
        platform_3c2 = self.results_df[self.results_df['experiment'].str.contains('3c2')]
        if len(platform_3c2) > 0:
            # Extract test platform from experiment name
            platform_3c2['test_platform'] = platform_3c2['experiment'].str.extract(r'test(\d)')
            platform_summary = platform_3c2.groupby('test_platform')['top_5_accuracy'].mean()
            platform_summary.plot(kind='bar', ax=ax)
            ax.set_title('Platform 3-choose-2: Avg Top-5 Accuracy by Test Platform')
            ax.set_ylabel('Top-5 Accuracy (%)')
            ax.set_xlabel('Test Platform')
        
        # 5. Best performing configurations
        ax = axes[1, 1]
        top_configs = self.results_df.nlargest(10, 'top_5_accuracy')[['experiment', 'dataset', 'top_5_accuracy']]
        y_pos = np.arange(len(top_configs))
        ax.barh(y_pos, top_configs['top_5_accuracy'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['experiment'][:20]}.../{row['dataset']}" for _, row in top_configs.iterrows()], fontsize=8)
        ax.set_xlabel('Top-5 Accuracy (%)')
        ax.set_title('Top 10 Best Performing Configurations')
        
        # 6. Heatmap of all results
        ax = axes[1, 2]
        # Create pivot table for heatmap
        heatmap_data = self.results_df.pivot_table(
            values='top_5_accuracy',
            index='experiment',
            columns='dataset',
            aggfunc='mean'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax)
        ax.set_title('Top-5 Accuracy Heatmap')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Results visualization saved to: {output_file}")
    
    def generate_report(self, output_file: str = 'ml_evaluation_report.txt'):
        """Generate detailed text report of results"""
        if self.results_df is None or len(self.results_df) == 0:
            print("No results to report")
            return
        
        report = f"""TypeNet ML Evaluation Report
{'='*50}

Evaluation Summary
------------------
Total experiments evaluated: {len(self.results_df)}
Imputation strategies: {', '.join(self.results_df['imputation'].unique())}
Dataset types: {', '.join(self.results_df['dataset'].unique())}

Overall Performance
-------------------
Average Top-1 Accuracy: {self.results_df['top_1_accuracy'].mean():.2f}%
Average Top-3 Accuracy: {self.results_df['top_3_accuracy'].mean():.2f}%
Average Top-5 Accuracy: {self.results_df['top_5_accuracy'].mean():.2f}%

Best Performing Configuration
-----------------------------
"""
        
        best_config = self.results_df.loc[self.results_df['top_5_accuracy'].idxmax()]
        report += f"Experiment: {best_config['experiment']}\n"
        report += f"Dataset: {best_config['dataset']}\n"
        report += f"Imputation: {best_config['imputation']}\n"
        report += f"Top-5 Accuracy: {best_config['top_5_accuracy']:.2f}%\n"
        
        report += "\nPerformance by Dataset Type\n"
        report += "---------------------------\n"
        dataset_summary = self.results_df.groupby('dataset')[['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy']].mean()
        report += dataset_summary.to_string()
        
        report += "\n\nPerformance by Experiment Type\n"
        report += "-------------------------------\n"
        
        # Session experiments
        session_results = self.results_df[self.results_df['experiment'].str.contains('session')]
        if len(session_results) > 0:
            report += "\nSession Experiments:\n"
            report += session_results.groupby('dataset')[['top_1_accuracy', 'top_5_accuracy']].mean().to_string()
        
        # Platform experiments
        platform_results = self.results_df[self.results_df['experiment'].str.contains('platform')]
        if len(platform_results) > 0:
            report += "\n\nPlatform Experiments:\n"
            report += platform_results.groupby('experiment')[['top_5_accuracy']].mean().sort_values('top_5_accuracy', ascending=False).head(10).to_string()
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Evaluation report saved to: {output_file}")


# Example usage and demonstration
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = TypeNetMLEvaluator('ml_experiments')
    
    # Example: Evaluate a single experiment
    print("Example: Evaluating a single experiment configuration")
    result = evaluator.evaluate_experiment(
        imputation='global',
        experiment_name='session_1vs2',
        dataset_type='dataset_1',
        similarity_method='cosine',
        normalize=True
    )
    print(f"Result: {result}")
    
    # Run all experiments
    print("\n" + "="*60)
    print("Running complete evaluation...")
    results_df = evaluator.run_all_experiments()
    
    # Generate visualizations and report
    evaluator.plot_results_summary()
    evaluator.generate_report()
    
    print("\n✅ Evaluation complete!")
    print("Generated files:")
    print("- ml_experiments/evaluation_results.csv")
    print("- ml_results_summary.png")
    print("- ml_evaluation_report.txt")