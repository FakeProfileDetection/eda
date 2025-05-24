import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import socket

warnings.filterwarnings('ignore')

class TypeNetEDA:
    """
    Comprehensive Exploratory Data Analysis for TypeNet keystroke data
    """
    
    def __init__(self, data_path='typenet_features_extracted.csv'):
        """Initialize with the extracted features dataset"""
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        self.timing_features = ['HL', 'IL', 'PL', 'RL']
        self.grouping_vars = ['platform_id', 'video_id', 'session_id', 'user_id']
        
        # Convert timing features from nanoseconds to milliseconds
        for feature in self.timing_features:
            self.df[f'{feature}_ms'] = self.df[feature] / 1_000_000
        
        print(f"Loaded {len(self.df):,} keystroke pairs from {self.df['user_id'].nunique()} users")
    
    def generate_summary_statistics(self):
        """Generate overall summary statistics"""
        summary = {
            'Total Records': len(self.df),
            'Total Users': self.df['user_id'].nunique(),
            'Total Platforms': self.df['platform_id'].nunique(),
            'Total Videos': self.df['video_id'].nunique(),
            'Total Sessions': len(self.df.groupby(['user_id', 'platform_id', 'video_id', 'session_id'])),
            'Valid Records': self.df['valid'].sum(),
            'Invalid Records': (~self.df['valid']).sum(),
            'Validity Rate': f"{(self.df['valid'].sum() / len(self.df) * 100):.2f}%"
        }
        
        return summary
    
    def analyze_errors(self):
        """Detailed error analysis"""
        error_df = self.df[~self.df['valid']].copy()
        
        # Error distribution
        error_dist = error_df['error_description'].value_counts()
        
        # Error rates by grouping variables
        error_rates = {}
        for var in self.grouping_vars:
            rates = self.df.groupby(var)['valid'].agg(['sum', 'count'])
            rates['error_rate'] = 1 - (rates['sum'] / rates['count'])
            error_rates[var] = rates.sort_values('error_rate', ascending=False)
        
        # Most common error keys
        error_keys = pd.concat([
            error_df['key1'].value_counts().head(10),
            error_df['key2'].value_counts().head(10)
        ]).groupby(level=0).sum().sort_values(ascending=False).head(10)
        
        return {
            'error_distribution': error_dist,
            'error_rates_by_group': error_rates,
            'common_error_keys': error_keys
        }
    
    def analyze_timing_features(self):
        """Analyze timing feature distributions"""
        valid_df = self.df[self.df['valid']].copy()
        
        timing_stats = {}
        for feature in self.timing_features:
            feature_ms = f'{feature}_ms'
            if feature_ms in valid_df.columns:
                # Remove outliers for better visualization (keep 99.5% of data)
                q_low = valid_df[feature_ms].quantile(0.0025)
                q_high = valid_df[feature_ms].quantile(0.9975)
                filtered = valid_df[(valid_df[feature_ms] >= q_low) & (valid_df[feature_ms] <= q_high)]
                
                timing_stats[feature] = {
                    'mean': valid_df[feature_ms].mean(),
                    'median': valid_df[feature_ms].median(),
                    'std': valid_df[feature_ms].std(),
                    'min': valid_df[feature_ms].min(),
                    'max': valid_df[feature_ms].max(),
                    'q25': valid_df[feature_ms].quantile(0.25),
                    'q75': valid_df[feature_ms].quantile(0.75),
                    'negative_count': (valid_df[feature_ms] < 0).sum() if feature in ['IL', 'RL'] else 0
                }
        
        return timing_stats
    
    def analyze_key_patterns(self):
        """Analyze most common keys and key combinations"""
        valid_df = self.df[self.df['valid']].copy()
        
        # Most common individual keys
        all_keys = pd.concat([valid_df['key1'], valid_df['key2']])
        key_freq = all_keys.value_counts().head(20)
        
        # Most common key pairs
        valid_df['key_pair'] = valid_df['key1'] + '->' + valid_df['key2']
        pair_freq = valid_df['key_pair'].value_counts().head(20)
        
        # Key transition matrix (top 10 keys)
        top_keys = all_keys.value_counts().head(10).index
        transition_matrix = pd.crosstab(
            valid_df[valid_df['key1'].isin(top_keys)]['key1'],
            valid_df[valid_df['key2'].isin(top_keys)]['key2'],
            normalize='index'
        )
        
        return {
            'key_frequency': key_freq,
            'pair_frequency': pair_freq,
            'transition_matrix': transition_matrix
        }
    
    def analyze_variations(self):
        """Analyze variations across platforms, videos, sessions, and users"""
        valid_df = self.df[self.df['valid']].copy()
        variations = {}
        
        for group_var in self.grouping_vars:
            group_stats = []
            
            for feature in self.timing_features:
                feature_ms = f'{feature}_ms'
                stats = valid_df.groupby(group_var)[feature_ms].agg(['mean', 'std', 'count'])
                stats['feature'] = feature
                stats['cv'] = stats['std'] / stats['mean']  # Coefficient of variation
                group_stats.append(stats)
            
            variations[group_var] = pd.concat(group_stats)
        
        return variations
    
    def create_interactive_dashboard(self, output_file='typenet_eda_dashboard.html'):
        """Create an interactive Plotly dashboard"""
        print("Creating interactive dashboard...")
        
        # Initialize figure with subplots
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Error Distribution', 'Timing Feature Distributions', 'Key Frequency',
                'Error Rates by Platform', 'HL Distribution by User', 'IL Distribution by User',
                'Key Transition Heatmap', 'PL vs RL Scatter', 'Session Length Distribution',
                'Platform Comparison', 'Video Comparison', 'Temporal Patterns'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'box'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'violin'}, {'type': 'violin'}],
                [{'type': 'heatmap'}, {'type': 'scatter'}, {'type': 'histogram'}],
                [{'type': 'box'}, {'type': 'box'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Error Distribution
        error_dist = self.df[~self.df['valid']]['error_description'].value_counts()
        fig.add_trace(
            go.Bar(x=error_dist.index, y=error_dist.values, name='Errors'),
            row=1, col=1
        )
        
        # 2. Timing Feature Distributions
        valid_df = self.df[self.df['valid']].copy()
        for i, feature in enumerate(self.timing_features):
            feature_ms = f'{feature}_ms'
            # Remove extreme outliers for visualization
            q99 = valid_df[feature_ms].quantile(0.99)
            filtered = valid_df[valid_df[feature_ms] <= q99]
            
            fig.add_trace(
                go.Box(
                    y=filtered[feature_ms],
                    name=feature,
                    boxpoints=False
                ),
                row=1, col=2
            )
        
        # 3. Key Frequency (top 15)
        all_keys = pd.concat([valid_df['key1'], valid_df['key2']])
        key_freq = all_keys.value_counts().head(15)
        fig.add_trace(
            go.Bar(x=key_freq.index, y=key_freq.values, name='Key Frequency'),
            row=1, col=3
        )
        
        # 4. Error Rates by Platform
        platform_errors = self.df.groupby('platform_id')['valid'].agg(['sum', 'count'])
        platform_errors['error_rate'] = 1 - (platform_errors['sum'] / platform_errors['count'])
        fig.add_trace(
            go.Bar(
                x=platform_errors.index.astype(str),
                y=platform_errors['error_rate'] * 100,
                name='Error Rate %'
            ),
            row=2, col=1
        )
        
        # 5 & 6. HL and IL Distribution by User (sample users)
        sample_users = valid_df['user_id'].value_counts().head(10).index
        
        for user in sample_users:
            user_data = valid_df[valid_df['user_id'] == user]
            
            # HL by user
            fig.add_trace(
                go.Violin(
                    y=user_data['HL_ms'],
                    name=f'User {user}',
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # IL by user
            fig.add_trace(
                go.Violin(
                    y=user_data['IL_ms'],
                    name=f'User {user}',
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=False
                ),
                row=2, col=3
            )
        
        # 7. Key Transition Heatmap
        top_keys = all_keys.value_counts().head(10).index
        transition_matrix = pd.crosstab(
            valid_df[valid_df['key1'].isin(top_keys)]['key1'],
            valid_df[valid_df['key2'].isin(top_keys)]['key2']
        )
        
        fig.add_trace(
            go.Heatmap(
                z=transition_matrix.values,
                x=transition_matrix.columns,
                y=transition_matrix.index,
                colorscale='Blues',
                showscale=False
            ),
            row=3, col=1
        )
        
        # 8. PL vs RL Scatter (sample for performance)
        sample_df = valid_df.sample(min(5000, len(valid_df)))
        fig.add_trace(
            go.Scatter(
                x=sample_df['PL_ms'],
                y=sample_df['RL_ms'],
                mode='markers',
                marker=dict(size=3, opacity=0.5),
                name='PL vs RL'
            ),
            row=3, col=2
        )
        
        # 9. Session Length Distribution
        session_lengths = self.df.groupby(['user_id', 'platform_id', 'video_id', 'session_id']).size()
        fig.add_trace(
            go.Histogram(
                x=session_lengths.values,
                nbinsx=50,
                name='Session Lengths'
            ),
            row=3, col=3
        )
        
        # 10-12. Platform, Video, and Temporal comparisons
        for i, (feature, row_col) in enumerate([
            ('HL_ms', (4, 1)),
            ('IL_ms', (4, 2)),
            ('PL_ms', (4, 3))
        ]):
            # Platform comparison
            platform_data = valid_df.groupby('platform_id')[feature].mean().sort_values()
            fig.add_trace(
                go.Box(
                    x=valid_df['platform_id'],
                    y=valid_df[feature],
                    name=f'{feature} by Platform',
                    showlegend=False
                ),
                row=row_col[0], col=row_col[1]
            )
        
        # Update layout
        fig.update_layout(
            height=2000,
            width=1800,
            title_text="TypeNet Keystroke Data - Exploratory Data Analysis Dashboard",
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Error Type", row=1, col=1)
        fig.update_xaxes(title_text="Feature", row=1, col=2)
        fig.update_xaxes(title_text="Key", row=1, col=3)
        fig.update_xaxes(title_text="Platform ID", row=2, col=1)
        fig.update_xaxes(title_text="Key1", row=3, col=1)
        fig.update_xaxes(title_text="PL (ms)", row=3, col=2)
        fig.update_xaxes(title_text="Session Length", row=3, col=3)
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Time (ms)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=3)
        fig.update_yaxes(title_text="Error Rate (%)", row=2, col=1)
        fig.update_yaxes(title_text="HL (ms)", row=2, col=2)
        fig.update_yaxes(title_text="IL (ms)", row=2, col=3)
        fig.update_yaxes(title_text="Key2", row=3, col=1)
        fig.update_yaxes(title_text="RL (ms)", row=3, col=2)
        fig.update_yaxes(title_text="Count", row=3, col=3)
        
        # Save dashboard
        output_path = os.path.join(self.output_dir, output_file)
        fig.write_html(output_path)
        print(f"Interactive dashboard saved to {output_path}")
        
        return fig
    
    def generate_detailed_report(self, output_file='typenet_eda_report.html'):
        """Generate a comprehensive HTML report with all analyses"""
        print("Generating detailed report...")
        
        # Gather all analyses
        summary = self.generate_summary_statistics()
        errors = self.analyze_errors()
        timing = self.analyze_timing_features()
        keys = self.analyze_key_patterns()
        variations = self.analyze_variations()
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TypeNet EDA Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; }}
                h3 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #007bff; color: white; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary-box {{ background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                .metric-label {{ color: #666; }}
                .warning {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .success {{ background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>TypeNet Keystroke Data - Exploratory Data Analysis Report</h1>
            
            <div class="summary-box">
                <h2>Executive Summary</h2>
                {"".join([f'<div class="metric"><div class="metric-label">{k}</div><div class="metric-value">{v}</div></div>' for k, v in summary.items()])}
            </div>
            
            <h2>1. Data Quality Analysis</h2>
            
            <h3>1.1 Error Distribution</h3>
            <table>
                <tr><th>Error Type</th><th>Count</th><th>Percentage</th></tr>
                {"".join([f'<tr><td>{error}</td><td>{count}</td><td>{count/len(self.df)*100:.2f}%</td></tr>' for error, count in errors['error_distribution'].items()])}
            </table>
            
            <h3>1.2 Error Rates by Grouping Variables</h3>
        """
        
        # Add error rates tables
        for var in self.grouping_vars:
            html_content += f"""
            <h4>Error Rates by {var}</h4>
            <table>
                <tr><th>{var}</th><th>Total Records</th><th>Valid Records</th><th>Error Rate</th></tr>
            """
            error_data = errors['error_rates_by_group'][var]
            for idx, row in error_data.iterrows():
                error_rate_pct = row['error_rate'] * 100
                row_class = 'warning' if error_rate_pct > 10 else ''
                html_content += f"""
                <tr class="{row_class}">
                    <td>{idx}</td>
                    <td>{row['count']}</td>
                    <td>{row['sum']}</td>
                    <td>{error_rate_pct:.2f}%</td>
                </tr>
                """
            html_content += "</table>"
        
        # Add timing features analysis
        html_content += """
            <h2>2. Timing Features Analysis</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Mean (ms)</th>
                    <th>Median (ms)</th>
                    <th>Std Dev (ms)</th>
                    <th>Min (ms)</th>
                    <th>Max (ms)</th>
                    <th>Q25 (ms)</th>
                    <th>Q75 (ms)</th>
                    <th>Negative Values</th>
                </tr>
        """
        
        for feature, stats in timing.items():
            html_content += f"""
                <tr>
                    <td><b>{feature}</b></td>
                    <td>{stats['mean']:.2f}</td>
                    <td>{stats['median']:.2f}</td>
                    <td>{stats['std']:.2f}</td>
                    <td>{stats['min']:.2f}</td>
                    <td>{stats['max']:.2f}</td>
                    <td>{stats['q25']:.2f}</td>
                    <td>{stats['q75']:.2f}</td>
                    <td>{stats['negative_count']}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <div class="warning">
                <b>Note on Negative Values:</b> IL (Inter-key Latency) and RL (Release Latency) can be negative, 
                indicating key overlap or release order different from press order. This is normal typing behavior.
            </div>
            
            <h2>3. Key Pattern Analysis</h2>
            
            <h3>3.1 Most Common Keys</h3>
            <table>
                <tr><th>Key</th><th>Frequency</th></tr>
        """
        
        for key, freq in keys['key_frequency'].head(10).items():
            html_content += f"<tr><td>{key}</td><td>{freq:,}</td></tr>"
        
        html_content += """
            </table>
            
            <h3>3.2 Most Common Key Pairs</h3>
            <table>
                <tr><th>Key Pair</th><th>Frequency</th></tr>
        """
        
        for pair, freq in keys['pair_frequency'].head(10).items():
            html_content += f"<tr><td>{pair}</td><td>{freq:,}</td></tr>"
        
        # Add variation analysis summary
        html_content += """
            </table>
            
            <h2>4. Variation Analysis</h2>
            <p>This section analyzes how typing patterns vary across different grouping variables.</p>
        """
        
        # Add recommendations
        html_content += """
            <h2>5. Key Findings and Recommendations</h2>
            <div class="success">
                <h3>Data Quality Insights:</h3>
                <ul>
        """
        
        validity_rate = (self.df['valid'].sum() / len(self.df) * 100)
        if validity_rate > 90:
            html_content += "<li>✓ High data quality with >90% valid records</li>"
        else:
            html_content += f"<li>⚠ Data quality concerns: only {validity_rate:.1f}% valid records</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="warning">
                <h3>Areas for Investigation:</h3>
                <ul>
                    <li>Review sessions/users with high error rates for data collection issues</li>
                    <li>Investigate keys with frequent errors (may indicate hardware/software issues)</li>
                    <li>Consider filtering criteria for outlier timing values in downstream analysis</li>
                </ul>
            </div>
            
            <p><i>Report generated on: {}</i></p>
        </body>
        </html>
        """.format(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Save report
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Detailed report saved to {output_path}")
        
        return output_path
    
    def generate_statistical_summary(self, output_file='typenet_statistics.csv'):
        """Generate CSV with detailed statistics for further analysis"""
        print("Generating statistical summary...")
        
        valid_df = self.df[self.df['valid']].copy()
        
        # User-level statistics
        user_stats = []
        for user in valid_df['user_id'].unique():
            user_data = valid_df[valid_df['user_id'] == user]
            
            stats = {
                'user_id': user,
                'total_keystrokes': len(user_data),
                'unique_keys': user_data['key1'].nunique(),
                'error_rate': 1 - (len(user_data) / len(self.df[self.df['user_id'] == user]))
            }
            
            # Add timing statistics
            for feature in self.timing_features:
                feature_ms = f'{feature}_ms'
                stats[f'{feature}_mean'] = user_data[feature_ms].mean()
                stats[f'{feature}_std'] = user_data[feature_ms].std()
                stats[f'{feature}_median'] = user_data[feature_ms].median()
            
            user_stats.append(stats)
        
        output_path = os.path.join(self.output_dir, output_file)
        user_stats_df = pd.DataFrame(user_stats)
        user_stats_df.to_csv(output_path, index=False)
        print(f"Statistical summary saved to {output_path}")
        
        return user_stats_df
    
    def run_complete_analysis(self):
        """Run all analyses and generate all outputs"""
        
        
        hostname = socket.gethostname()
        now = datetime.now().strftime("%Y-%m-%d_%H%M%S") # Removed Min and Sec suffix for brevity
        base_dir = os.path.dirname(os.path.abspath(__file__)) # Or use os.getcwd() if script is run from its location
        self.output_dir = os.path.join(base_dir, f"typenet-extraction-eda--{now}-{hostname}")
        os.makedirs(self.output_dir, exist_ok=True)        
        
        print("\n" + "="*60)
        print("Running Complete TypeNet EDA")
        print("="*60 + "\n")
        
        # Generate all outputs
        dashboard = self.create_interactive_dashboard()
        report = self.generate_detailed_report()
        stats = self.generate_statistical_summary()
        
        # Print summary to console
        summary = self.generate_summary_statistics()
        print("\nAnalysis Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Analysis complete! Generated files:")
        print("  1. typenet_eda_dashboard.html - Interactive dashboard")
        print("  2. typenet_eda_report.html - Detailed statistical report")
        print("  3. typenet_statistics.csv - User-level statistics for further analysis")
        print("\nShare these files with your team for collaborative analysis!")
        
        return dashboard, report, stats


# Script execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TypeNetEDA('processed_data-2025-05-24_144726-Loris-MBP.cable.rcn.com/typenet_features.csv')
    
    # Run complete analysis
    analyzer.run_complete_analysis()