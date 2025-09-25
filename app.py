#!/usr/bin/env python3
"""
Flask app for visualizing evaluation metrics from SimplePredictions models.

Provides web interface to select models and generate 4x2 grid of plots:
- Rows: precision@k, recall@k, total_recall@k, hits@k  
- Columns: range 1-1000, range 1-max (full evaluation set)
"""

import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Any
import io
import base64

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

def discover_models() -> Dict[str, Any]:
    """Discover all models with evaluation_metrics.json files, organized hierarchically."""
    models_hierarchy = {}
    
    pattern = "graphs/**/evaluation_metrics.json"
    for metrics_file in glob.glob(pattern, recursive=True):
        model_dir = os.path.dirname(metrics_file)
        provenance_file = os.path.join(model_dir, "provenance.json")
        
        if not os.path.exists(provenance_file):
            continue
            
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            with open(provenance_file, 'r') as f:
                provenance = json.load(f)
                
            # Parse the directory structure: graphs/{graph_name}/embeddings/{embed_version}/models/{model_version}
            path_parts = model_dir.split('/')
            if len(path_parts) < 6 or path_parts[0] != 'graphs':
                continue
                
            graph_name = path_parts[1]
            embed_version = path_parts[3] 
            model_version = path_parts[5]
            
            # Extract meaningful info from provenance
            negative_method = provenance['input_data']['negative_sampling_method']
            has_contraindications = provenance['input_data']['contraindications'] is not None
            
            # Create descriptive labels
            negative_label = "Contraindications" if has_contraindications else "Random"
            
            # Get graph description from graph provenance if available
            graph_description = None
            # model_dir is like: graphs/robokop_base_nonredundant_CCDD/embeddings/embeddings_0/models/model_0
            # We need: graphs/robokop_base_nonredundant_CCDD/graph/provenance.json
            graph_dir = os.path.join(path_parts[0], path_parts[1], "graph", "provenance.json")
            if os.path.exists(graph_dir):
                try:
                    with open(graph_dir, 'r') as f:
                        graph_provenance = json.load(f)
                        graph_description = graph_provenance.get('description', None)
                except:
                    pass
            
            # Build model info
            model_info = {
                'id': model_dir,
                'model_dir': model_dir,
                'graph_name': graph_name,
                'graph_description': graph_description,
                'embed_version': embed_version,
                'model_version': model_version,
                'negative_method': negative_method,
                'negative_label': negative_label,
                'has_contraindications': has_contraindications,
                'timestamp': metrics['timestamp'],
                'total_combinations': metrics['evaluation_summary']['all_combinations'],
                'evaluation_positives': metrics['evaluation_summary']['evaluation_positives'],
                'metrics': metrics,
                'provenance': provenance,
                'embedding_dim': provenance['input_data']['embedding_info']['embedding_dim'],
                'embedding_params': provenance['input_data']['embedding_info']['embedding_provenance']['parameters']
            }
            
            # Organize hierarchically
            if graph_name not in models_hierarchy:
                models_hierarchy[graph_name] = {}
            if embed_version not in models_hierarchy[graph_name]:
                models_hierarchy[graph_name][embed_version] = {}
            
            models_hierarchy[graph_name][embed_version][model_version] = model_info
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {metrics_file}: {e}")
            continue
    
    # Sort the hierarchy alphabetically
    sorted_hierarchy = {}
    for graph_name in sorted(models_hierarchy.keys()):
        sorted_hierarchy[graph_name] = {}
        for embed_version in sorted(models_hierarchy[graph_name].keys()):
            sorted_hierarchy[graph_name][embed_version] = {}
            for model_version in sorted(models_hierarchy[graph_name][embed_version].keys()):
                sorted_hierarchy[graph_name][embed_version][model_version] = models_hierarchy[graph_name][embed_version][model_version]
    
    return sorted_hierarchy

def extract_graph_type(model_dir: str) -> str:
    """Extract graph type from model directory path."""
    parts = model_dir.split('/')
    for part in parts:
        if 'CCDD' in part or 'CGD' in part:
            return part
    return 'unknown'

def flatten_models(models_hierarchy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the hierarchical models structure for plotting."""
    flat_models = []
    for graph_name in models_hierarchy:
        for embed_version in models_hierarchy[graph_name]:
            for model_version in models_hierarchy[graph_name][embed_version]:
                flat_models.append(models_hierarchy[graph_name][embed_version][model_version])
    return flat_models

def generate_plots(selected_models: List[Dict[str, Any]]) -> str:
    """Generate 4x2 grid of plots for selected models."""
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('Model Evaluation Metrics Comparison', fontsize=16, y=0.98)
    
    # Define metrics and their labels
    metrics_info = [
        ('precision_at_k', 'Precision@K'),
        ('recall_at_k', 'Recall@K'), 
        ('total_recall_at_k', 'Total Recall@K'),
        ('hits_at_k', 'Hits@K')
    ]
    
    # Define k ranges for columns
    k_ranges = [
        (1, 1000, '1-1000'),
        (1, None, '1-Max')  # None means use all k values
    ]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_models)))
    
    for row, (metric_key, metric_label) in enumerate(metrics_info):
        for col, (k_min, k_max, range_label) in enumerate(k_ranges):
            ax = axes[row, col]
            
            for i, model in enumerate(selected_models):
                metrics = model['metrics']
                k_values = [int(k) for k in metrics['k_values']]
                metric_values = [metrics[metric_key][str(k)] for k in k_values]
                
                # Filter by k range
                if k_max is not None:
                    filtered_data = [(k, v) for k, v in zip(k_values, metric_values) if k_min <= k <= k_max]
                else:
                    filtered_data = [(k, v) for k, v in zip(k_values, metric_values) if k >= k_min]
                
                if filtered_data:
                    ks, values = zip(*filtered_data)
                    
                    # Create model label with more descriptive info
                    label = f"{model['graph_name']} ({model['negative_label']})"
                    
                    ax.plot(ks, values, marker='o', linewidth=2, markersize=4, 
                           color=colors[i], label=label, alpha=0.8)
            
            ax.set_xlabel('K')
            ax.set_ylabel(metric_label)
            ax.set_title(f'{metric_label} (K range: {range_label})')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            
            # Add total recall max line for total recall plots
            if metric_key == 'total_recall_at_k':
                for model in selected_models:
                    total_recall_max = model['metrics']['total_recall_max']
                    ax.axhline(y=total_recall_max, linestyle='--', alpha=0.5, 
                              color='gray', linewidth=1)
            
            # Only add legend to top-right plot to avoid clutter
            if row == 0 and col == 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_str

@app.route('/')
def index():
    """Main page showing available models."""
    models_hierarchy = discover_models()
    return render_template('index.html', models_hierarchy=models_hierarchy)

@app.route('/generate_plots', methods=['POST'])
def generate_plots_endpoint():
    """Generate plots for selected models."""
    data = request.get_json()
    selected_model_ids = data.get('model_ids', [])
    
    if not selected_model_ids:
        return jsonify({'error': 'No models selected'}), 400
    
    # Get model data for selected IDs
    models_hierarchy = discover_models()
    all_models = flatten_models(models_hierarchy)
    selected_models = [m for m in all_models if m['id'] in selected_model_ids]
    
    if not selected_models:
        return jsonify({'error': 'Selected models not found'}), 404
    
    try:
        img_str = generate_plots(selected_models)
        return jsonify({'image': img_str})
    except Exception as e:
        return jsonify({'error': f'Plot generation failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)