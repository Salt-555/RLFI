"""
Lineage Tracker - Visualize model family trees and genetic evolution
"""
import streamlit as st
import pandas as pd
import sqlite3
import json
import os
import yaml
from datetime import datetime
from collections import defaultdict

st.set_page_config(
    page_title="Lineage | RLFI",
    page_icon="🧬",
    layout="wide"
)

# Amber terminal CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&display=swap');
    
    *:not([data-testid="collapsedControl"]):not([class*="icon"]):not(svg):not(path) { font-family: 'Share Tech Mono', 'Courier New', monospace !important; }
    
    .stApp {
        background-color: #141414;
        background-image: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(20, 10, 0, 0.3) 2px, rgba(20, 10, 0, 0.3) 4px);
    }
    
    .page-header {
        font-family: 'VT323', monospace !important;
        font-size: 2.5rem;
        color: #ffaa00;
        border-bottom: 1px dashed #996600;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .tree-node {
        background: #1a1a1a;
        border: 1px solid #996600;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        display: inline-block;
    }
    
    .tree-node-promoted { border-color: #44ff44; }
    .tree-node-culled { border-color: #cc3333; }
    .tree-node-active { border-color: #ffaa00; border-width: 2px; }
    
    .generation-label {
        color: #996600;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }
    
    .family-tree {
        background: #1a1a1a;
        border: 1px solid #996600;
        padding: 1rem;
        margin: 1rem 0;
        overflow-x: auto;
    }
    
    .tree-line { color: #996600; }
    
    h1, h2, h3 { color: #ffaa00 !important; font-family: 'VT323', monospace !important; }
    .stMarkdown, p, span, div { color: #cc8800; }
    hr { border-color: #996600 !important; border-style: dashed !important; }
    
    .stat-box {
        background: #1a1a1a;
        border: 1px solid #cc8800;
        padding: 1rem;
        text-align: center;
    }
    .stat-number {
        font-family: 'VT323', monospace !important;
        font-size: 2rem;
        color: #ffcc00;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #996600;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


def get_all_models_with_lineage():
    """Get all models and their parent relationships."""
    models = {}
    
    # Scan metadata files for lineage info
    dirs = ['autotest_models', 'models', 'champion_models']
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            continue
        
        for filename in os.listdir(dir_path):
            if filename.endswith('_metadata.yaml'):
                filepath = os.path.join(dir_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        meta = yaml.safe_load(f)
                    
                    if meta and 'model_id' in meta:
                        model_id = meta['model_id']
                        models[model_id] = {
                            'model_id': model_id,
                            'parent_model_id': meta.get('parent_model_id'),
                            'generation': meta.get('generation', 1),
                            'algorithm': meta.get('algorithm'),
                            'tickers': meta.get('tickers', []),
                            'total_timesteps': meta.get('total_timesteps'),
                            'training_date': meta.get('training_date'),
                            'lineage': meta.get('lineage', [])
                        }
                except:
                    continue
    
    # Also get lifecycle state from database
    db_path = 'autotest_strategies.db'
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT model_id, current_state, backtest_expected_sharpe FROM model_lifecycle')
            for row in cursor.fetchall():
                model_id, state, sharpe = row
                if model_id in models:
                    models[model_id]['state'] = state
                    models[model_id]['sharpe'] = sharpe
                else:
                    # Model in DB but no metadata file
                    models[model_id] = {
                        'model_id': model_id,
                        'parent_model_id': None,
                        'generation': 1,
                        'state': state,
                        'sharpe': sharpe
                    }
            conn.close()
        except:
            pass
    
    return models


def build_family_trees(models):
    """Build family tree structures from model data."""
    # Find root models (no parent)
    roots = []
    children_map = defaultdict(list)
    
    for model_id, data in models.items():
        parent = data.get('parent_model_id')
        if parent and parent in models:
            children_map[parent].append(model_id)
        elif not parent:
            roots.append(model_id)
    
    # Also add models whose parents aren't in our data as roots
    for model_id, data in models.items():
        parent = data.get('parent_model_id')
        if parent and parent not in models and model_id not in roots:
            roots.append(model_id)
    
    return roots, children_map


def render_tree_node(model_id, models, level=0):
    """Render a single tree node with ASCII art."""
    data = models.get(model_id, {})
    state = data.get('state', 'unknown')
    gen = data.get('generation', 1)
    sharpe = data.get('sharpe')
    algo = data.get('algorithm', '?')
    
    # State styling
    state_class = ''
    if state == 'promoted':
        state_class = 'tree-node-promoted'
        state_icon = '[+]'
    elif state == 'culled':
        state_class = 'tree-node-culled'
        state_icon = '[X]'
    elif state == 'paper_trading':
        state_class = 'tree-node-active'
        state_icon = '[>]'
    else:
        state_icon = '[ ]'
    
    sharpe_str = f"S:{sharpe:.2f}" if sharpe else ""
    algo_str = algo.upper()[:3] if algo else ""
    
    indent = "│   " * level
    branch = "├── " if level > 0 else ""
    
    return f'{indent}{branch}<span class="tree-node {state_class}">{state_icon} {model_id} <span style="color:#996600">G{gen} {algo_str} {sharpe_str}</span></span>'


def render_family_tree(root, models, children_map, level=0, is_last=False):
    """Recursively render a family tree."""
    lines = []
    lines.append(render_tree_node(root, models, level))
    
    children = children_map.get(root, [])
    for i, child in enumerate(children):
        child_is_last = (i == len(children) - 1)
        lines.extend(render_family_tree(child, models, children_map, level + 1, child_is_last))
    
    return lines


# Page Header
st.markdown('<div class="page-header">[GENETICS] LINEAGE TRACKER</div>', unsafe_allow_html=True)

# Load data
models = get_all_models_with_lineage()

if not models:
    st.warning("No models with lineage data found.")
    st.info("Lineage tracking requires model metadata files with parent_model_id fields.")
    st.stop()

roots, children_map = build_family_trees(models)

# Stats
st.markdown("### Evolution Statistics")
col1, col2, col3, col4 = st.columns(4)

total_models = len(models)
gen1_models = sum(1 for m in models.values() if m.get('generation', 1) == 1)
offspring_models = total_models - gen1_models
max_gen = max((m.get('generation', 1) for m in models.values()), default=1)

with col1:
    st.markdown(f'''
    <div class="stat-box">
        <div class="stat-number">{total_models}</div>
        <div class="stat-label">Total Models</div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
    <div class="stat-box">
        <div class="stat-number">{gen1_models}</div>
        <div class="stat-label">Original (G1)</div>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown(f'''
    <div class="stat-box">
        <div class="stat-number">{offspring_models}</div>
        <div class="stat-label">Offspring</div>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    st.markdown(f'''
    <div class="stat-box">
        <div class="stat-number">{max_gen}</div>
        <div class="stat-label">Max Generation</div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

# Family Trees
st.markdown("### Family Trees")
st.markdown("""
<p style="color: #996600; font-size: 0.85rem;">
Legend: <span style="color:#44ff44">[+] Promoted</span> | 
<span style="color:#cc3333">[X] Culled</span> | 
<span style="color:#ffaa00">[>] Active</span> | 
[ ] Other | G# = Generation | S: = Sharpe
</p>
""", unsafe_allow_html=True)

# Render each family tree
families_with_offspring = [(r, children_map.get(r, [])) for r in roots if children_map.get(r)]
families_without_offspring = [r for r in roots if not children_map.get(r)]

if families_with_offspring:
    st.markdown("#### Families with Offspring")
    
    for root, children in families_with_offspring:
        tree_lines = render_family_tree(root, models, children_map)
        tree_html = '<br>'.join(tree_lines)
        
        with st.expander(f"Family: {root} ({len(children)} direct offspring)", expanded=True):
            st.markdown(f'<div class="family-tree">{tree_html}</div>', unsafe_allow_html=True)

if families_without_offspring:
    st.markdown("#### Standalone Models (No Offspring)")
    
    # Group by generation
    by_gen = defaultdict(list)
    for model_id in families_without_offspring:
        gen = models[model_id].get('generation', 1)
        by_gen[gen].append(model_id)
    
    for gen in sorted(by_gen.keys()):
        with st.expander(f"Generation {gen} ({len(by_gen[gen])} models)"):
            for model_id in by_gen[gen]:
                node_html = render_tree_node(model_id, models, 0)
                st.markdown(node_html, unsafe_allow_html=True)

st.markdown("---")

# Generation breakdown
st.markdown("### Generation Breakdown")

gen_data = defaultdict(lambda: {'count': 0, 'promoted': 0, 'culled': 0, 'avg_sharpe': []})

for model_id, data in models.items():
    gen = data.get('generation', 1)
    gen_data[gen]['count'] += 1
    
    state = data.get('state')
    if state == 'promoted':
        gen_data[gen]['promoted'] += 1
    elif state == 'culled':
        gen_data[gen]['culled'] += 1
    
    sharpe = data.get('sharpe')
    if sharpe is not None:
        gen_data[gen]['avg_sharpe'].append(sharpe)

# Create dataframe
gen_rows = []
for gen in sorted(gen_data.keys()):
    d = gen_data[gen]
    avg_sharpe = sum(d['avg_sharpe']) / len(d['avg_sharpe']) if d['avg_sharpe'] else None
    survival_rate = (d['count'] - d['culled']) / d['count'] * 100 if d['count'] > 0 else 0
    
    gen_rows.append({
        'Generation': f"G{gen}",
        'Count': d['count'],
        'Promoted': d['promoted'],
        'Culled': d['culled'],
        'Survival %': f"{survival_rate:.0f}%",
        'Avg Sharpe': f"{avg_sharpe:.2f}" if avg_sharpe else "N/A"
    })

if gen_rows:
    gen_df = pd.DataFrame(gen_rows)
    st.dataframe(gen_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #996600; font-size: 0.8rem;">GENETIC LINEAGE TRACKER // RLFI COLOSSEUM</p>', unsafe_allow_html=True)
