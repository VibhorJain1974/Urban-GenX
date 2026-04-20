#!/usr/bin/env python
"""Run graphify incremental update and save results to file."""
import json
import sys
from pathlib import Path
from graphify.detect import detect_incremental
from graphify.extract import collect_files, extract
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections
from graphify.report import generate
from graphify.export import to_json
from networkx.readwrite import json_graph
import networkx as nx

output = {
    'status': 'started',
    'steps': {}
}

try:
    # Step 1: Detect changes
    print("Step 1: Detecting changes...", file=sys.stderr)
    result = detect_incremental(Path('.'))
    files_total = result.get('total_files', 0)
    files_changed = result.get('new_total', 0)
    words_total = result.get('total_words', 0)

    output['steps']['detection'] = {
        'files_total': files_total,
        'files_changed': files_changed,
        'words_total': words_total,
        'new_files': result.get('new_files', {})
    }

    if files_changed == 0:
        output['status'] = 'no_changes'
        print(json.dumps(output, indent=2))
        sys.exit(0)

    # Step 2: Check if code-only changes
    print("Step 2: Analyzing change types...", file=sys.stderr)
    code_exts = {'.py','.ts','.js','.go','.rs','.java','.cpp','.c','.rb','.swift','.kt','.cs','.scala','.php','.cc','.cxx','.hpp','.h','.kts','.lua','.toc'}
    new_files = result.get('new_files', {})
    all_changed = [f for files in new_files.values() for f in files]
    code_only = all(Path(f).suffix.lower() in code_exts for f in all_changed)

    output['steps']['change_analysis'] = {
        'code_only': code_only,
        'changed_files': all_changed[:10]
    }

    # Step 3: Extract changed files (code AST only if code-only)
    print("Step 3: Extracting changed files...", file=sys.stderr)
    code_files = []
    for f in all_changed:
        if Path(f).suffix.lower() in code_exts:
            code_files.extend(collect_files(Path(f)) if Path(f).is_dir() else [Path(f)])

    if code_files:
        extraction = extract(code_files)
        nodes_extracted = len(extraction.get('nodes', []))
        edges_extracted = len(extraction.get('edges', []))
    else:
        extraction = {'nodes': [], 'edges': [], 'hyperedges': [], 'input_tokens': 0, 'output_tokens': 0}
        nodes_extracted = 0
        edges_extracted = 0

    output['steps']['extraction'] = {
        'nodes': nodes_extracted,
        'edges': edges_extracted,
        'input_tokens': extraction.get('input_tokens', 0),
        'output_tokens': extraction.get('output_tokens', 0)
    }

    # Step 4: Load existing graph and merge
    print("Step 4: Merging with existing graph...", file=sys.stderr)
    existing_data = json.loads(Path('graphify-out/graph.json').read_text())
    G_existing = json_graph.node_link_graph(existing_data, edges='links')
    G_new = build_from_json(extraction)
    G_existing.update(G_new)

    # Step 5: Cluster and analyze
    print("Step 5: Clustering and analysis...", file=sys.stderr)
    communities = cluster(G_existing)
    cohesion = score_all(G_existing, communities)
    gods = god_nodes(G_existing)
    surprises = surprising_connections(G_existing, communities)

    # Step 6: Generate outputs
    print("Step 6: Generating outputs...", file=sys.stderr)
    detection = json.loads(Path('.graphify_detect.json').read_text()) if Path('.graphify_detect.json').exists() else {'total_files': files_total, 'total_words': words_total, 'files': {}}
    detection['total_files'] = files_total
    detection['total_words'] = words_total

    tokens = {
        'input': extraction.get('input_tokens', 0),
        'output': extraction.get('output_tokens', 0)
    }

    labels = {cid: f'Community {cid}' for cid in communities}
    report = generate(G_existing, communities, cohesion, labels, gods, surprises, detection, tokens, '.', suggested_questions=[])

    Path('graphify-out/GRAPH_REPORT.md').write_text(report)
    to_json(G_existing, communities, 'graphify-out/graph.json')

    # Final report
    output['status'] = 'success'
    output['steps']['final'] = {
        'nodes_total': G_existing.number_of_nodes(),
        'edges_total': G_existing.number_of_edges(),
        'communities': len(communities),
        'god_nodes': len(gods),
        'surprises': len(surprises)
    }

    output['report_sections'] = {
        'god_nodes': gods[:5] if gods else [],
        'surprising_connections': surprises[:5] if surprises else []
    }

except Exception as e:
    output['status'] = 'error'
    output['error'] = str(e)
    import traceback
    output['traceback'] = traceback.format_exc()

print(json.dumps(output, indent=2))
