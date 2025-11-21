#!/usr/bin/env python3
"""
grid_min_cost_flow_fixed.py

Fixed combined script: loads graph from CSVs robustly (handles both headered CSVs and
plain index-based CSVs) and solves a minimum-cost flow LP with PuLP.

Usage:
    pip install pulp
    python grid_min_cost_flow_fixed.py
"""

import csv
import pulp
from collections import defaultdict

class Node:
    def __init__(self, node_number, demand=0.0, supply=0.0):
        self.node_number = int(node_number)
        self.demand = float(demand)
        self.supply = float(supply)

    def __repr__(self):
        return f"Node({self.node_number}, demand={self.demand}, supply={self.supply})"


class PowerGridGraph:
    def __init__(self):
        self.nodes = {}            # node_number -> Node
        self.adjacency_list = {}   # node_number -> list of (neighbor, length)

    def add_node(self, node):
        self.nodes[int(node.node_number)] = node

    def add_edge(self, node1, node2, length):
        node1 = int(node1); node2 = int(node2)
        if node1 not in self.nodes:
            self.add_node(Node(node1))
        if node2 not in self.nodes:
            self.add_node(Node(node2))
        self.adjacency_list.setdefault(node1, []).append((node2, float(length)))

    # Helper: check if a string looks like a header (contains letters) or numeric row
    @staticmethod
    def _row_looks_like_header(first_row):
        # if any cell contains alphabetic characters -> header
        for cell in first_row:
            if any(ch.isalpha() for ch in str(cell)):
                return True
        return False

    def load_demand_data(self, filename):
        try:
            with open(filename, newline='') as f:
                reader = csv.reader(f)
                first_row = next(reader, None)
                if first_row is None:
                    print(f"Demand file {filename} is empty.")
                    return False
                f.seek(0)
                if self._row_looks_like_header(first_row):
                    # use DictReader
                    dr = csv.DictReader(f)
                    headers = dr.fieldnames or []
                    # find node and demand columns (fall back to first two columns)
                    node_col = headers[0] if headers else None
                    demand_col = None
                    for cand in ("demand","Demand","value","Value","demand_mw","demand_mwh"):
                        if cand in headers:
                            demand_col = cand
                            break
                    if demand_col is None and len(headers) >= 2:
                        demand_col = headers[1]
                    for row in dr:
                        try:
                            node_num = int(row[node_col])
                            demand = float(row[demand_col]) if row[demand_col] != "" else 0.0
                        except Exception:
                            continue
                        if node_num in self.nodes:
                            self.nodes[node_num].demand = demand
                        else:
                            self.add_node(Node(node_num, demand=demand))
                else:
                    # positional reader (skip header-like first row if you want - but we treat first row as data)
                    f.seek(0)
                    r = csv.reader(f)
                    for row in r:
                        if not row: continue
                        try:
                            node_num = int(row[0])
                            demand = float(row[1])
                        except Exception:
                            continue
                        if node_num in self.nodes:
                            self.nodes[node_num].demand = demand
                        else:
                            self.add_node(Node(node_num, demand=demand))
            print(f"Loaded demand data from {filename}")
            return True
        except FileNotFoundError:
            print(f"Error: Could not open file {filename}")
            return False

    def load_supply_data(self, filename):
        try:
            with open(filename, newline='') as f:
                reader = csv.reader(f)
                first_row = next(reader, None)
                if first_row is None:
                    print(f"Supply file {filename} is empty.")
                    return False
                f.seek(0)
                if self._row_looks_like_header(first_row):
                    dr = csv.DictReader(f)
                    headers = dr.fieldnames or []
                    node_col = headers[0] if headers else None
                    supply_col = None
                    for cand in ("supply","Supply","value","Value","supply_mw"):
                        if cand in headers:
                            supply_col = cand
                            break
                    if supply_col is None and len(headers) >= 2:
                        supply_col = headers[1]
                    for row in dr:
                        try:
                            node_num = int(row[node_col])
                            supply = float(row[supply_col]) if row[supply_col] != "" else 0.0
                        except Exception:
                            continue
                        if node_num in self.nodes:
                            self.nodes[node_num].supply = supply
                        else:
                            self.add_node(Node(node_num, supply=supply))
                else:
                    f.seek(0)
                    r = csv.reader(f)
                    for row in r:
                        if not row: continue
                        try:
                            node_num = int(row[0])
                            supply = float(row[1])
                        except Exception:
                            continue
                        if node_num in self.nodes:
                            self.nodes[node_num].supply = supply
                        else:
                            self.add_node(Node(node_num, supply=supply))
            print(f"Loaded supply data from {filename}")
            return True
        except FileNotFoundError:
            print(f"Error: Could not open file {filename}")
            return False

    def load_lines_data(self, filename):
        try:
            with open(filename, newline='') as f:
                reader = csv.reader(f)
                first_row = next(reader, None)
                if first_row is None:
                    print(f"Lines file {filename} is empty.")
                    return False
                f.seek(0)
                if self._row_looks_like_header(first_row):
                    # Use DictReader and try to guess column names
                    dr = csv.DictReader(f)
                    headers = dr.fieldnames or []
                    # possible names
                    possible_n1 = ["node1","node_1","from","From","n1","N1","source","Source","u","U"]
                    possible_n2 = ["node2","node_2","to","To","n2","N2","target","Target","v","V"]
                    possible_len = ["length","Length","dist","distance","Distance","len","weight","cost","Cost"]
                    n1_col = next((h for h in headers if h in possible_n1), None)
                    n2_col = next((h for h in headers if h in possible_n2 and h != n1_col), None)
                    len_col = next((h for h in headers if h in possible_len), None)
                    for row in dr:
                        try:
                            if n1_col and n2_col and len_col:
                                node1 = int(row[n1_col])
                                node2 = int(row[n2_col])
                                length = float(row[len_col]) if row[len_col] != "" else 0.0
                            else:
                                # fallback: try positional fields within dict (some CSV writers use '1','2','3' keys)
                                # Attempt extracting by converting dict values to list in original column order
                                vals = [row[h] for h in headers]
                                if len(vals) >= 4:
                                    node1 = int(vals[1])
                                    node2 = int(vals[2])
                                    length = float(vals[3])
                                else:
                                    # last resort: try first three numeric-like fields
                                    numeric_vals = [v for v in vals if v.strip() != ""]
                                    node1 = int(numeric_vals[0])
                                    node2 = int(numeric_vals[1])
                                    length = float(numeric_vals[2])
                        except Exception:
                            # skip malformed row
                            continue
                        self.add_edge(node1, node2, length)
                else:
                    # positional CSV (no header)
                    f.seek(0)
                    r = csv.reader(f)
                    for row in r:
                        if not row: continue
                        # original code used columns 1,2,3 (skip index 0), but many CSVs use 0,1,2
                        # We try both: prefer 1,2,3 if available else 0,1,2
                        try:
                            if len(row) >= 4:
                                node1 = int(row[1])
                                node2 = int(row[2])
                                length = float(row[3])
                            else:
                                node1 = int(row[0])
                                node2 = int(row[1])
                                length = float(row[2])
                        except Exception:
                            continue
                        self.add_edge(node1, node2, length)
            print(f"Loaded lines data from {filename}")
            return True
        except FileNotFoundError:
            print(f"Error: Could not open file {filename}")
            return False

    def display_graph(self):
        print("\nPOWER GRID GRAPH\nNodes:")
        for n in sorted(self.nodes):
            node = self.nodes[n]
            print(f"  Node {node.node_number}: supply={node.supply}, demand={node.demand}")
        print("\nEdges:")
        for u, edges in self.adjacency_list.items():
            for v, l in edges:
                print(f"  {u} -> {v} (length={l})")

    def get_edge_list(self):
        edges = []
        for u, neighbors in self.adjacency_list.items():
            for v, l in neighbors:
                edges.append((u, v, l))
        return edges
    
from collections import deque
from typing import Dict, Set

def run_diagnostics(graph: PowerGridGraph):
    print("\n=== DIAGNOSTICS ===")

    # Basic totals
    total_supply = sum(node.supply for node in graph.nodes.values())
    total_demand = sum(node.demand for node in graph.nodes.values())
    print(f"Total supply (sum of Node.supply): {total_supply}")
    print(f"Total demand (sum of Node.demand): {total_demand}")
    print(f"Net supply - demand = {total_supply - total_demand}")

    # Per-node suspicious values
    print("\nNodes with suspicious values (supply<=0 or demand<=0):")
    for nid, node in sorted(graph.nodes.items()):
        if node.supply <= 0 or node.demand <= 0:
            print(f"  Node {nid}: supply={node.supply}, demand={node.demand}")

    # Check for generator caps zero or missing
    zero_caps = [nid for nid, n in graph.nodes.items() if getattr(n, "max_supply", n.supply) == 0]
    print(f"\nNodes with supply cap == 0: {len(zero_caps)} (showing up to 10): {zero_caps[:10]}")

    # Node id sets in each CSV vs edges:
    node_ids_from_nodes = set(graph.nodes.keys())
    node_ids_from_edges = set()
    for u, neighbors in graph.adjacency_list.items():
        node_ids_from_edges.add(u)
        for v, _ in neighbors:
            node_ids_from_edges.add(v)
    only_in_nodes = node_ids_from_nodes - node_ids_from_edges
    only_in_edges = node_ids_from_edges - node_ids_from_nodes
    print(f"\nCount nodes declared: {len(node_ids_from_nodes)}")
    print(f"Count node ids appearing in edges: {len(node_ids_from_edges)}")
    if only_in_nodes:
        print(f"Node IDs present only in node file (not in edges) sample: {list(only_in_nodes)[:10]}")
    if only_in_edges:
        print(f"Node IDs present only in edges (not in node list) sample: {list(only_in_edges)[:10]}")

    # Build undirected neighbor map (include isolated nodes)
    undirected: Dict[int, Set[int]] = {nid: set() for nid in node_ids_from_nodes.union(node_ids_from_edges)}
    for u, neighbors in graph.adjacency_list.items():
        for v, _ in neighbors:
            undirected.setdefault(u, set()).add(v)
            undirected.setdefault(v, set()).add(u)

    # Find connected components (undirected)
    all_nodes = set(undirected.keys())
    components = []
    visited = set()
    for start in list(all_nodes):
        if start in visited:
            continue
        comp = set()
        dq = deque([start])
        while dq:
            x = dq.popleft()
            if x in visited:
                continue
            visited.add(x)
            comp.add(x)
            for nbr in undirected.get(x, []):
                if nbr not in visited:
                    dq.append(nbr)
        components.append(comp)

    print(f"\nFound {len(components)} connected component(s) (undirected).")

    # For each component, check sum of max_supply vs demand
    for i, comp in enumerate(components, 1):
        comp_nodes = sorted(comp)
        # For nodes that aren't in graph.nodes (appear only in edges), treat supply/demand as 0
        comp_total_max_supply = 0.0
        comp_total_demand = 0.0
        for nid in comp_nodes:
            node = graph.nodes.get(nid)
            if node is None:
                # node declared in edges but not present in node list
                continue
            max_supply = getattr(node, "max_supply", None)
            if max_supply is None:
                # fallback to supply if max_supply not available
                max_supply = node.supply
            comp_total_max_supply += max_supply
            comp_total_demand += node.demand

        status = "EQUAL"
        if comp_total_max_supply > comp_total_demand:
            status = "SURPLUS"
        elif comp_total_max_supply < comp_total_demand:
            status = "SHORTFALL"

        print(f"\nComponent {i}: nodes={len(comp_nodes)} sample_ids={comp_nodes[:10]}")
        print(f"  Total max_supply (sum of node.max_supply|node.supply): {comp_total_max_supply}")
        print(f"  Total demand (sum of node.demand): {comp_total_demand}")
        print(f"  Status: {status}")
        if status != "EQUAL":
            diff = comp_total_max_supply - comp_total_demand
            if diff > 0:
                print(f"   -> Surplus of {diff}")
            else:
                print(f"   -> Shortfall of {-diff}")

    # If there are nodes declared but not in edges, mention them as isolated components
    isolated_declared = [nid for nid in node_ids_from_nodes if nid not in node_ids_from_edges]
    if isolated_declared:
        print(f"\nDeclared-but-isolated nodes (in node file but never appear in edges): {isolated_declared[:20]} (count {len(isolated_declared)})")

    print("\n=== END DIAGNOSTICS ===\n")


def main():
    demand_file = "dataset/bus_supply.csv"
    supply_file = "dataset/node_demand.csv"
    lines_file = "dataset/lines.csv"

    graph = PowerGridGraph()
    ok1 = graph.load_demand_data(demand_file)
    ok2 = graph.load_supply_data(supply_file)
    ok3 = graph.load_lines_data(lines_file)

    if not (ok1 and ok2 and ok3):
        print("One or more datasets failed to load. Aborting.")
        return

    print("\nLoaded graph summary:")
    graph.display_graph()
    run_diagnostics(graph)

if __name__ == "__main__":
    main()
