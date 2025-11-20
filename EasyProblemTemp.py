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
def solve_min_cost_flow(graph: PowerGridGraph, verbose=False, gen_cost_coeff=1e-6):
    """
    Solve min-cost flow where local generation at each node is a decision variable:
      0 <= g_n <= node.supply

    Objective: minimize sum(edge_length * flow_on_edge) + gen_cost_coeff * sum(g_n)
    - gen_cost_coeff is small by default to discourage unnecessary generation if transmission
      cost would otherwise be reduced by producing locally. Set to 0 to make generation
      costless.

    Returns (flows_dict, gen_dict, objective_value) or None if infeasible/non-optimal.
    """
    edges = graph.get_edge_list()
    prob = pulp.LpProblem("MinCostFlow_with_Generation", pulp.LpMinimize)

    # 1) Flow vars per directed edge (nonnegative)
    flow_vars = {}
    for (u, v, length) in edges:
        var_name = f"f_{u}_{v}"
        flow_vars[(u, v)] = pulp.LpVariable(var_name, lowBound=0)

    # 2) Generation vars per node (0 .. node.supply)
    gen_vars = {}
    for nid, node in graph.nodes.items():
        # Only create a generator variable if node.supply > 0 (you may still create for all nodes)
        # but creating for all nodes is fine; cap is 0 for nodes with no capacity.
        up = float(node.supply) if node.supply is not None else 0.0
        gen_vars[nid] = pulp.LpVariable(f"g_{nid}", lowBound=0, upBound=up)

    # 3) Objective: transmission cost + (small) generation cost
    trans_cost_term = pulp.lpSum([length * flow_vars[(u, v)] for (u, v, length) in edges])
    gen_cost_term = gen_cost_coeff * pulp.lpSum([gen_vars[nid] for nid in gen_vars])
    prob += trans_cost_term + gen_cost_term, "Total_Cost"

    # 4) Flow conservation at each node:
    #    sum_in - sum_out + g_n - demand == 0
    # Note: This expects node.demand to be a positive consumption value.
    for nid, node in graph.nodes.items():
        inflow_vars = [flow_vars[(u, v)] for (u, v, _l) in edges if v == nid]
        outflow_vars = [flow_vars[(u, v)] for (u, v, _l) in edges if u == nid]
        demand_val = float(node.demand)
        prob += (pulp.lpSum(inflow_vars) - pulp.lpSum(outflow_vars) + gen_vars[nid] + demand_val == 0), f"flow_cons_node_{nid}"

    # 5) Solve
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0)
    result = prob.solve(solver)

    status = pulp.LpStatus.get(prob.status, "Unknown")
    if verbose:
        print("Solver status:", status)
    if status != "Optimal":
        print(f"LP did not find an optimal solution. Status: {status}")
        return None

    # 6) Collect results
    flows = { (u,v): float(pulp.value(flow_vars[(u,v)])) for (u, v, _l) in edges }
    gens = { nid: float(pulp.value(gen_vars[nid])) for nid in gen_vars }
    objective_value = float(pulp.value(prob.objective))

    return flows, gens, objective_value

def main():
    demand_file = "dataset/Gen_WI_Demand_Values.csv"
    supply_file = "dataset/Gen_WI_Supply_Values.csv"
    lines_file = "dataset/Gen_WI_Lines.csv"

    graph = PowerGridGraph()
    ok1 = graph.load_demand_data(demand_file)
    ok2 = graph.load_supply_data(supply_file)
    ok3 = graph.load_lines_data(lines_file)

    if not (ok1 and ok2 and ok3):
        print("One or more datasets failed to load. Aborting.")
        return

    print("\nLoaded graph summary:")
    graph.display_graph()

    print("\nSolving minimum-cost flow LP...")
    result = solve_min_cost_flow(graph, verbose=True, gen_cost_coeff=1e-4)  # tweak coeff as needed
    if result is None:
        print("No feasible/optimal solution.")
        return
    flows, gens, obj = result

    print("\nOptimal objective value (total cost):", obj)
    print("\nEdge flows (nonzero shown):")
    for (u, v), f in sorted(flows.items(), key=lambda x: (-abs(x[1]), x[0])):
        if abs(f) > 1e-9:
            print(f"  {u} -> {v} : {f:.6f}")
    print("\nGenerator outputs (g_n):")
    for nid, g in sorted(gens.items(), key=lambda x: -x[1]):
        if g > 1e-9:
            print(f"  Node {nid} generates: {g:.6f} (cap: {graph.nodes[nid].supply})")

if __name__ == "__main__":
    main()
