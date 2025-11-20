#!/usr/bin/env python3
"""
min_cost_flow_pulp.py

Reads:
  - Gen_WI_Supply_Values.csv   (node, supply)
  - Gen_WI_Demand_Values.csv   (node, demand)
  - Gen_WI_Lines.csv           (some id, from_node, to_node, length, ...)

Builds and solves a minimum-cost flow LP:
  minimize sum_{(i,j)} flow_{i,j} * length_{i,j}
  subject to:
    flow_{i,j} >= 0
    for every node k: sum_outflow(k) - sum_inflow(k) == supply_k - demand_k

If total supply != total demand, the script adds a dummy node to absorb the imbalance
(with zero-cost artificial edges) so the LP remains feasible; the solution will use
the dummy edges only as needed to balance totals.
Outputs:
  - /mnt/data/min_flow_solution.csv  (all edges and flows)
  - /mnt/data/min_flow_summary.txt   (status, total cost, nonzero flow count)
"""

import pandas as pd
import pulp
import os

# --- Paths (adjust if needed) ---
supply_file = "dataset/Gen_WI_Supply_Values.csv"
demand_file = "dataset/Gen_WI_Demand_Values.csv"
lines_file = "dataset/Gen_WI_Lines.csv"

# --- Load data ---
df_supply = pd.read_csv(supply_file)
df_demand = pd.read_csv(demand_file)
df_lines = pd.read_csv(lines_file)

# Expect columns: supply: [node, supply], demand: [node, demand]
supply = dict(zip(df_supply.iloc[:,0].astype(int), df_supply.iloc[:,1].astype(float)))
demand = dict(zip(df_demand.iloc[:,0].astype(int), df_demand.iloc[:,1].astype(float)))

# Assume lines are in columns: [something, from, to, length, ...]
from_col = df_lines.columns[1]
to_col   = df_lines.columns[2]
len_col  = df_lines.columns[3]

edges = []
for _, r in df_lines.iterrows():
    u = int(r[from_col])
    v = int(r[to_col])
    cost = float(r[len_col])
    edges.append((u, v, cost))

# All nodes appearing anywhere
nodes = sorted(set(list(supply.keys()) + list(demand.keys()) + [u for u,_,_ in edges] + [v for _,v,_ in edges]))

# --- LP model ---
prob = pulp.LpProblem("MinCostFlow", pulp.LpMinimize)

# flow variables for each directed edge
flow = {}
for (u,v,c) in edges:
    flow[(u,v)] = pulp.LpVariable(f"f_{u}_{v}", lowBound=0, cat="Continuous")

# objective
prob += pulp.lpSum(flow[(u,v)] * c for (u,v,c) in edges)

# node balances (out - in = supply - demand)
supply_vals = {n: supply.get(n, 0.0) for n in nodes}
demand_vals = {n: demand.get(n, 0.0) for n in nodes}

total_supply = sum(supply_vals.values())
total_demand = sum(demand_vals.values())
print("Total supply:", total_supply, "Total demand:", total_demand)

dummy_node = None
if abs(total_supply - total_demand) > 1e-9:
    # Create a special dummy node to absorb mismatch
    dummy_node = -999999
    nodes.append(dummy_node)
    print("Added dummy node to balance supply/demand. Imbalance:", total_supply - total_demand)

# add constraints for each real node
for n in nodes:
    if n == dummy_node:
        continue
    out_vars = [flow[(u,v)] for (u,v,c) in edges if u == n and (u,v) in flow]
    in_vars  = [flow[(u,v)] for (u,v,c) in edges if v == n and (u,v) in flow]
    prob += (pulp.lpSum(out_vars) - pulp.lpSum(in_vars) == supply_vals.get(n, 0.0) - demand_vals.get(n, 0.0)), f"node_balance_{n}"

# If dummy node exists, add zero-cost balancing edges between dummy and nodes that need absorbing
if dummy_node is not None:
    for n in list(nodes):
        if n == dummy_node:
            continue
        net = supply_vals.get(n,0.0) - demand_vals.get(n,0.0)
        if net > 0:
            # node has surplus -> can send to dummy
            flow[(n, dummy_node)] = pulp.LpVariable(f"f_{n}_{dummy_node}", lowBound=0, cat="Continuous")
            edges.append((n, dummy_node, 0.0))
        elif net < 0:
            # node has deficit -> can receive from dummy
            flow[(dummy_node, n)] = pulp.LpVariable(f"f_{dummy_node}_{n}", lowBound=0, cat="Continuous")
            edges.append((dummy_node, n, 0.0))
    # balance dummy to absorb net
    out_vars = [flow[(u,v)] for (u,v,c) in edges if u == dummy_node and (u,v) in flow]
    in_vars  = [flow[(u,v)] for (u,v,c) in edges if v == dummy_node and (u,v) in flow]
    prob += (pulp.lpSum(out_vars) - pulp.lpSum(in_vars) == 0.0), "node_balance_dummy"

# Rebuild objective to include added edges if any
prob.objective = pulp.lpSum(flow[(u,v)] * c for (u,v,c) in edges if (u,v) in flow)

# --- Solve ---
print("Solving LP ...")
solver = pulp.PULP_CBC_CMD(msg=1)  # default CBC
status = prob.solve(solver)
print("Status:", pulp.LpStatus[prob.status])

# --- Collect results ---
rows = []
for (u,v,c) in edges:
    if (u,v) in flow:
        val = pulp.value(flow[(u,v)])
        rows.append({"from": int(u), "to": int(v), "cost": float(c), "flow": float(val if val is not None else 0.0)})

df_res = pd.DataFrame(rows)
df_res.to_csv("min_flow_solution.csv", index=False)

total_cost = pulp.value(prob.objective)
nonzero = df_res[df_res["flow"] > 1e-9]
with open("min_flow_summary.txt","w") as f:
    f.write(f"Status: {pulp.LpStatus[prob.status]}\n")
    f.write(f"Total cost: {total_cost}\n")
    f.write(f"Nonzero flows: {len(nonzero)}\n")

print("Total cost:", total_cost)
print("Nonzero flow count:", len(nonzero))
print("Saved min_flow_solution.csv and min_flow_summary.txt.")
