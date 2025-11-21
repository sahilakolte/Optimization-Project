import pandas as pd
import pulp

# --- Paths (adjust if needed) ---
supply_file = "../dataset/Gen_WI_Supply_Values.csv"
demand_file = "../dataset/Gen_WI_Demand_Values.csv"
lines_file  = "../dataset/Gen_WI_Lines.csv"

# --- Load data ---
df_supply = pd.read_csv(supply_file)
df_demand = pd.read_csv(demand_file)
df_lines  = pd.read_csv(lines_file)

# Expect: supply.csv  -> [node, max_supply]
#         demand.csv  -> [node, demand]
max_supply = dict(zip(df_supply.iloc[:,0].astype(int),
                      df_supply.iloc[:,1].astype(float)))
demand = dict(zip(df_demand.iloc[:,0].astype(int),
                  df_demand.iloc[:,1].astype(float)))

# Lines assumed: [id, from, to, length, ...]
from_col = df_lines.columns[1]
to_col   = df_lines.columns[2]
len_col  = df_lines.columns[3]

edges = []
for _, r in df_lines.iterrows():
    u = int(r[from_col])
    v = int(r[to_col])
    cost = float(r[len_col])
    edges.append((u, v, cost))

# All nodes in the graph
nodes = sorted(set([u for u,_,_ in edges] + [v for _,v,_ in edges]))

# --- LP model ---
prob = pulp.LpProblem("MinCostFlow", pulp.LpMinimize)

# --- Flow variables for each directed edge
flow = {}
for idx, (u, v, c) in enumerate(edges):
    flow[(u, v, idx)] = pulp.LpVariable(f"f_{u}_{v}_{idx}",
                                        lowBound=None, cat="Continuous")

# --- Absolute value variables for flow
abs_flow = {}
for idx, (u, v, c) in enumerate(edges):
    f = flow[(u, v, idx)]
    abs_f = pulp.LpVariable(f"abs_f_{u}_{v}_{idx}", lowBound=0)
    abs_flow[(u, v, idx)] = abs_f

    # abs_f >= ±f
    prob += abs_f >=  f
    prob += abs_f >= -f

# --- Generation decision variables ---
# gen[n] ∈ [0, max_supply[n]]
gen = {}
for n in nodes:
    gen[n] = pulp.LpVariable(
        f"gen_{n}",
        lowBound=0,
        upBound=max_supply.get(n, 0.0),
        cat="Continuous"
    )

# --- Objective: minimize cost * |flow|
prob += pulp.lpSum(abs_flow[(u, v, idx)] * abs(float(c))
                   for idx, (u, v, c) in enumerate(edges))

# --- Node balance constraints ---
# generation + inflow – outflow = demand
for n in nodes:
    outVars = [flow[(u, v, idx)]
               for idx, (u, v, c) in enumerate(edges) if u == n]
    inVars  = [flow[(u, v, idx)]
               for idx, (u, v, c) in enumerate(edges) if v == n]

    prob += (
        gen[n] + pulp.lpSum(inVars) - pulp.lpSum(outVars)
        == demand.get(n, 0.0)
    ), f"node_balance_{n}"

# --- Solve the LP ---
print("Solving LP ...")
solver = pulp.PULP_CBC_CMD(msg=1)
status = prob.solve(solver)
print("Status:", pulp.LpStatus[prob.status])

# --- Collect results ---
rows = []
for idx, (u, v, c) in enumerate(edges):
    val = pulp.value(flow[(u, v, idx)])
    if val is None:
        val = 0.0

    # Export direction of actual flow
    if val >= 0:
        rows.append({"from": u, "to": v, "cost": abs(float(c)), "flow": float(val)})
    else:
        rows.append({"from": v, "to": u, "cost": abs(float(c)), "flow": float(-val)})

df_res = pd.DataFrame(rows)
df_res.to_csv("min_flow_solution.csv", index=False)

total_cost = pulp.value(prob.objective)
with open("min_flow_summary.txt", "w") as f:
    f.write(f"Status: {pulp.LpStatus[prob.status]}\n")
    f.write(f"Total cost: {total_cost}\n")

print("Total cost:", total_cost)
print("Saved min_flow_solution.csv and min_flow_summary.txt.")
