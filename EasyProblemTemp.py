import pandas as pd
import pulp

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
nodes = sorted(set([u for u,_,_ in edges] + [v for _,v,_ in edges]))

# --- LP model ---
prob = pulp.LpProblem("MinCostFlow", pulp.LpMinimize)

# flow variables for each directed edge
flow = {}
for idx, (u, v, c) in enumerate(edges):
    flow[(u, v, idx)] = pulp.LpVariable(f"f_{u}_{v}_{idx}", lowBound=None, cat="Continuous")

abs_flow = {}  # auxiliary vars

for idx, (u, v, c) in enumerate(edges):
    f = flow[(u, v, idx)]
    
    # abs(f) variable
    abs_f = pulp.LpVariable(f"abs_f_{u}_{v}_{idx}", lowBound=0)
    abs_flow[(u, v, idx)] = abs_f

    # enforce abs_f = |f|
    prob += abs_f >=  f
    prob += abs_f >= -f

# objective -- cost per undirected edge = (flow_uv + flow_vu) * c
prob += pulp.lpSum(abs_flow[(u, v, idx)] * c 
                   for idx, (u, v, c) in enumerate(edges))

# node balances (out - in = supply - demand)
supply_vals = {n: supply.get(n, 0.0) for n in nodes}
demand_vals = {n: demand.get(n, 0.0) for n in nodes}

total_supply = sum(supply_vals.values())
total_demand = sum(demand_vals.values())
print("Total supply:", total_supply, "Total demand:", total_demand)

# add constraints for each real node
for n in nodes:
    outVars = [flow[(u, v, idx)] for idx, (u, v, c) in enumerate(edges) if u == n]
    inVars = [flow[(u, v, idx)] for idx, (u, v, c) in enumerate(edges) if v == n]
    prob += (
        pulp.lpSum(inVars) - pulp.lpSum(outVars)
        <= supply_vals.get(n, 0.0) - demand_vals.get(n, 0.0)
    ), f"node_balance_{n}"

# --- Solve ---
print("Solving LP ...")
solver = pulp.PULP_CBC_CMD(msg=1)  # default CBC
status = prob.solve(solver)
print("Status:", pulp.LpStatus[prob.status])

# --- Collect results ---
rows = []
for idx, (u,v,c) in enumerate(edges):
    if (u,v, idx) in flow:
        val = pulp.value(flow[(u,v, idx)])

        if(val >= 0):
            rows.append({"from": int(u), "to": int(v), "cost": abs(float(c)), "flow": float(val if val is not None else 0.0)})
        else:
            rows.append({"from": int(v), "to": int(u), "cost": abs(float(c)), "flow": float(-val if val is not None else 0.0)})

df_res = pd.DataFrame(rows)
df_res.to_csv("min_flow_solution.csv", index=False)

total_cost = pulp.value(prob.objective)
with open("min_flow_summary.txt","w") as f:
    f.write(f"Status: {pulp.LpStatus[prob.status]}\n")
    f.write(f"Total cost: {total_cost}\n")

print("Total cost:", total_cost)
print("Saved min_flow_solution.csv and min_flow_summary.txt.")
