import pandas as pd
import cvxpy as cp
import numpy as np

# --- Paths (adjust if needed) ---
supply_file = "../dataset/Gen_WI_Supply_Values.csv"
demand_file = "../dataset/Gen_WI_Demand_Values.csv"
lines_file = "../dataset/Gen_WI_Lines.csv"

supply_scale = 100.0  # Adjust if needed
r_per_length = 0.5 # Estimated resistance per unit length
max_flow = 1e9  # Arbitrary large number for flow upper bounds
mySolver = cp.SCIP  # Preferred solver

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

# --- CVXPY Convex Optimization Model ---

# Create flow variables for each directed edge
num_edges = len(edges)
f_pos = cp.Variable(num_edges, nonneg=True)
f_neg = cp.Variable(num_edges, nonneg=True)
heatLoss_pos = cp.Variable(num_edges, nonneg=True)
heatLoss_neg = cp.Variable(num_edges, nonneg=True)
y = cp.Variable(num_edges, boolean=True)

# Extract costs as numpy array
costs = np.array([r_per_length * c for _, _, c in edges])

# Objective: minimize sum of SQUARED flows weighted by costs
# This is convex since square function is convex
# Using cp.sum_squares or cp.square both work
objective = cp.Minimize(cp.sum(heatLoss_pos + heatLoss_neg))

# Build constraints for node balance
constraints = []

for i in range(num_edges):
    constraints += [
        heatLoss_pos[i] >= costs[i] * cp.square(f_pos[i]),
        heatLoss_neg[i] >= costs[i] * cp.square(f_neg[i]),
        f_pos[i] <= max_flow * y[i],
        f_neg[i] <= max_flow * (1 - y[i])
    ]

# Supply and demand values for all nodes
supply_vals = {n: supply.get(n, 0.0) for n in nodes}
demand_vals = {n: demand.get(n, 0.0) for n in nodes}

# --- Add supply decision variables ---
gen = {n: cp.Variable(name=f"gen_{n}") for n in nodes}

# Supply upper bounds
for n in nodes:
    max_sup = supply_vals.get(n, 0.0)
    constraints.append(gen[n] >= 0)
    constraints.append(gen[n] <= max_sup*supply_scale)

for n in nodes:
    in_edges = [idx for idx, (u, v, c) in enumerate(edges) if v == n]
    out_edges = [idx for idx, (u, v, c) in enumerate(edges) if u == n]

    flow_in = cp.sum(f_pos[in_edges]) if in_edges else 0
    flow_in += cp.sum(f_neg[out_edges]) if out_edges else 0
    flow_out = cp.sum(f_pos[out_edges]) if out_edges else 0
    flow_out += cp.sum(f_neg[in_edges]) if in_edges else 0

    heatLoss_in = cp.sum(heatLoss_pos[in_edges]) if in_edges else 0
    heatLoss_in += cp.sum(heatLoss_neg[out_edges]) if out_edges else 0
    
    constraints.append(gen[n] + flow_in - flow_out - heatLoss_in == demand_vals.get(n, 0.0))

# --- Solve using CVXPY ---
print("\nSolving convex optimization problem with SQUARED flow objective...")
prob = cp.Problem(objective, constraints)

# Solve - quadratic problems work well with OSQP, ECOS, or SCS
try:
    prob.solve(
        solver=mySolver,
        verbose=True,
        max_iter=300_000
    )
    print(f"Solved with {mySolver}")
except:
    print(f"Failed to solve with {mySolver}, trying default solver...")
    prob.solve(verbose=True, max_iter=300_000)


print(f"\nStatus: {prob.status}")
print(f"Optimal value (sum of cost * flow²): {prob.value}")

# --- Collect results ---
if prob.status in ["optimal", "optimal_inaccurate"]:
    rows = []
    flow_values = f_pos.value - f_neg.value
    
    for idx, (u, v, c) in enumerate(edges):
        val = flow_values[idx]
        
        # Store flow in the direction it actually flows
        if val >= 0:
            rows.append({
                "from": int(u), 
                "to": int(v), 
                "cost": abs(float(c)), 
                "flow": float(val),
                "flow_squared": float(val * val),
                "cost_times_flow_squared": float(c * val * val)
            })
        else:
            rows.append({
                "from": int(v), 
                "to": int(u), 
                "cost": abs(float(c)), 
                "flow": float(-val),
                "flow_squared": float(val * val),
                "cost_times_flow_squared": float(c * val * val)
            })
    
    df_res = pd.DataFrame(rows)
    df_res.to_csv("min_flow_solution_squared.csv", index=False)
    
    # Write summary
    with open("min_flow_summary_squared.txt", "w") as f:
        f.write(f"Status: {prob.status}\n")
        f.write(f"Objective: Minimize sum of (cost * flow^2)\n")
        f.write(f"Total cost: {prob.value}\n")
        f.write(f"Solver: CVXPY\n")
    
    print(f"\nTotal cost (sum of cost * flow^2): {prob.value}")
    print("Saved min_flow_solution_squared.csv and min_flow_summary_squared.txt.")
else:
    print(f"Optimization failed with status: {prob.status}")

# --- Save generation results ---
gen_rows = []
for n in nodes:
    gen_rows.append({
        "node": n,
        "generation": float(gen[n].value),
        "max_supply": float(supply_vals.get(n, 0.0))
    })

df_gen = pd.DataFrame(gen_rows)
df_gen.to_csv("generation_results.csv", index=False)
print("Saved generation_results.csv")
