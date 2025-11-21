import pandas as pd
import cvxpy as cp
import numpy as np

# --- Paths (adjust if needed) ---
supply_file = "../dataset/Gen_WI_Supply_Values.csv"
demand_file = "../dataset/Gen_WI_Demand_Values.csv"
lines_file = "../dataset/Gen_WI_Lines.csv"

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
flow = cp.Variable(num_edges, name="flow")

# Extract costs as numpy array
costs = np.array([c for _, _, c in edges])

# Objective: minimize sum of SQUARED flows weighted by costs
# This is convex since square function is convex
# Using cp.sum_squares or cp.square both work
objective = cp.Minimize(cp.sum(cp.multiply(costs, cp.square(flow))) * 0.5)

# Build constraints for node balance
constraints = []

# Supply and demand values for all nodes
supply_vals = {n: supply.get(n, 0.0) for n in nodes}
demand_vals = {n: demand.get(n, 0.0) for n in nodes}

# --- Add supply decision variables ---
gen = {n: cp.Variable(name=f"gen_{n}") for n in nodes}

# Supply upper bounds
for n in nodes:
    max_sup = supply_vals.get(n, 0.0)
    constraints.append(gen[n] >= 0)
    constraints.append(gen[n] <= max_sup)

for n in nodes:
    in_edges = [idx for idx, (u, v, c) in enumerate(edges) if v == n]
    out_edges = [idx for idx, (u, v, c) in enumerate(edges) if u == n]

    flow_in = cp.sum(flow[in_edges]) if in_edges else 0
    flow_out = cp.sum(flow[out_edges]) if out_edges else 0

    constraints.append(gen[n] + flow_in - flow_out == demand_vals.get(n, 0.0))

# --- Solve using CVXPY ---
print("\nSolving convex optimization problem with SQUARED flow objective...")
prob = cp.Problem(objective, constraints)

# Solve - quadratic problems work well with OSQP, ECOS, or SCS
try:
    prob.solve(
        solver=cp.OSQP,
        verbose=True,
        max_iter=300_000
    )
    print(f"Solved with OSQP")
except:
    try:
        print("OSQP failed, trying ECOS...")
        prob.solve(solver=cp.ECOS, verbose=True)
        print(f"Solved with ECOS")
    except:
        print("ECOS failed, trying SCS...")
        prob.solve(solver=cp.SCS, verbose=True)
        print(f"Solved with SCS")

print(f"\nStatus: {prob.status}")
print(f"Optimal value (sum of cost * flow²): {prob.value}")

# --- Collect results ---
if prob.status in ["optimal", "optimal_inaccurate"]:
    rows = []
    flow_values = flow.value
    
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
