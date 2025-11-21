# pyomo_miqcqp_mindtpy.py
import pandas as pd
import numpy as np
from pyomo.environ import (
    ConcreteModel, Var, Set, Param, NonNegativeReals, Binary, Reals,
    Constraint, Objective, minimize, SolverFactory, value
)
from pyomo.network import Port

# --- Paths (adjust if needed) ---
supply_file = "../dataset/Gen_WI_Supply_Values.csv"
demand_file = "../dataset/Gen_WI_Demand_Values.csv"
lines_file = "../dataset/Gen_WI_Lines.csv"

supply_scale = 100.0  # Adjust if needed
r_per_length = 0.5    # Estimated resistance per unit length
max_flow = 1e9        # large M for linking with binaries if needed

# --- Load data ---
df_supply = pd.read_csv(supply_file)
df_demand = pd.read_csv(demand_file)
df_lines = pd.read_csv(lines_file)

supply = dict(zip(df_supply.iloc[:, 0].astype(int), df_supply.iloc[:, 1].astype(float)))
demand = dict(zip(df_demand.iloc[:, 0].astype(int), df_demand.iloc[:, 1].astype(float)))

# columns as in your original code (assumes 1:from, 2:to, 3:length)
from_col = df_lines.columns[1]
to_col   = df_lines.columns[2]
len_col  = df_lines.columns[3]

edges = []
for _, r in df_lines.iterrows():
    u = int(r[from_col])
    v = int(r[to_col])
    length = float(r[len_col])
    edges.append((u, v, length))

# nodes (include supply/demand nodes even if isolated)
nodes = sorted(set([u for u, _, _ in edges] + [v for _, v, _ in edges] +
                   list(supply.keys()) + list(demand.keys())))

# index mapping for nodes if you want
node_idx = {n: i for i, n in enumerate(nodes)}

# --- Build Pyomo model ---
model = ConcreteModel(name="miqcqp_flows")

# Index sets
E = list(range(len(edges)))
N = list(nodes)

model.EDGES = Set(initialize=E)
model.NODES = Set(initialize=N)

# Parameters for edges: from, to, length, cost (= r_per_length * length)
edge_from = {i: edges[i][0] for i in E}
edge_to   = {i: edges[i][1] for i in E}
edge_len  = {i: edges[i][2] for i in E}
edge_cost = {i: r_per_length * edges[i][2] for i in E}

model.u = Param(model.EDGES, initialize=edge_from, within=Reals)
model.v = Param(model.EDGES, initialize=edge_to, within=Reals)
model.length = Param(model.EDGES, initialize=edge_len, within=Reals)
model.cost = Param(model.EDGES, initialize=edge_cost, within=Reals)

# Supply and demand param for nodes (supply is maximum generator capacity)
supply_vals = {n: float(supply.get(n, 0.0)) for n in N}
demand_vals = {n: float(demand.get(n, 0.0)) for n in N}
model.max_supply = Param(model.NODES, initialize=supply_vals, within=Reals)
model.demand = Param(model.NODES, initialize=demand_vals, within=Reals)

# --- Variables ---
# flows in positive/negative direction per directed edge (non-neg)
model.f_pos = Var(model.EDGES, domain=NonNegativeReals)
model.f_neg = Var(model.EDGES, domain=NonNegativeReals)

# heatLoss variables (non-neg)
model.heat_pos = Var(model.EDGES, domain=NonNegativeReals)
model.heat_neg = Var(model.EDGES, domain=NonNegativeReals)

# binary direction variable per edge
model.y = Var(model.EDGES, domain=Binary)

# generation per node (continuous non-negative)
model.gen = Var(model.NODES, domain=NonNegativeReals)

# --- Constraints ---

# Quadratic (nonlinear) heat constraints: heat >= cost * f^2
def heat_pos_rule(m, e):
    return m.heat_pos[e] >= m.cost[e] * (m.f_pos[e]**2)
model.heat_pos_con = Constraint(model.EDGES, rule=heat_pos_rule)

def heat_neg_rule(m, e):
    return m.heat_neg[e] >= m.cost[e] * (m.f_neg[e]**2)
model.heat_neg_con = Constraint(model.EDGES, rule=heat_neg_rule)

# Big-M linking to avoid simultaneous positive flows both directions (enforce one direction)
# f_pos <= M * y
def fpos_link_rule(m, e):
    return m.f_pos[e] <= max_flow * m.y[e]
model.fpos_link = Constraint(model.EDGES, rule=fpos_link_rule)

# f_neg <= M * (1 - y)
def fneg_link_rule(m, e):
    return m.f_neg[e] <= max_flow * (1 - m.y[e])
model.fneg_link = Constraint(model.EDGES, rule=fneg_link_rule)

# Generation bounds: 0 <= gen <= max_supply * supply_scale
def gen_upper_rule(m, n):
    return m.gen[n] <= m.max_supply[n] * supply_scale
model.gen_up = Constraint(model.NODES, rule=gen_upper_rule)

# Node balance constraints (same structure as your CVXPY code)
# gen[n] + flow_in - flow_out - heatLoss_in == demand[n]
def node_balance_rule(m, n):
    # incoming edges: edges where v==n
    in_edges = [e for e in E if int(m.v[e]) == int(n)]
    # outgoing edges: edges where u==n
    out_edges = [e for e in E if int(m.u[e]) == int(n)]

    flow_in = sum(m.f_pos[e] for e in in_edges) if in_edges else 0
    flow_in += sum(m.f_neg[e] for e in out_edges) if out_edges else 0

    flow_out = sum(m.f_pos[e] for e in out_edges) if out_edges else 0
    flow_out += sum(m.f_neg[e] for e in in_edges) if in_edges else 0

    heat_in = sum(m.heat_pos[e] for e in in_edges) if in_edges else 0
    heat_in += sum(m.heat_neg[e] for e in out_edges) if out_edges else 0

    return m.gen[n] + flow_in - flow_out - heat_in == m.demand[n]

model.node_balance = Constraint(model.NODES, rule=node_balance_rule)

# Objective: minimize sum of heat losses (both directions)
def obj_rule(m):
    return sum(m.heat_pos[e] + m.heat_neg[e] for e in m.EDGES)
model.obj = Objective(rule=obj_rule, sense=minimize)

# --- Solve with MindtPy (free) using IPOPT (NLP) + CBC (MILP) ---
# Requires ipopt and coin-or-cbc on PATH
mindtpy_solver = SolverFactory('mindtpy')
# Algorithm strategy 'OA' (outer-approximation) is common for MINLP / MIQCP
solve_options = {
    'strategy': 'OA',
    'mip_solver': 'cbc',     # free MILP solver
    'nlp_solver': 'ipopt',   # free NLP solver
    # additional options can be passed, e.g. time limits or tolerances:
    # 'time_limit': 600,
    # 'nlp_solver_args': {'max_iter': 1000}
}

print("Solving with MindtPy (OA) using ipopt + cbc ...")
try:
    results = mindtpy_solver.solve(model, **solve_options)
    print("MindtPy solve finished")
except Exception as e:
    print("MindtPy failed or not available:", e)
    # Try Bonmin (if available) as a fallback
    try:
        print("Trying bonmin (if installed)...")
        bonmin = SolverFactory('bonmin')
        results = bonmin.solve(model, tee=True)
    except Exception as e2:
        print("Bonmin also failed / not installed:", e2)
        raise SystemExit("No suitable free MINLP solver found on PATH. Install ipopt+cbc or bonmin/couenne.")

# --- Report results and save CSVs (guarding values) ---
print("Solver status:", results.solver.status, results.solver.termination_condition)

# Extract flows as signed = f_pos - f_neg
rows = []
for e in E:
    u = int(value(model.u[e]))
    v = int(value(model.v[e]))
    length = float(value(model.length[e]))
    cost = float(value(model.cost[e]))
    fp = float(value(model.f_pos[e]) if value(model.f_pos[e]) is not None else 0.0)
    fn = float(value(model.f_neg[e]) if value(model.f_neg[e]) is not None else 0.0)
    signed = fp - fn
    if signed >= 0:
        rows.append({
            "from": u,
            "to": v,
            "length": length,
            "flow": signed,
            "flow_squared": signed*signed,
            "cost_times_flow_squared": cost * (signed*signed)
        })
    else:
        rows.append({
            "from": v,
            "to": u,
            "length": length,
            "flow": -signed,
            "flow_squared": signed*signed,
            "cost_times_flow_squared": cost * (signed*signed)
        })

pd.DataFrame(rows).to_csv("min_flow_solution_pyomo.csv", index=False)
print("Saved min_flow_solution_pyomo.csv")

# Generation CSV
gen_rows = []
for n in N:
    gv = value(model.gen[n])
    gen_rows.append({"node": int(n), "generation": float(gv) if gv is not None else None, "max_supply": float(supply.get(n, 0.0))})
pd.DataFrame(gen_rows).to_csv("generation_results_pyomo.csv", index=False)
print("Saved generation_results_pyomo.csv")
