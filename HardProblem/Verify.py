import pandas as pd

# ---- Load CSVs ----
df_lines = pd.read_csv("min_flow_solution_squared.csv")
df_supply = pd.read_csv("generation_results.csv")
df_demand = pd.read_csv("../dataset/GEN_WI_Demand_Values.csv")

# Rename columns for readability
df_lines.columns  = ['from', 'to', 'cost', 'flow', 'flow_squared', 'cost_times_flow_squared']
df_supply.columns = ['node', 'generation', 'max_supply']
df_demand.columns = ['node', 'demand']

# ---- Convert supply & demand into dictionaries ----
supply = dict(zip(df_supply['node'], df_supply['generation']))
max_supply = dict(zip(df_supply['node'], df_supply['max_supply']))
demand = dict(zip(df_demand['node'], df_demand['demand']))

# ---- Get list of all nodes ----
nodes = sorted(
    set(df_lines['from']).union(df_lines['to']).union(supply.keys()).union(demand.keys())
)

# Initialize trackers
results = []

for n in nodes:
    inflow = df_lines.loc[df_lines['to'] == n, 'flow'].sum()
    outflow = df_lines.loc[df_lines['from'] == n, 'flow'].sum()

    s = supply.get(n, 0)
    max_s = max_supply.get(n, float('inf'))
    d = demand.get(n, 0)

    lhs = inflow + s
    rhs = d + outflow

    flow_balance_ok = abs(lhs - rhs) <= 3e-2
    supply_ok = s <= max_s + 4e-3

    results.append({
        "node": n,
        "inflow": inflow,
        "supply": s,
        "max_supply": max_s,
        "demand": d,
        "outflow": outflow,
        "lhs = inflow + supply": lhs,
        "rhs = demand + outflow": rhs,
        "flow_balance_ok": flow_balance_ok,
        "supply_ok": supply_ok
    })

# ---- Convert results to DataFrame ----
df_check = pd.DataFrame(results)

# Print violations
violations = df_check[~df_check['flow_balance_ok'] | ~df_check['supply_ok']]
if len(violations) == 0:
    print("✔ All flow balance constraints and max supply limits satisfied.")
else:
    print("❌ Violations found:")
    print(violations)

# Save report
df_check.to_csv("flow_balance_check.csv", index=False)
print("\nFull report saved to: flow_balance_check.csv")
