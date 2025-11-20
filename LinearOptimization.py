import pulp

def solve_linear_program(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    # Create a linear programming problem
    prob = pulp.LpProblem("Linear_Optimization_Problem", pulp.LpMinimize)

    # Create decision variables
    num_vars = len(c)
    x = [pulp.LpVariable(f"x{i}", lowBound=(bounds[i][0] if bounds else None), upBound=(bounds[i][1] if bounds else None)) for i in range(num_vars)]

    # Objective function
    prob += pulp.lpSum([c[i] * x[i] for i in range(num_vars)]), "Objective"

    # Inequality constraints
    if A_ub is not None and b_ub is not None:
        for i in range(len(b_ub)):
            prob += (pulp.lpSum([A_ub[i][j] * x[j] for j in range(num_vars)]) <= b_ub[i]), f"Inequality_Constraint_{i}"

    # Equality constraints
    if A_eq is not None and b_eq is not None:
        for i in range(len(b_eq)):
            prob += (pulp.lpSum([A_eq[i][j] * x[j] for j in range(num_vars)]) == b_eq[i]), f"Equality_Constraint_{i}"

    # Solve the problem
    prob.solve()

    # Extract results
    solution = [pulp.value(var) for var in x]
    objective_value = pulp.value(prob.objective)

    return solution, objective_value

if __name__ == "__main__":
    # Example usage
    c = [1, 2]
    A_ub = [[-1, 1], [3, 2]]
    b_ub = [1, 12]
    A_eq = [[1, 1]]
    b_eq = [7]
    bounds = [(0, None), (0, None)]

    solution, objective_value = solve_linear_program(c, A_ub, b_ub, A_eq, b_eq, bounds)
    print("Solution:", solution)
    print("Objective Value:", objective_value)