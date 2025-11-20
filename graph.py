import csv

class Node:
    def __init__(self, node_number, demand=0.0, supply=0.0):
        self.node_number = node_number
        self.demand = demand
        self.supply = supply

    def display(self):
        print(f"Node {self.node_number}: Demand={self.demand}, Supply={self.supply}")


class PowerGridGraph:
    def __init__(self):
        self.nodes = {}  # node_number -> Node object
        self.adjacency_list = {}  # node_number -> [(neighbor, length)]

    def add_node(self, node):
        self.nodes[node.node_number] = node

    def add_edge(self, node1, node2, length):
        # Ensure nodes exist
        if node1 not in self.nodes:
            self.add_node(Node(node1))
        if node2 not in self.nodes:
            self.add_node(Node(node2))

        # Add directed edge
        self.adjacency_list.setdefault(node1, []).append((node2, length))

    # -----------------------------
    # Load demand CSV
    # -----------------------------
    def load_demand_data(self, filename):
        try:
            with open(filename, "r") as file:
                reader = csv.reader(file)
                next(reader)  # skip header

                for row in reader:
                    node_num = int(row[0])
                    demand = -float(row[1])  # negative demand

                    if node_num in self.nodes:
                        self.nodes[node_num].demand = demand
                    else:
                        self.add_node(Node(node_num, demand=demand))

            print(f"Loaded demand data from {filename}")
            return True

        except FileNotFoundError:
            print(f"Error: Could not open file {filename}")
            return False

    # -----------------------------
    # Load supply CSV
    # -----------------------------
    def load_supply_data(self, filename):
        try:
            with open(filename, "r") as file:
                reader = csv.reader(file)
                next(reader)

                for row in reader:
                    node_num = int(row[0])
                    supply = float(row[1])

                    if node_num in self.nodes:
                        self.nodes[node_num].supply = supply
                    else:
                        self.add_node(Node(node_num, supply=supply))

            print(f"Loaded supply data from {filename}")
            return True

        except FileNotFoundError:
            print(f"Error: Could not open file {filename}")
            return False

    # -----------------------------
    # Load lines CSV
    # -----------------------------
    def load_lines_data(self, filename):
        try:
            with open(filename, "r") as file:
                reader = csv.reader(file)
                next(reader)

                for row in reader:
                    node1 = int(row[1])
                    node2 = int(row[2])
                    length = float(row[3])
                    self.add_edge(node1, node2, length)

            print(f"Loaded lines data from {filename}")
            return True

        except FileNotFoundError:
            print(f"Error: Could not open file {filename}")
            return False

    # -----------------------------
    def display_graph(self):
        print("\nPOWER GRID GRAPH\n")
        print("Nodes:")
        print("--------------------------------------")

        for node in self.nodes.values():
            node.display()

        print("\nAdjacency List (Connections):")
        print("--------------------------------------")

        for node, edges in self.adjacency_list.items():
            edge_info = ", ".join([f"(Node {n}, Length={l})" for n, l in edges])
            print(f"Node {node} connects to: {edge_info}")

    # -----------------------------
    def get_node_count(self):
        return len(self.nodes)

    def get_edge_count(self):
        return sum(len(edges) for edges in self.adjacency_list.values())

    def get_node(self, node_number):
        return self.nodes.get(node_number, None)

    def get_neighbors(self, node_number):
        return self.adjacency_list.get(node_number, [])


# ============================================================
# Main Equivalent
# ============================================================

if __name__ == "__main__":
    graph = PowerGridGraph()

    print("Building Power Grid Graph...\n")

    graph.load_demand_data("dataset/Gen_WI_Demand_Values.csv")
    graph.load_supply_data("dataset/Gen_WI_Supply_Values.csv")
    graph.load_lines_data("dataset/Gen_WI_Lines.csv")

    # Display graph
    graph.display_graph()

    # Stats
    print("\n========== GRAPH STATISTICS ==========")
    print("Total Nodes:", graph.get_node_count())
    print("Total Edges:", graph.get_edge_count())

    # Example usage
    print("\n========== EXAMPLE USAGE ==========")
    node = graph.get_node(12075)
    if node:
        print("Accessing Node 12075:")
        node.display()

        print("\nNeighbors of Node 12075:")
        neighbors = graph.get_neighbors(12075)
        for n, dist in neighbors:
            print(f"  -> Node {n} (Distance: {dist})")
