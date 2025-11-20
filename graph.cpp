#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <iomanip>

using namespace std;

class Node {
public:
    int nodeNumber;
    double demand;
    double supply;
    
    Node() : nodeNumber(0), demand(0.0), supply(0.0) {}
    
    Node(int num, double dem = 0.0, double sup = 0.0) 
        : nodeNumber(num), demand(dem), supply(sup) {}
    
    void display() const {
        cout << "Node " << nodeNumber 
             << ": Demand=" << demand 
             << ", Supply=" << supply << endl;
    }
};

class PowerGridGraph {
private:
    map<int, Node> nodes;
    map<int, vector<pair<int, double>>> adjacencyList; // node -> [(neighbor, length)]
    
public:
    void addNode(const Node& node) {
        nodes[node.nodeNumber] = node;
    }
    
    void addEdge(int node1, int node2, double length) {
        // Ensure both nodes exist
        if (nodes.find(node1) == nodes.end()) {
            nodes[node1] = Node(node1);
        }
        if (nodes.find(node2) == nodes.end()) {
            nodes[node2] = Node(node2);
        }
        
        // Add directed edge from node1 to node2
        adjacencyList[node1].push_back(make_pair(node2, length));
    }
    
    bool loadDemandData(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filename << endl;
            return false;
        }
        
        string line;
        getline(file, line); // Skip header
        
        while (getline(file, line)) {
            stringstream ss(line);
            string nodeStr, demandStr;
            
            getline(ss, nodeStr, ',');
            getline(ss, demandStr, ',');
            
            int nodeNum = stoi(nodeStr);
            double demand = -stod(demandStr); // Store demand as negative
            
            if (nodes.find(nodeNum) != nodes.end()) {
                nodes[nodeNum].demand = demand;
            } else {
                Node node(nodeNum, demand, 0.0);
                addNode(node);
            }
        }
        
        file.close();
        cout << "Loaded demand data from " << filename << endl;
        return true;
    }
    
    bool loadSupplyData(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filename << endl;
            return false;
        }
        
        string line;
        getline(file, line); // Skip header
        
        while (getline(file, line)) {
            stringstream ss(line);
            string nodeStr, supplyStr;
            
            getline(ss, nodeStr, ',');
            getline(ss, supplyStr, ',');
            
            int nodeNum = stoi(nodeStr);
            double supply = stod(supplyStr);
            
            if (nodes.find(nodeNum) != nodes.end()) {
                nodes[nodeNum].supply = supply;
            } else {
                Node node(nodeNum, 0.0, supply);
                addNode(node);
            }
        }
        
        file.close();
        cout << "Loaded supply data from " << filename << endl;
        return true;
    }
    
    bool loadLinesData(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filename << endl;
            return false;
        }
        
        string line;
        getline(file, line); // Skip header
        
        while (getline(file, line)) {
            stringstream ss(line);
            string lineNumStr, node1Str, node2Str, lengthStr;
            
            getline(ss, lineNumStr, ',');
            getline(ss, node1Str, ',');
            getline(ss, node2Str, ',');
            getline(ss, lengthStr, ',');
            
            int node1 = stoi(node1Str);
            int node2 = stoi(node2Str);
            double length = stod(lengthStr);
            
            addEdge(node1, node2, length);
        }
        
        file.close();
        cout << "Loaded lines data from " << filename << endl;
        return true;
    }
    
    void displayGraph() const {
        cout << "\nPOWER GRID GRAPH\n";
        cout << "\nNodes:\n";
        cout << "--------------------------------------\n";
        for (const auto& pair : nodes) {
            pair.second.display();
        }
        
        cout << "\nAdjacency List (Connections):\n";
        cout << "--------------------------------------\n";
        for (const auto& pair : adjacencyList) {
            cout << "Node " << pair.first << " connects to: ";
            for (const auto& edge : pair.second) {
                cout << "[Node " << edge.first << ", Length=" << edge.second << "] ";
            }
            cout << endl;
        }
    }
    
    int getNodeCount() const {
        return nodes.size();
    }
    
    int getEdgeCount() const {
        int count = 0;
        for (const auto& pair : adjacencyList) {
            count += pair.second.size();
        }
        return count; // Each directed edge is counted once
    }
    
    Node* getNode(int nodeNumber) {
        if (nodes.find(nodeNumber) != nodes.end()) {
            return &nodes[nodeNumber];
        }
        return nullptr;
    }
    
    vector<pair<int, double>> getNeighbors(int nodeNumber) {
        if (adjacencyList.find(nodeNumber) != adjacencyList.end()) {
            return adjacencyList[nodeNumber];
        }
        return vector<pair<int, double>>();
    }
};

int main() {
    PowerGridGraph graph;
    
    cout << "Building Power Grid Graph...\n" << endl;
    
    // Load the three CSV files
    graph.loadDemandData("dataset/Gen_WI_Demand_Values.csv");
    graph.loadSupplyData("dataset/Gen_WI_Supply_Values.csv");
    graph.loadLinesData("dataset/Gen_WI_Lines.csv");
    
    // Display the graph
    graph.displayGraph();
    
    // Display summary statistics
    cout << "\n========== GRAPH STATISTICS ==========\n";
    cout << "Total Nodes: " << graph.getNodeCount() << endl;
    cout << "Total Edges: " << graph.getEdgeCount() << endl;
    
    // Example: Access a specific node
    cout << "\n========== EXAMPLE USAGE ==========\n";
    Node* node = graph.getNode(12075);
    if (node != nullptr) {
        cout << "Accessing Node 12075:\n";
        node->display();
        
        cout << "\nNeighbors of Node 12075:\n";
        vector<pair<int, double>> neighbors = graph.getNeighbors(12075);
        for (const auto& neighbor : neighbors) {
            cout << "  -> Node " << neighbor.first 
                 << " (Distance: " << neighbor.second << ")\n";
        }
    }
    
    return 0;
}