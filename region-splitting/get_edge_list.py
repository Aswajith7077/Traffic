import networkx as nx
import sumolib

net = sumolib.net.readNet("sumo/simple.net.xml")

G = nx.DiGraph()

print("Edges:", (net.getEdges()))

for edge in net.getEdges():
    if edge.isSpecial():  # skip internal edges
        continue

    from_node = edge.getFromNode().getID()
    to_node = edge.getToNode().getID()

    G.add_edge(from_node, to_node, id=edge.getID(), weight=edge.getLaneNumber())

print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))
