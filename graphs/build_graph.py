import pandas as pd
import networkx as nx

def load_data(filepath):
    return pd.read_csv(filepath)

def build_graph(df):
    G = nx.Graph()
    
    for _, row in df.iterrows():
        item_id = row["id"]
        attrs = row.to_dict()
        G.add_node(item_id, **attrs)
        
        # TODO: Elaborate this
        # Example compatibility logic (simple version)
        for other_id, other_attrs in G.nodes(data=True):
            if other_id != item_id:
                if row["occasion"] == other_attrs.get("occasion") and row["warmth_level"] == other_attrs.get("warmth_level"):
                    G.add_edge(item_id, other_id, reason="occasion & warmth_level match")
    
    return G

if __name__ == "__main__":
    df = load_data("/workspaces/stylegraph/data/clothing_items.csv")
    G = build_graph(df)
    print(f"Graph built with {G.number_of_nodes()} items and {G.number_of_edges()} connections.")
