import os
from dotenv import load_dotenv
from graphs.build_graph import load_data, build_graph
from rag.generate_outfit import get_outfit_recommendation
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    df = load_data("data/clothing_items.csv")
    G = build_graph(df)
    query = "a casual summer brunch"
    outfit = get_outfit_recommendation(G, query)
    print("\n Suggested Outfit:\n", outfit)
