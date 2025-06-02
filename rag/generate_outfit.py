from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import networkx as nx
import os
from dotenv import load_dotenv

load_dotenv()


def get_outfit_recommendation(G, query, num_items=3):
    # TODO: Filter compatible nodes
    options = list(G.nodes(data=True))[:num_items]

    item_descriptions = "\n".join([
        f"{item['type']}: {item['color']} {item['name']} ({item['style']})"
        for _, item in options
    ])

    prompt = f"""You are a personal stylist. Based on the following clothing items:\n{item_descriptions}\nSuggest a stylish outfit for the prompt: '{query}'."""

    response = client.chat.completions.create(model="gpt-4",
    messages=[{"role": "user", "content": prompt}])

    return response.choices[0].message.content
