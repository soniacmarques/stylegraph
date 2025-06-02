from openai import OpenAI
import networkx as nx
import os
from dotenv import load_dotenv
import ast

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_tags_from_query(query):
    #TODO: dinamically define occasion, style and warmth_level based on csv data
    tag_prompt = f"""
Extract wardrobe-related tags from the following outfit request:

Query: "{query}"

Respond with a JSON object using only these options:
- occasion: ['casual', 'party', 'business']
- style: ['girly', 'minimalist', 'cozy', 'classic', 'sporty', 'edgy', 'boho']
- warmth_level: ['light', 'medium', 'warm']

Each category should have a minimum of one option selected.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": tag_prompt}]
    )

    try:
        return ast.literal_eval(response.choices[0].message.content.strip())
    except:
        return {"occasion": None, "style": None, "warmth_level": None}
    

def find_seed_node(G, tags):
    best_score = -1
    best_node = None

    for n, d in G.nodes(data=True):
        score = sum(
            1 for k in tags if tags[k] and tags[k] == d.get(k)
        )
        if score > best_score:
            best_score = score
            best_node = n

    return best_node if best_node is not None else next(iter(G.nodes))


def get_outfit_recommendation(G, query):
    # Step 1: Extract tags from the query
    tags = extract_tags_from_query(query)
    print("Extracted Tags:", tags)

    # Step 2: Find a seed node based on tags
    #TODO: Debug this function - White T-shirt seems to always be the chosen seed item
    # going to the fallback as condition is not met by any item?
    seed = find_seed_node(G, tags)
    print("Seed Item:", G.nodes[seed]["name"])

    # Step 3: Get neighbors and build outfit
    neighbors = list(G.neighbors(seed))
    connected = [(seed, G.nodes[seed])] + [(n, G.nodes[n]) for n in neighbors]

    selected_options = []
    types_needed = {"top", "bottom", "layer", "shoes"}

    for n, d in connected:
        if d["type"] in types_needed:
            selected_options.append((n, d))
            types_needed.remove(d["type"])
        if not types_needed:
            break

    # Step 4: Format prompt for final outfit
    item_descriptions = "\n".join([
        f"{item['type']}: {item['color']} {item['name']} ({item['style']})"
        for _, item in connected #TODO: improve logic and then change 'connected' to 'selected_options'
    ])

    prompt = f"""You are a personal stylist. 
Based on the following clothing items:\n{item_descriptions}\n
Suggest a stylish outfit for the prompt: '{query}'.

Guidelines:
- Use only these items.
- Do not invent other clothing.
- You may suggest 1 accessory that complements the outfit.
- Be concise and explain the outfit's vibe.
"""

    print(f"\nPrompt sent to LLM:\n{prompt}\n")

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
