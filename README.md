# stylegraph
Using GraphRAG to manage a wardrobe 

Component - Tool
Graph - Python + NetworkX or Neo4j
Embeddings (for descriptions/tags) - OpenAI or sentence-transformers
LLM - OpenAI GPT or Hugging Face
Vector DB (for search-enhanced RAG) - FAISS or Chroma
Orchestration - LangChain
Frontend - Streamlit


1. data/clothing_items.csv ─┐
                            │
                            ▼
2. graphs/build_graph.py ──► Builds the graph from CSV

                            ▼
3. main.py ────────────────► Runs the full pipeline:
                                - Loads data
                                - Builds the graph
                                - Selects some items
                                - Calls the LLM

                            ▼
4. rag/generate_outfit.py ─► Uses selected items to prompt OpenAI
                                - Builds prompt
                                - Gets outfit suggestion from LLM
                                - Returns response

                            ▼
💡 Output = Printed outfit suggestion + explanation
