import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from langchain_community.graphs import Neo4jGraph

llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = llm.embeddings.create(
        input="What does Hallucination mean?",
        model="text-embedding-ada-002"
    )

embedding = response.data[0].embedding

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

# Run query
result = graph.query("""
CALL db.index.vector.queryNodes('chunkVector', 6, $embedding)
YIELD node, score
RETURN node.text, score
""", {"embedding": embedding})

# Display results
for row in result:
    print(row['node.text'], row['score'])
