from langchain_ollama import OllamaEmbeddings

print("Iniciando...")

embeddings = OllamaEmbeddings(
    model="bge-m3"
)

print("Generando embedding...")

resultado = embeddings.embed_query(
    "hola mundo"
)

print("OK")
print(len(resultado))