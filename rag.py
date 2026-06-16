import os
from dotenv import load_dotenv

# ORIGINAL
# from langchain_community.document_loaders import DirectoryLoader

# ALTERNATIVA MÁS LIVIANA
from langchain_community.document_loaders import PyPDFDirectoryLoader

from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

print("📂 Cargando PDFs...")

# pdfs = DirectoryLoader(
    #"documentos",
    # glob="*.pdf"
# ).load()

loader = PyPDFDirectoryLoader("documentos")
pdfs = loader.load()

print(f"✅ PDFs cargados: {len(pdfs)}")

print("🤖 Descargando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "BAAI/bge-m3"
)

splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=1250,
    chunk_overlap=150
)

print("📦 Fragmentando documentos...")
print("PDFs cargados:", len(pdfs))

fragmentos = splitter.split_documents(pdfs)

print(f"✅ Chunks generados: {len(fragmentos)}")

print("🧠 Inicializando embeddings...")

embeddings = OllamaEmbeddings(
    model="bge-m3"
)

print("✅ Embeddings listos")

print("🗄️ Construyendo FAISS...")

# vector_store = FAISS.from_documents(
#     documents=fragmentos,
#     embedding=embeddings
# )

texts = [doc.page_content for doc in fragmentos]

embeddings_list = []

for i, text in enumerate(texts):
    print(f"🧠 Embedding {i+1}/{len(texts)}")

    emb = embeddings.embed_query(text)
    embeddings_list.append(emb)

vector_store = FAISS.from_embeddings(
    list(zip(texts, embeddings_list)),
    embedding=embeddings
)

print("✅ FAISS creado correctamente")

prompt = ChatPromptTemplate(
    [
        (
            "system",
            """
Responde usando exclusivamente el contenido que se incluye a continuación.

Genera una respuesta concisa.

Contexto:
{contexto}
"""
        ),
        (
            "human",
            "{query}"
        )
    ]
)

retriever = vector_store.as_retriever()

print("🚀 Cargando Gemma...")

modelo = OllamaLLM(
    model="gemma3:4b"
)

print("✅ Gemma cargado")

# TEST RÁPIDO
print("🧪 Probando Gemma...")

try:
    respuesta_test = modelo.invoke(
        "Responde únicamente OK"
    )

    print("Respuesta Gemma:")
    print(respuesta_test)

except Exception as e:
    print("❌ Error Gemma")
    print(e)

cadena = prompt | modelo | StrOutputParser()

pregunta = 'Cómo solicitar el seguro de viaje?'

print("\n🔎 Pregunta original:")
print(pregunta)

# ==========================
# BÚSQUEDA NORMAL
# ==========================

print("\n📚 Recuperando contexto...")

# modelo.invoke(pregunta)

trechos = retriever.invoke(
    pregunta
)

print(
    f"✅ Fragmentos recuperados: {len(trechos)}"
)

contexto = "\n\n".join(
    trecho.page_content
    for trecho in trechos
)

# cadena.invoke({"query": pregunta, "contexto":contexto})

# rag_chain = (
#     {"contexto": RunnablePassthrough() | retriever,
#      "query":RunnablePassthrough()}
#      | prompt | modelo | StrOutputParser()
# )

# rag_chain.invoke(pregunta)

query_model = OllamaLLM(
    model="gemma3:4b"
)

rewriter_prompt_template =  """
Genera la consulta de búsqueda para la base de datos de vectores (Vector DB) a partir de una pregunta del usuario, 
permitiendo una respuesta más precisa por medio de la búsqueda semántica. 
Basta devolver la consulta revisada del Vector DB, entre comillas.

# PREGUNTA DEL USUARIO: {user_question}
# CONSULTA REVISADA DEL VECTOR DB:
"""

rewriter_prompt = PromptTemplate.from_template(
    rewriter_prompt_template
)

rewriter_chain = (
    rewriter_prompt
    | query_model
    | StrOutputParser()
)

# ==========================
# DESCOMENTAR SI QUIERES PROBAR
# ==========================

# print("\n📝 Query Rewriting")
# print(
#     rewriter_chain.invoke(
#         {"user_question": pregunta}
#     )
# )

# ==========================
# MULTI QUERY RETRIEVAL
# ==========================

# rewriter_chain.invoke(pregunta)

# rag_chain = (
#     {"contexto": RunnablePassthrough() | rewriter_chain | retriever,
#      "query":RunnablePassthrough()}
#      | prompt | modelo | StrOutputParser()
# )

# rag_chain.invoke(pregunta)

template_multipregunta = """
Eres un asistente de modelo de lenguajes de IA. Tu tarea es generar cinco versiones diferentes de la pregunta
del usuario para recuperar documentos relevantes de una base de datos vectorial. Al generar multiples
perspectivas sobre la pregunta del usuario, tu objetivo es auxiliar al usuario a superar algunas de las
limitaciones de la búsqueda por similitud basada en distancia. Debes generar únicamente las preguntas alternativas
separadas en filas diferentes (new line) sin ningún texto adicional.

# PREGUNTA ORIGINAL: {question}

# FORMATO DE SALIDA :
["primera pregunta","segunda pregunta",...,"quinta pregunta"]
""" 

prompt_multipregunta = PromptTemplate.from_template(
    template_multipregunta
)

chain_multipregunta = (
    prompt_multipregunta
    | modelo
    | CommaSeparatedListOutputParser()
)

print("\n🔄 Generando preguntas alternativas...")

preguntas = chain_multipregunta.invoke(
    {"question": pregunta}
)

print("\n✅ Preguntas generadas:")

# preguntas = chain_multipregunta.invoke(pregunta)
# print(preguntas)

for i, p in enumerate(preguntas, start=1):
    print(f"{i}. {p}")

rag_chain = (
    {
        "contexto": RunnablePassthrough()
        | retriever,

        "query": RunnablePassthrough()
    }
    | prompt
    | modelo
    | StrOutputParser()
)

print("\n🚀 Ejecutando Multi Query RAG")

# for p in preguntas:
    # rag_chain.invoke(p)

for p in preguntas:

    try:

        respuesta = rag_chain.invoke(
            p
        )

        print("\n==============================")
        print("❓ Pregunta")
        print(p)

        print("\n💬 Respuesta")
        print(respuesta)

    except Exception as e:

        print("\n❌ Error")
        print(e)

print("\n🏁 Proceso finalizado")