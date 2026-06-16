from langchain_ollama import OllamaLLM

modelo = OllamaLLM(model="gemma3:4b")

respuesta = modelo.invoke("Hola")

print(respuesta)