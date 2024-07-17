from langchain_community.vectorstores import Chroma
# from langchain.utilities import SQLDatabase
# from langchain_experimental.sql import SQLDatabaseChain
# from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.llms.ollama import Ollama
# from langchain_community.chat_models import ChatOllama
from get_embedding_function import get_embedding_function
import textwrap



CHROMA_PATH = "chroma"
SENSOR_DATA_PATH = "chromaSensorData"
PROMPT_TEMPLATE = """
Answer the question based on the following context and if you don't know the answer, use your own knowledge.:

{context}

---

Answer the question based on the above context and if you don't know the answer, use your own knowledge: {question}
"""
NEW_PROMPT_TEMPLATE = """
Provide a clear and concise answer to the question below based on the context related to plants. Do not include any phrases that indicate the source of the information. if you can't answer the question, say you don't know and urge them to ask questions related to the context.
If they say hi, hello or sny other greeting words, greet them back simply.:
{context}

---

Question: {question}

"""


def get_response(query, chain):
    # Getting response from chain
    response = chain({'query': query})
    
    # Wrapping the text for better output in Jupyter Notebook
    wrapped_text = textwrap.fill(response['result'], width=100)
    print(wrapped_text)

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    # print("Chroma ready.")

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    # print("Search complete.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(NEW_PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print("Prompt ready.")
    

    model = Ollama(model="llama3")
    # print("Model ready.")
    response_text = model.invoke(prompt)
    # print("Response ready.")

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"{response_text}\n"

    return formatted_response

