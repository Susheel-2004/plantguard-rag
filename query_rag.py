from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
# from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import retrieval_qa
from get_embedding_function import get_embedding_function
import textwrap



CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """<s> [INST] You are an assistant for question-answering tasks about Chariot Solutions. Use the following pieces of retrieved context 
                        to answer the question. If you don't know the answer, just say that you don't know. [/INST] </s> 
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
# prompt = """
# ### System:
# You are an AI Assistant that follows instructions extreamly well. \
# Help as much as you can.

# ### User:
# {prompt}

# ### Response:

# """

# template = """
# ### System:
# You are an respectful and honest assistant. You have to answer the user's \
# questions using only the context provided to you. If you don't know the answer, \
# just say you don't know. Don't try to make up an answer.

# ### Context:
# {context}

# ### User:
# {question}

# ### Response:
# """

def load_qa_chain(retriever, llm, prompt):
    return retrieval_qa.from_chain_type(
        llm=llm,
        retriever=retriever, # here we are using the vectorstore as a retriever
        chain_type="stuff",
        return_source_documents=True, # including source documents in output
        chain_type_kwargs={'prompt': prompt} # customizing the prompt
    )

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
    print("Chroma ready.")

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    print("Search complete.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("Prompt ready.")
    

    model = Ollama(model="mistral")
    print("Model ready.")
    response_text = model.invoke(prompt)
    print("Response ready.")

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"{response_text}\n"

    return formatted_response
