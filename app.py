from flask import Flask, request, render_template, jsonify, make_response
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from time import sleep
from query_rag import query_rag

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/user", methods=['POST'])
def user_query():
    data = request.get_json()
    print(data)
    query_text = data['question']
    print(query_text)
    response = query_rag(query_text)
    print(response)
    response_object = make_response(jsonify({"response":response}))
    response_object.status_code = 200
    return response_object

@app.route("/test", methods=["GET", "POST"])
def test():
    if (request.method == "POST"):
        sleep(5)
        data = request.get_json()
        return jsonify({"response": f"Hello {data["name"]}"})
    return jsonify({"response": "Hello, World!"})

if __name__ == "__main__":
    app.run(debug=True)