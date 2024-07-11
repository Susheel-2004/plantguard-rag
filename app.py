from flask import Flask, request, render_template, jsonify, make_response
from time import sleep
from query_rag import query_rag

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/query", methods=['POST'])
def user_query():
    data = request.get_json()
    query_text = data['question']
    response = query_rag(query_text)
    response_object = make_response(jsonify({"response":response}))
    response_object.status_code = 200
    return response_object


if __name__ == "__main__":
    app.run(debug=True)