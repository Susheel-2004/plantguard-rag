from flask import Flask, request, render_template, jsonify, make_response
from time import sleep
from query_rag import query_rag
from populate_general_database import add_tuple_to_chroma
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

CHROMA_PATH = "chroma"
# SENSOR_DATA_PATH = "chromaSensorData"
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

@app.route("/populate", methods=['POST'])
def populate():
    data = request.get_json()
    row = data['tuple']
    # add_tuple_to_chroma(row)
    print(row)

    return jsonify({"response":"Populating the database. This may take a while."})


if __name__ == "__main__":
    app.run(debug=True)