import json
from flask import Flask, request
from flask_cors import CORS
from detect import Detector

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def index():
    return "<h1>Hello There! It's EthanLoo.</h1>"


@app.route("/detect", methods=['POST'])
def detect():
    response = {"data": False}
    try:
        detector = Detector()
        data = request.get_data()
        body = json.loads(data)
        res = detector.detect(body['codeX'], body['codeY'])
        for flag in res:
            if flag:
                response['data'] = True
    except Exception as error:
        response['error'] = error

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0")
