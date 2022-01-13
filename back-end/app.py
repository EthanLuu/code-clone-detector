import json
from flask import Flask, request

import sys
sys.path.append("../detector")
from detect import Detector

app = Flask(__name__)


@app.route("/")
def detect():
    detector = Detector()
    data = request.get_data()
    body = json.loads(data)
    res = detector.detect(body['codeX'], body['codeY'])
    print(res)
    return "Hello"


if __name__ == "__main__":
    app.run(debug=True)
