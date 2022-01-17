#!/usr/bin/env python

from flask import Flask, Response, render_template
from utils import Stream

app = Flask(__name__)


@app.route("/")
def index():
    return {"status": "success"}


@app.route("/stream")
def stream():
    cv_stream = Stream()
    return Response(
        cv_stream.gen(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", threaded=True)
