#!/usr/bin/python
from flaskapp import app
from flask import render_template, request, Flask

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0')
