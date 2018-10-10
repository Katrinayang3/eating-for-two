"""
Initialization for the design
"""

from flask import Flask
app = Flask(__name__)
from flaskapp import views
