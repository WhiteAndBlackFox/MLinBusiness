# -*- coding: utf-8 -*-
from flask import Flask

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'tmp'
app.config['ALLOWED_EXTENSIONS'] = set(['csv'])
app.config['TIMESTAMP'] = '%Y%m%d%H%M%S'
app.config['PATH_PROJECT'] = 'Lessons9'
app.config['PATH_MODEL'] = 'save_model'

from app import views
