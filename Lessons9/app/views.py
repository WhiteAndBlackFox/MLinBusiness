# -*- coding: utf-8 -*-
import fnmatch
import os
import h2o
from datetime import datetime

from flask import render_template, request, redirect, send_file, safe_join, jsonify
from flaskthreads import ThreadPoolWithAppContextExecutor
from werkzeug.utils import secure_filename

from .model import TeachModels
from app import app


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


def do_create_model(filename):
    m = TeachModels(filename)
    m.create()
    print(f"Обучена модель: {m.get_name_model()}")


@app.route('/create_model/<path:filename>')
def create_model(filename):
    with ThreadPoolWithAppContextExecutor(max_workers=2) as pool:
        future = pool.submit(do_create_model, (filename))
        future.result()
    return redirect('/list_load_data')


@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    root_dir = os.path.dirname(os.getcwd())
    sj = safe_join(root_dir, app.config['PATH_PROJECT'], app.config['UPLOAD_FOLDER'], filename)
    return send_file(sj, as_attachment=True)


@app.route('/list_load_data')
def list_load_data():
    list_data = {}
    list_csv = fnmatch.filter(os.listdir(app.config['UPLOAD_FOLDER']), '*.csv')
    full_list = [os.path.join(app.config['UPLOAD_FOLDER'], i) for i in list_csv]
    for idx, ld in enumerate(sorted(full_list, key=os.path.getmtime)):
        time_uploads = datetime.fromtimestamp(os.path.getmtime(ld)).strftime('%d-%m-%Y %H:%M:%S')
        filename = ld.replace(app.config['UPLOAD_FOLDER'] + '\\', '')
        model_name = None
        if (os.path.exists(os.path.join(app.config['PATH_MODEL'], f"model_{filename.split('.')[0]}"))):
            model_name = f"model_{filename.split('.')[0]}"
        list_data[idx] = {
            'filename': filename,
            'time_upload': time_uploads,
            'model_name': model_name
        }
    return render_template('list_load_data.html', list_data=list_data)


@app.route('/load_new_data', methods=['GET', 'POST'])
def load_new_data():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = "%s_%s.%s" % (
                filename.split('.')[0], datetime.now().strftime(app.config['TIMESTAMP']), filename.split('.')[1])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect("/list_load_data")
    return render_template('load_new_data.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    request_json = request.json
    if 'model' not in request_json:
        return jsonify({"code": 500, "message": "Не задана модель (параметр model)!"})
    ID = request_json["ID"]
    h2o.init()

    path_model = os.path.join(app.config['PATH_MODEL'], request_json['model'])
    models = os.listdir(path_model)

    if len(models) == 0:
        return jsonify({"code": 500, "message": "Не найдена модель!"})

    glm = h2o.load_model(os.path.join(path_model, models[0]))

    hf = h2o.H2OFrame([[0 * 35]], column_names=[
        'LicAge',
        'Gender',
        'MariStat',
        'DrivAge',
        'HasKmLimit',
        'BonusMalus',
        'OutUseNb',
        'RiskArea',
        'VehUsg_Private',
        'VehUsg_Private+trip to office',
        'VehUsg_Professional',
        'VehUsg_Professional run',
        'SocioCateg_CSP1',
        'SocioCateg_CSP2',
        'SocioCateg_CSP3',
        'SocioCateg_CSP4',
        'SocioCateg_CSP5',
        'SocioCateg_CSP6',
        'SocioCateg_CSP7'
    ])

    try:

        hf[0, 'LicAge'] = request_json["LicAge"]
        hf[0, 'Gender'] = 0 if request_json["Gender"] == 'Male' else 1
        hf[0, 'MariStat'] = 0 if request_json["MariStat"] == 'Other' else 1
        hf[0, 'DrivAge'] = request_json["DrivAge"]
        hf[0, 'HasKmLimit'] = request_json["HasKmLimit"]
        hf[0, 'BonusMalus'] = request_json["BonusMalus"]
        hf[0, 'OutUseNb'] = request_json["OutUseNb"]
        hf[0, 'RiskArea'] = request_json["RiskArea"]

        hf[0, 'VehUsg_Private'] = 1 if request_json["VehUsage"] == 'Private' else 0
        hf[0, 'VehUsg_Private+trip to office'] = 1 if request_json["VehUsage"] == 'Private+trip to office' else 0
        hf[0, 'VehUsg_Professional'] = 1 if request_json["VehUsage"] == 'Professional' else 0
        hf[0, 'VehUsg_Professional run'] = 1 if request_json["VehUsage"] == 'Professional run' else 0

        key_csp = request_json["SocioCateg"][3]
        hf[0, f"SocioCateg_CSP{key_csp}"] = 1

        predict_glm = glm.predict(hf)
        value_glm = predict_glm.as_data_frame()['predict']

        print("value_glm", value_glm)

    except Exception as e:
        return jsonify({'code': 500, 'Message': e})

    h2o.shutdown()
    return jsonify({'ID': ID, 'value': str(value_glm[0])})
