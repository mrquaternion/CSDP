import threading

from datetime import date, datetime, time
from flask import Flask, jsonify, render_template, redirect, url_for, session

from file_transfer import file_transfer
from app_forms import AreaTypePointForm, AreaTypeBoundingBoxForm, AreaTypePolygonsForm, CredentialsForm


# ================== App Config ==================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'

STEPS = ['Query Type', 'Area Type', 'Configuration', 'Credentials', 'Remote Monitoring']
AREA_TYPE_STEPS = ['Point', 'Bounding Box', 'Polygons']
QUERY_TYPES = {
    'download_process': {'label': 'Download and Process Data', 'allowed_area_types': {'point', 'bbox', 'polygons'}},
    'gas_flux_predictions': {'label': 'Gas Flux Predictions', 'allowed_area_types': {'bbox', 'polygons'}},
}

TRANSFER_STATE = {
    'running': False,
    'output': '',
    'error': '',
}
TRANSFER_LOCK = threading.Lock()


# ================== App Routes ==================
def _to_session_safe(data):
    safe_data = {}
    for key, value in data.items():
        if isinstance(value, (datetime, date, time)):
            safe_data[key] = value.isoformat()
        else:
            safe_data[key] = value
    return safe_data


@app.route('/')
def index():
    return redirect(url_for('query_type'))


@app.route('/query-type', methods=['GET'])
def query_type():
    return render_template('query_type.html')


@app.route('/query-type/<query_type>')
def select_query_type(query_type):
    """
    Select the query type.
    """

    if query_type not in QUERY_TYPES:
        return 'Invalid query type', 404
    session['query_type'] = query_type
    return redirect(url_for('area_type'))


@app.route('/area-type')
def area_type():
    """
    Select the area type.
    """

    query_type = session.get('query_type')
    if query_type not in QUERY_TYPES:
        return redirect(url_for('query_type'))

    allowed_area_types = QUERY_TYPES[query_type]['allowed_area_types']
    return render_template(
        'area_type.html',
        query_type=query_type,
        query_type_label=QUERY_TYPES[query_type]['label'],
        allowed_area_types=allowed_area_types,
    )


@app.route('/configuration/<area_type>', methods=['GET', 'POST'])
def configuration(area_type):
    """
    Configure the inputs based on the area type.
    """

    query_type = session.get('query_type')
    if query_type not in QUERY_TYPES:
        return redirect(url_for('query_type'))

    allowed_area_types = QUERY_TYPES[query_type]['allowed_area_types']
    if area_type not in allowed_area_types:
        return 'Invalid area type', 404

    match area_type:
        case 'point':
            form = AreaTypePointForm()
        case 'bbox':
            form = AreaTypeBoundingBoxForm()
        case 'polygons':
            form = AreaTypePolygonsForm()

    if form.validate_on_submit():
        session['configuration_data'] = _to_session_safe(form.data)
        return redirect(url_for('credentials'))

    return render_template('configuration.html', template_form=form, area_type=area_type)

 
@app.route('/credentials', methods=['GET', 'POST'])
def credentials():
    """
    Enter the CC credentials.
    """

    form = CredentialsForm()
    if form.validate_on_submit():
        session['account'] = form.data['account']
        return redirect(url_for('monitoring'))

    return render_template('credentials.html', template_form=form)


# ================== Monitoring ==================
@app.route('/remote-monitoring', methods=['GET'])
def monitoring():
    return render_template('remote_monitoring.html')


def _run_transfer(account: str):
    try:
        output = file_transfer(account) or ''
        error = ''
    except Exception as exc:
        output = ''
        error = str(exc)

    with TRANSFER_LOCK:
        TRANSFER_STATE['running'] = False
        TRANSFER_STATE['output'] = output
        TRANSFER_STATE['error'] = error


@app.route('/remote-monitoring/start', methods=['POST'])
def start_monitoring():
    account = session.get('account')
    if not account:
        return jsonify({'ok': False, 'error': 'Missing account in session.'}), 400

    with TRANSFER_LOCK:
        if TRANSFER_STATE['running']:
            return jsonify({'ok': True, 'running': True})

        TRANSFER_STATE['running'] = True
        TRANSFER_STATE['output'] = ''
        TRANSFER_STATE['error'] = ''

    thread = threading.Thread(target=_run_transfer, args=(account,), daemon=True)
    thread.start()
    return jsonify({'ok': True, 'running': True})


@app.route('/remote-monitoring/status', methods=['GET'])
def monitoring_status():
    with TRANSFER_LOCK:
        return jsonify({
            'running': TRANSFER_STATE['running'],
            'output': TRANSFER_STATE['output'],
            'error': TRANSFER_STATE['error'],
        })
