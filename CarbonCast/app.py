from datetime import date, datetime, time
from flask import Flask, render_template, redirect, url_for, session
from app_forms import AreaTypePointForm, AreaTypeBoundingBoxForm, AreaTypePolygonsForm, CredentialsForm
from monitoring import monitoring_bp


# ================== App Config ==================
app = Flask(__name__)
app.register_blueprint(monitoring_bp)
app.config['SECRET_KEY'] = 'mysecret'

STEPS = ['Query Type', 'Area Type', 'Configuration', 'Credentials', 'Remote Monitoring']
AREA_TYPE_STEPS = ['Point', 'Bounding Box', 'Polygons']
QUERY_TYPES = {
    'download_process': {'label': 'Download and Process Data', 'allowed_area_types': {'point', 'bbox', 'polygons'}},
    'gas_flux_predictions': {'label': 'Gas Flux Predictions', 'allowed_area_types': {'bbox', 'polygons'}},
}


# ================== App Routes ==================
def _to_session_safe(data):
    safe_data = {}
    for key, value in data.items():
        if isinstance(value, (datetime, date, time)):
            safe_data[key] = value.isoformat()
        else:
            safe_data[key] = value
    return safe_data


# ===== ROUTE 1 =====
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


# ===== ROUTE 2 =====
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


# ===== ROUTE 3 =====
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
        return redirect(url_for('area_type'))

    match area_type:
        case 'point':
            form = AreaTypePointForm()
        case 'bbox':
            form = AreaTypeBoundingBoxForm()
        case 'polygons':
            form = AreaTypePolygonsForm()

    if form.validate_on_submit():
        session['configuration_data'] = _to_session_safe(form.data)
        print(session['configuration_data'])
        return redirect(url_for('credentials'))

    return render_template('configuration.html', template_form=form, area_type=area_type)

 
 # ===== ROUTE 4 =====
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


# ===== ROUTE 5 =====
@app.route('/remote-monitoring', methods=['GET'])
def monitoring():
    account = session.get('account', '')
    slurm_account_default = account.split('@', 1)[0] if account else ''
    return render_template(
        'remote_monitoring.html',
        slurm_account_default=slurm_account_default,
    )
