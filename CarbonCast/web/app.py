from datetime import date, datetime, time
from pathlib import Path

from flask import Flask, redirect, render_template, session, url_for

from .forms import (
    AreaTypeBoundingBoxForm,
    AreaTypePointForm,
    AreaTypePolygonsForm,
    CredentialsForm,
)
from .monitoring import monitoring_bp


QUERY_TYPES = {
    "download_process": {
        "label": "Download and Process Data",
        "allowed_area_types": {"point", "bbox", "polygons"},
    },
    "gas_flux_predictions": {
        "label": "Gas Flux Predictions",
        "allowed_area_types": {"bbox", "polygons"},
    },
}


def _to_session_safe(data):
    safe_data = {}
    for key, value in data.items():
        if isinstance(value, (datetime, date, time)):
            safe_data[key] = value.isoformat()
        else:
            safe_data[key] = value
    return safe_data


def create_app():
    project_root = Path(__file__).resolve().parents[1]
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    app.register_blueprint(monitoring_bp)
    app.config["SECRET_KEY"] = "mysecret"

    @app.route("/")
    def index():
        return redirect(url_for("query_type"))

    @app.route("/query-type", methods=["GET"])
    def query_type():
        return render_template("query_type.html")

    @app.route("/query-type/<query_type>")
    def select_query_type(query_type):
        if query_type not in QUERY_TYPES:
            return "Invalid query type", 404
        session["query_type"] = query_type
        return redirect(url_for("area_type"))

    @app.route("/area-type")
    def area_type():
        query_type = session.get("query_type")
        if query_type not in QUERY_TYPES:
            return redirect(url_for("query_type"))

        allowed_area_types = QUERY_TYPES[query_type]["allowed_area_types"]
        return render_template(
            "area_type.html",
            query_type=query_type,
            query_type_label=QUERY_TYPES[query_type]["label"],
            allowed_area_types=allowed_area_types,
        )

    @app.route("/configuration/<area_type>", methods=["GET", "POST"])
    def configuration(area_type):
        query_type = session.get("query_type")
        if query_type not in QUERY_TYPES:
            return redirect(url_for("query_type"))

        allowed_area_types = QUERY_TYPES[query_type]["allowed_area_types"]
        if area_type not in allowed_area_types:
            return redirect(url_for("area_type"))

        if area_type == "point":
            form = AreaTypePointForm()
        elif area_type == "bbox":
            form = AreaTypeBoundingBoxForm()
        else:
            form = AreaTypePolygonsForm()

        if form.validate_on_submit():
            session["configuration_data"] = _to_session_safe(form.data)
            return redirect(url_for("credentials"))

        return render_template("configuration.html", template_form=form, area_type=area_type)

    @app.route("/credentials", methods=["GET", "POST"])
    def credentials():
        form = CredentialsForm()
        if form.validate_on_submit():
            session["account"] = form.data["account"]
            return redirect(url_for("monitoring"))

        return render_template("credentials.html", template_form=form)

    @app.route("/remote-monitoring", methods=["GET"])
    def monitoring():
        account = session.get("account", "")
        query_type = session.get("query_type", "")
        slurm_account_default = account.split("@", 1)[0] if account else ""
        return render_template(
            "remote_monitoring.html",
            query_type=query_type,
            slurm_account_default=slurm_account_default,
        )

    return app


app = create_app()
