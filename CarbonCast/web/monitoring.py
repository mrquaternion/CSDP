import json
import queue
import threading
import time
from typing import Any

from flask import Blueprint, Response, jsonify, render_template, request, session

from .monitoring_steps.download_step import run_download_step
from .monitoring_steps.postprocessing_step import (
    monitor_postprocessing_job,
    prepare_remote_gas_flux_assets,
    prepare_remote_postprocessing_assets,
    submit_gas_flux_job,
    submit_postprocessing_job,
    validate_optional_job_resources,
    validate_post_processing_payload,
    validate_slurm_account,
    write_generated_gas_flux_job_script,
    write_generated_job_script,
    fetch_remote_outputs,
)


# ================== Monitoring Config ==================
monitoring_bp = Blueprint("monitoring", __name__, url_prefix="/remote-monitoring")
OUTPUTS_READY_URL = "/remote-monitoring/outputs-ready"


def _empty_step_state():
    return {
        "status": "idle",
        "local_output": "",
        "remote_output": "",
        "error": "",
    }


STEPS = {
    "download": _empty_step_state(),
    "post_processing": _empty_step_state(),
    "gas_flux": _empty_step_state(),
}

WORKFLOW_STATE = {
    "phase": "idle",
    "running": False,
    "running_step": None,
    "error": "",
    "download_completed": False,
    "post_processing_completed": False,
    "gas_flux_completed": False,
    "artifacts_path": "",
}

STATE_LOCK = threading.Lock()
STREAM_LOCK = threading.Lock()
STREAM_SUBSCRIBERS = set()


# ================== Events and State ==================
def _broadcast_event(event):
    with STREAM_LOCK:
        subscribers = list(STREAM_SUBSCRIBERS)
    for stream_queue in subscribers:
        stream_queue.put(event)


def _serialize_state_locked():
    download_output = (
        STEPS["download"]["local_output"]
        + STEPS["post_processing"]["local_output"]
        + STEPS["gas_flux"]["local_output"]
    )
    sync_output = (
        STEPS["download"]["remote_output"]
        + STEPS["post_processing"]["remote_output"]
        + STEPS["gas_flux"]["remote_output"]
    )
    error = (
        WORKFLOW_STATE["error"]
        or STEPS["download"]["error"]
        or STEPS["post_processing"]["error"]
        or STEPS["gas_flux"]["error"]
    )
    return {
        "download_output": download_output,
        "sync_output": sync_output,
        "output": download_output + sync_output,
        "error": error,
        "running": WORKFLOW_STATE["running"],
        "phase": WORKFLOW_STATE["phase"],
        "running_step": WORKFLOW_STATE["running_step"],
        "download_completed": WORKFLOW_STATE["download_completed"],
        "post_processing_completed": WORKFLOW_STATE["post_processing_completed"],
        "gas_flux_completed": WORKFLOW_STATE["gas_flux_completed"],
        "artifacts_path": WORKFLOW_STATE["artifacts_path"],
        "can_view_outputs": bool(WORKFLOW_STATE["post_processing_completed"] and WORKFLOW_STATE["artifacts_path"]),
        "can_start_download": not WORKFLOW_STATE["running"],
        "can_start_postprocessing": (not WORKFLOW_STATE["running"]) and WORKFLOW_STATE["download_completed"],
        "can_start_gas_flux": (
            (not WORKFLOW_STATE["running"])
            and WORKFLOW_STATE["post_processing_completed"]
        ),
    }


def _broadcast_state_snapshot():
    with STATE_LOCK:
        snapshot = _serialize_state_locked()
    _broadcast_event({"type": "state", **snapshot})


def _emit_step_output(step_name: str, text: str, stream_type: str):
    if not text:
        return

    with STATE_LOCK:
        if stream_type == "sync":
            STEPS[step_name]["remote_output"] += text
        elif stream_type == "error":
            STEPS[step_name]["error"] = (STEPS[step_name]["error"] or "") + text
        else:
            STEPS[step_name]["local_output"] += text

    if stream_type == "sync":
        _broadcast_event({"type": "sync_output", "text": text})
    elif stream_type == "error":
        _broadcast_event({"type": "error", "text": text, "step": step_name})
    else:
        _broadcast_event({"type": "download_output", "text": text})


# ================== Background Workers ==================
def _run_download(account: str, configuration_data: Any | None):
    try:
        run_download_step(
            account=account,
            configuration_data=configuration_data,
            emit_output=lambda text, stream: _emit_step_output("download", text, stream),
        )
        with STATE_LOCK:
            STEPS["download"]["status"] = "done"
            STEPS["download"]["error"] = ""
            WORKFLOW_STATE["running"] = False
            WORKFLOW_STATE["running_step"] = None
            WORKFLOW_STATE["phase"] = "idle"
            WORKFLOW_STATE["error"] = ""
            WORKFLOW_STATE["download_completed"] = True
            WORKFLOW_STATE["post_processing_completed"] = False
            WORKFLOW_STATE["gas_flux_completed"] = False
        _broadcast_event({"type": "done", "step": "download"})
    except Exception as exc:
        error = str(exc)
        with STATE_LOCK:
            STEPS["download"]["status"] = "failed"
            STEPS["download"]["error"] = error
            WORKFLOW_STATE["running"] = False
            WORKFLOW_STATE["running_step"] = None
            WORKFLOW_STATE["phase"] = "failed"
            WORKFLOW_STATE["error"] = error
        _broadcast_event({"type": "error", "text": error, "step": "download"})
        _broadcast_event({"type": "done", "step": "download"})

    _broadcast_state_snapshot()


def _run_post_processing(
    account: str,
    configuration_data: Any | None,
    memory: str,
    cpus: int,
    wall_time: str,
    slurm_account: str,
    use_gas_flux_workflow: bool,
):
    try:
        del use_gas_flux_workflow
        generated_job_script = write_generated_job_script(
            slurm_account=slurm_account,
            memory=memory,
            cpus=cpus,
            wall_time=wall_time,
        )

        remote_pipeline_dir, remote_config_path, remote_job_script_path = prepare_remote_postprocessing_assets(
            account=account,
            configuration_data=configuration_data,
            local_job_script_path=generated_job_script,
            include_gas_flux_repo=False,
            emit_output=lambda text, stream: _emit_step_output("post_processing", text, stream),
        )

        job_id = submit_postprocessing_job(
            account=account,
            remote_pipeline_dir=remote_pipeline_dir,
            remote_job_script_path=remote_job_script_path,
            remote_config_path=remote_config_path,
            emit_output=lambda text, stream: _emit_step_output("post_processing", text, stream),
        )

        final_state = monitor_postprocessing_job(
            account=account,
            remote_pipeline_dir=remote_pipeline_dir,
            job_id=job_id,
            emit_output=lambda text, stream: _emit_step_output("post_processing", text, stream),
        )
        if final_state != "COMPLETED":
            raise RuntimeError(f"Post-processing Slurm job {job_id} finished with state: {final_state}")

        local_out_dir = fetch_remote_outputs(
            account=account,
            remote_pipeline_dir=remote_pipeline_dir,
            emit_output=lambda text, stream: _emit_step_output("post_processing", text, stream),
        )

        with STATE_LOCK:
            STEPS["post_processing"]["status"] = "done"
            STEPS["post_processing"]["error"] = ""
            WORKFLOW_STATE["running"] = False
            WORKFLOW_STATE["running_step"] = None
            WORKFLOW_STATE["phase"] = "done"
            WORKFLOW_STATE["error"] = ""
            WORKFLOW_STATE["post_processing_completed"] = True
            WORKFLOW_STATE["gas_flux_completed"] = False
            WORKFLOW_STATE["artifacts_path"] = local_out_dir
        _broadcast_event({"type": "artifacts_ready", "path": local_out_dir, "url": OUTPUTS_READY_URL})
        _broadcast_event({"type": "done", "step": "post_processing"})
    except Exception as exc:
        error = str(exc)
        with STATE_LOCK:
            STEPS["post_processing"]["status"] = "failed"
            STEPS["post_processing"]["error"] = error
            WORKFLOW_STATE["running"] = False
            WORKFLOW_STATE["running_step"] = None
            WORKFLOW_STATE["phase"] = "failed"
            WORKFLOW_STATE["error"] = error
        _broadcast_event({"type": "error", "text": error, "step": "post_processing"})
        _broadcast_event({"type": "done", "step": "post_processing"})

    _broadcast_state_snapshot()


def _run_gas_flux(
    account: str,
    slurm_account: str,
    memory: str,
    cpus: int,
    wall_time: str,
):
    try:
        generated_job_script = write_generated_gas_flux_job_script(
            slurm_account=slurm_account,
            memory=memory,
            cpus=cpus,
            wall_time=wall_time,
        )
        remote_repo_dir, remote_job_script_path = prepare_remote_gas_flux_assets(
            account=account,
            local_job_script_path=generated_job_script,
            emit_output=lambda text, stream: _emit_step_output("gas_flux", text, stream),
        )
        job_id = submit_gas_flux_job(
            account=account,
            remote_repo_dir=remote_repo_dir,
            remote_job_script_path=remote_job_script_path,
            emit_output=lambda text, stream: _emit_step_output("gas_flux", text, stream),
        )
        final_state = monitor_postprocessing_job(
            account=account,
            remote_pipeline_dir=remote_repo_dir,
            job_id=job_id,
            emit_output=lambda text, stream: _emit_step_output("gas_flux", text, stream),
        )
        if final_state != "COMPLETED":
            raise RuntimeError(f"Gas flux Slurm job {job_id} finished with state: {final_state}")

        with STATE_LOCK:
            STEPS["gas_flux"]["status"] = "done"
            STEPS["gas_flux"]["error"] = ""
            WORKFLOW_STATE["running"] = False
            WORKFLOW_STATE["running_step"] = None
            WORKFLOW_STATE["phase"] = "done"
            WORKFLOW_STATE["error"] = ""
            WORKFLOW_STATE["gas_flux_completed"] = True
        _broadcast_event({"type": "done", "step": "gas_flux"})
    except Exception as exc:
        error = str(exc)
        with STATE_LOCK:
            STEPS["gas_flux"]["status"] = "failed"
            STEPS["gas_flux"]["error"] = error
            WORKFLOW_STATE["running"] = False
            WORKFLOW_STATE["running_step"] = None
            WORKFLOW_STATE["phase"] = "failed"
            WORKFLOW_STATE["error"] = error
        _broadcast_event({"type": "error", "text": error, "step": "gas_flux"})
        _broadcast_event({"type": "done", "step": "gas_flux"})

    _broadcast_state_snapshot()


# ================== Routes ==================
@monitoring_bp.route("/start-download", methods=["POST"])
@monitoring_bp.route("/start", methods=["POST"])
def monitoring_start_download():
    account = session.get("account")
    configuration_data = session.get("configuration_data")
    if not account:
        return jsonify({"ok": False, "error": "Missing account in session."}), 400

    with STATE_LOCK:
        if WORKFLOW_STATE["running"]:
            return jsonify({"ok": True, "running": True, **_serialize_state_locked()})

        STEPS["download"] = _empty_step_state()
        STEPS["download"]["status"] = "running"
        STEPS["post_processing"] = _empty_step_state()
        STEPS["gas_flux"] = _empty_step_state()

        WORKFLOW_STATE["running"] = True
        WORKFLOW_STATE["running_step"] = "download"
        WORKFLOW_STATE["phase"] = "downloading"
        WORKFLOW_STATE["error"] = ""
        WORKFLOW_STATE["download_completed"] = False
        WORKFLOW_STATE["post_processing_completed"] = False
        WORKFLOW_STATE["gas_flux_completed"] = False
        WORKFLOW_STATE["artifacts_path"] = ""
        snapshot = _serialize_state_locked()

    _broadcast_state_snapshot()
    thread = threading.Thread(target=_run_download, args=(account, configuration_data), daemon=True)
    thread.start()
    return jsonify({"ok": True, **snapshot})


@monitoring_bp.route("/start-postprocessing", methods=["POST"])
def monitoring_start_postprocessing():
    account = session.get("account")
    configuration_data = session.get("configuration_data")
    if not account:
        return jsonify({"ok": False, "error": "Missing account in session."}), 400

    payload = request.get_json(silent=True) or {}
    try:
        memory, cpus, wall_time, slurm_account = validate_post_processing_payload(payload)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    with STATE_LOCK:
        if WORKFLOW_STATE["running"]:
            return jsonify({"ok": True, "running": True, **_serialize_state_locked()})
        if not WORKFLOW_STATE["download_completed"]:
            return jsonify({"ok": False, "error": "Download must complete before post-processing."}), 400

        STEPS["post_processing"] = _empty_step_state()
        STEPS["post_processing"]["status"] = "running"
        STEPS["gas_flux"] = _empty_step_state()
        WORKFLOW_STATE["running"] = True
        WORKFLOW_STATE["running_step"] = "post_processing"
        WORKFLOW_STATE["phase"] = "postprocessing"
        WORKFLOW_STATE["error"] = ""
        WORKFLOW_STATE["gas_flux_completed"] = False
        WORKFLOW_STATE["artifacts_path"] = ""
        snapshot = _serialize_state_locked()

    _broadcast_state_snapshot()
    thread = threading.Thread(
        target=_run_post_processing,
        args=(
            account,
            configuration_data,
            memory,
            cpus,
            wall_time,
            slurm_account,
            False,
        ),
        daemon=True,
    )
    thread.start()
    return jsonify({"ok": True, **snapshot})


@monitoring_bp.route("/start-gas-flux", methods=["POST"])
def monitoring_start_gas_flux():
    account = session.get("account")
    query_type = session.get("query_type")
    if not account:
        return jsonify({"ok": False, "error": "Missing account in session."}), 400
    if query_type != "gas_flux_predictions":
        return jsonify({"ok": False, "error": "Gas flux step is only available for gas_flux_predictions."}), 400

    payload = request.get_json(silent=True) or {}
    try:
        slurm_account = validate_slurm_account(payload.get("slurm_account"))
        memory, cpus, wall_time = validate_optional_job_resources(
            payload.get("gas_flux_job_config"),
            fallback_memory="24G",
            fallback_cpus=8,
            fallback_wall_time="08:00:00",
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    with STATE_LOCK:
        if WORKFLOW_STATE["running"]:
            return jsonify({"ok": True, "running": True, **_serialize_state_locked()})
        if not WORKFLOW_STATE["post_processing_completed"]:
            return jsonify({"ok": False, "error": "Post-processing must complete before gas flux."}), 400

        STEPS["gas_flux"] = _empty_step_state()
        STEPS["gas_flux"]["status"] = "running"
        WORKFLOW_STATE["running"] = True
        WORKFLOW_STATE["running_step"] = "gas_flux"
        WORKFLOW_STATE["phase"] = "gasflux"
        WORKFLOW_STATE["error"] = ""
        WORKFLOW_STATE["gas_flux_completed"] = False
        snapshot = _serialize_state_locked()

    _broadcast_state_snapshot()
    thread = threading.Thread(
        target=_run_gas_flux,
        args=(account, slurm_account, memory, cpus, wall_time),
        daemon=True,
    )
    thread.start()
    return jsonify({"ok": True, **snapshot})


@monitoring_bp.route("/stream", methods=["GET"])
def monitoring_stream():
    def event_stream():
        stream_queue = queue.Queue()
        with STREAM_LOCK:
            STREAM_SUBSCRIBERS.add(stream_queue)
        try:
            with STATE_LOCK:
                snapshot = {
                    "type": "snapshot",
                    **_serialize_state_locked(),
                }
            yield f"data: {json.dumps(snapshot)}\n\n"

            while True:
                try:
                    event = stream_queue.get(timeout=1)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    time.sleep(0.1)
        finally:
            with STREAM_LOCK:
                STREAM_SUBSCRIBERS.discard(stream_queue)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(event_stream(), headers=headers, mimetype="text/event-stream")


@monitoring_bp.route("/status", methods=["GET"])
def monitoring_status():
    with STATE_LOCK:
        return jsonify(_serialize_state_locked())


@monitoring_bp.route("/outputs-ready", methods=["GET"])
def outputs_ready():
    with STATE_LOCK:
        can_view_outputs = bool(WORKFLOW_STATE["post_processing_completed"] and WORKFLOW_STATE["artifacts_path"])
        artifacts_path = WORKFLOW_STATE["artifacts_path"]
    return render_template(
        "outputs_ready.html",
        can_view_outputs=can_view_outputs,
        artifacts_path=artifacts_path,
    )
