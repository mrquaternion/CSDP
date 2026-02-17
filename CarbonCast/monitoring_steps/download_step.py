from typing import Any, Callable

from .download_and_sync import download_and_sync


def run_download_step(
    account: str,
    configuration_data: Any | None,
    emit_output: Callable[[str, str], None],
) -> None:
    def _on_output(text: str, stream_type: str = "download"):
        emit_output(text, stream_type)

    download_and_sync(account, configuration_data, on_output=_on_output)
