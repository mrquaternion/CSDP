document.addEventListener("DOMContentLoaded", () => {
    const downloadEl = document.getElementById('download-output');
    const syncEl = document.getElementById('sync-output');
    const errorEl = document.getElementById('transfer-error');
    const startDownloadBtn = document.getElementById('start-download-btn');
    const startPostBtn = document.getElementById('start-postprocessing-btn');
    const statusEl = document.getElementById('transfer-status');
    const memoryInput = document.getElementById('post-memory');
    const cpusInput = document.getElementById('post-cpus');
    const timeInput = document.getElementById('post-time');
    const slurmAccountInput = document.getElementById('post-slurm-account');
    let downloadBuffer = '';
    let syncBuffer = '';
    let errorBuffer = '';
    let stream = null;

    function normalizeOutput(text) {
        return (text || '').replace(/\r/g, '\n');
    }

    function isNearBottom(el, threshold = 24) {
        return el.scrollHeight - el.clientHeight - el.scrollTop <= threshold;
    }

    function scrollToBottom(el) {
        el.scrollTop = el.scrollHeight;
    }

    function statusLabelFromPhase(phase) {
        const phaseMap = {
            idle: 'Idle',
            downloading: 'Downloading',
            postprocessing: 'Postprocessing',
            done: 'Done',
            failed: 'Failed',
        };
        return phaseMap[phase] || 'Idle';
    }

    function applyState(state) {
        const shouldAutoScrollDownload = isNearBottom(downloadEl);
        const shouldAutoScrollSync = isNearBottom(syncEl);

        downloadBuffer = normalizeOutput(state.download_output);
        syncBuffer = normalizeOutput(state.sync_output);
        errorBuffer = normalizeOutput(state.error);
        downloadEl.textContent = downloadBuffer || 'No output yet.';
        syncEl.textContent = syncBuffer || 'No output yet.';
        errorEl.textContent = errorBuffer || 'No errors.';

        if (shouldAutoScrollDownload) {
            scrollToBottom(downloadEl);
        }
        if (shouldAutoScrollSync) {
            scrollToBottom(syncEl);
        }

        if (statusEl) {
            statusEl.textContent = statusLabelFromPhase(state.phase);
        }

        if (startDownloadBtn) {
            startDownloadBtn.disabled = !Boolean(state.can_start_download);
        }
        if (startPostBtn) {
            startPostBtn.disabled = !Boolean(state.can_start_postprocessing);
        }
    }

    async function startDownload() {
        if (startDownloadBtn) {
            startDownloadBtn.disabled = true;
        }

        const response = await fetch('/remote-monitoring/start-download', { method: 'POST' });
        const data = await response.json();
        if (!response.ok || !data.ok) {
            errorEl.textContent = data.error || 'Unknown error.';
            if (statusEl) {
                statusEl.textContent = 'Failed';
            }
            if (startDownloadBtn) {
                startDownloadBtn.disabled = false;
            }
            return;
        }

        applyState(data);
        startStream();
    }

    async function startPostprocessing() {
        if (startPostBtn) {
            startPostBtn.disabled = true;
        }

        const payload = {
            memory: memoryInput ? memoryInput.value.trim() : '',
            cpus: cpusInput ? cpusInput.value.trim() : '',
            time: timeInput ? timeInput.value.trim() : '',
            slurm_account: slurmAccountInput ? slurmAccountInput.value.trim() : '',
        };

        const response = await fetch('/remote-monitoring/start-postprocessing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
            errorEl.textContent = data.error || 'Unknown error.';
            if (statusEl) {
                statusEl.textContent = 'Failed';
            }
            if (startPostBtn) {
                startPostBtn.disabled = false;
            }
            return;
        }

        applyState(data);
        startStream();
    }

    function startStream() {
        if (stream) return;

        stream = new EventSource('/remote-monitoring/stream');
        stream.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'snapshot') {
                applyState(data);
                return;
            }

            if (data.type === 'state') {
                applyState(data);
                return;
            }

            if (data.type === 'download_output') {
                const shouldAutoScroll = isNearBottom(downloadEl);
                downloadBuffer += normalizeOutput(data.text);
                downloadEl.textContent = downloadBuffer || 'No output yet.';
                if (shouldAutoScroll) {
                    scrollToBottom(downloadEl);
                }
                return;
            }

            if (data.type === 'sync_output') {
                const shouldAutoScroll = isNearBottom(syncEl);
                syncBuffer += normalizeOutput(data.text);
                syncEl.textContent = syncBuffer || 'No output yet.';
                if (shouldAutoScroll) {
                    scrollToBottom(syncEl);
                }
                return;
            }

            if (data.type === 'error') {
                errorBuffer += normalizeOutput(data.text || 'Unknown error.\n');
                errorEl.textContent = errorBuffer || 'No errors.';
                return;
            }
        };

        stream.onerror = () => {
            errorEl.textContent = 'Streaming connection failed.';
            if (statusEl) {
                statusEl.textContent = 'Disconnected';
            }
            if (startDownloadBtn) {
                startDownloadBtn.disabled = false;
            }
            if (startPostBtn) {
                startPostBtn.disabled = false;
            }
        };
    }

    if (startDownloadBtn) {
        startDownloadBtn.addEventListener('click', startDownload);
    }
    if (startPostBtn) {
        startPostBtn.addEventListener('click', startPostprocessing);
    }
    startStream();
    scrollToBottom(downloadEl);
    scrollToBottom(syncEl);
});
