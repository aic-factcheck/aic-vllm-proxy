import asyncio
import os
import shlex
import signal
import subprocess
from datetime import datetime
from typing import Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, PlainTextResponse

app = FastAPI()

# ----------------------------
# Config
# ----------------------------
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8225
VLLM_BASE = f"http://{VLLM_HOST}:{VLLM_PORT}"

PROXY_REQUEST_TIMEOUT_S = 600.0          # inference requests may be long
VLLM_STARTUP_TIMEOUT_S = 900.0           # model loading can take minutes
VLLM_POLL_INTERVAL_S = 1.0

LOG_DIR = "logs/vllm"
os.makedirs(LOG_DIR, exist_ok=True)

# ----------------------------
# State
# ----------------------------
current_process: Optional[subprocess.Popen] = None
current_model_spec: Optional[str] = None

current_log_path: Optional[str] = None
current_log_file = None  # file handle


# ----------------------------
# vLLM lifecycle helpers
# ----------------------------
def _make_log_path_prefix(model_spec: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # Keep it filename-safe and not insane length
    safe = (
        model_spec.replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace("--", "_")
    )
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in ("_", "-", "."))
    safe = safe[:120].rstrip("_")
    return f"{LOG_DIR}/{ts}_{safe}"

def _canonicalize_model_spec(model_spec: str, port: int) -> tuple[str, list[str]]:
    """
    Returns (canonical_string, tokens) where:
      - args are shlex-split
      - --port is forced to exactly `port` (replaces any existing --port value)
    """
    tokens = shlex.split(model_spec)

    # Remove any existing --port <value> (handle repeated occurrences)
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "--port":
            i += 2  # skip flag + value
            continue
        out.append(tokens[i])
        i += 1

    # Force our port at the end (last-wins, deterministic)
    out += ["--port", str(port)]

    # Canonical string mainly for logging/storage
    canonical = " ".join(shlex.quote(t) for t in out)
    return canonical, out



def start_vllm_server(model_spec: str) -> Tuple[subprocess.Popen, str]:
    """
    Start vllm serve with arbitrary args (model_spec is shlex-split).
    Creates a unique log file per run, then renames to include PID.
    Returns (proc, log_path).
    """
    global current_log_file, current_log_path

    # Build command safely (supports quoted args)
    cmd = ["vllm", "serve"] + shlex.split(model_spec)

    # Prepare log file (PID not known yet)
    prefix = _make_log_path_prefix(model_spec)
    tmp_log_path = f"{prefix}_pid-PENDING.log"
    log_f = open(tmp_log_path, "a", buffering=1)  # line-buffered
    log_f.write(f"[proxy] starting: {' '.join(cmd)}\n")

    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Rename log file to include real PID (POSIX-safe; file handle remains valid)
    pid_log_path = f"{prefix}_pid-{proc.pid}.log"
    try:
        os.rename(tmp_log_path, pid_log_path)
        log_f.write(f"[proxy] pid={proc.pid} log={pid_log_path}\n")
    except Exception as e:
        # If rename fails, keep the tmp name but still record it
        pid_log_path = tmp_log_path
        log_f.write(f"[proxy] pid={proc.pid} (rename failed: {e})\n")

    current_log_file = log_f
    current_log_path = pid_log_path
    return proc, pid_log_path


def kill_current_server():
    global current_process, current_model_spec, current_log_file, current_log_path

    if not current_process:
        return

    try:
        if current_log_file:
            current_log_file.write("[proxy] stopping current vLLM...\n")
            current_log_file.flush()
    except Exception:
        pass

    # Try graceful shutdown first
    current_process.send_signal(signal.SIGTERM)
    try:
        current_process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        # Force kill
        current_process.kill()
        current_process.wait(timeout=15)

    # Close log file handle
    try:
        if current_log_file:
            current_log_file.write("[proxy] vLLM process exited.\n")
            current_log_file.flush()
            current_log_file.close()
    except Exception:
        pass

    current_process = None
    current_model_spec = None
    current_log_file = None
    current_log_path = None


async def wait_for_ready(
    proc: subprocess.Popen,
    timeout_s: float = VLLM_STARTUP_TIMEOUT_S,
) -> bool:
    """
    Poll vLLM until /v1/models returns 200 (and looks like OpenAI models list),
    but fail fast if the vLLM process exits (e.g., model load failure).
    """
    deadline = asyncio.get_event_loop().time() + timeout_s

    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            # 1) Process still alive?
            rc = proc.poll()
            if rc is not None:
                # Process exited
                return False

            # 2) API ready?
            try:
                r = await client.get(f"{VLLM_BASE}/v1/models", timeout=2.0)
                if r.status_code == 200:
                    # Optional: sanity-check response shape to avoid false positives
                    try:
                        j = r.json()
                        if isinstance(j, dict) and j.get("object") == "list" and "data" in j:
                            return True
                        # If it’s 200 but not expected shape, keep waiting
                    except Exception:
                        # If JSON parse fails, keep waiting
                        pass
            except Exception:
                pass

            await asyncio.sleep(VLLM_POLL_INTERVAL_S)

    return False


# ----------------------------
# Control endpoints
# ----------------------------
@app.get("/status")
def status():
    return {
        "running": current_process is not None,
        "pid": current_process.pid if current_process else None,
        "model_spec": current_model_spec,
        "vllm_base": VLLM_BASE,
        "log_path": current_log_path,
    }

switch_lock = asyncio.Lock()

@app.post("/switch")
async def switch_model(payload: dict):
    """
    Switch to a different model "flavor" identified by model_spec (full CLI args).
    payload: {"model_spec": "meta-llama/Llama-2-7b-hf --port 8000 --gpu-memory-utilization 0.9 ..."}
    """
    
    async with switch_lock:
        global current_process, current_model_spec

        raw_spec = payload.get("model_spec")
        if not raw_spec:
            raise HTTPException(status_code=400, detail="model_spec is required")
        
        model_spec, model_tokens = _canonicalize_model_spec(raw_spec, VLLM_PORT)
        
        # If identical to current, no-op
        if current_process is not None and current_model_spec == model_spec:
            return {
                "status": "already running",
                "pid": current_process.pid,
                "model_spec": current_model_spec,
                "log_path": current_log_path,
            }

        # Kill existing and start new
        kill_current_server()
        proc, log_path = start_vllm_server(model_spec)
        current_process = proc
        current_model_spec = model_spec

        ready = await wait_for_ready(proc)
        if not ready:
            # Capture exit code if it died
            rc = proc.poll()

            # Ensure it’s not left running in a weird state
            kill_current_server()

            if rc is not None:
                raise HTTPException(
                    status_code=500,
                    detail=f"vLLM exited during startup (exit_code={rc}). See log: {log_path}",
                )

            raise HTTPException(
                status_code=504,
                detail=f"vLLM did not become ready within {VLLM_STARTUP_TIMEOUT_S}s. See log: {log_path}",
            )

        return {"status": "ready", "pid": proc.pid, "model_spec": model_spec, "log_path": log_path}


# ----------------------------
# Log exposure endpoints
# ----------------------------
@app.get("/logs/current", response_class=PlainTextResponse)
def get_current_log():
    """
    Returns the full current log file contents as plain text.
    (For very large logs, prefer /logs/tail.)
    """
    if not current_log_path or not os.path.exists(current_log_path):
        raise HTTPException(status_code=404, detail="No active log file")
    with open(current_log_path, "r", errors="replace") as f:
        return f.read()


@app.get("/logs/tail", response_class=PlainTextResponse)
def tail_current_log(lines: int = 200):
    """
    Returns the last N lines of the current log file as plain text.
    """
    if lines < 1:
        raise HTTPException(status_code=400, detail="lines must be >= 1")
    if not current_log_path or not os.path.exists(current_log_path):
        raise HTTPException(status_code=404, detail="No active log file")

    # Simple tail implementation (good enough for moderate files)
    with open(current_log_path, "r", errors="replace") as f:
        data = f.readlines()
    return "".join(data[-lines:])


# ----------------------------
# Catch-all proxy route (MUST BE LAST)
# ----------------------------
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_all(request: Request, path: str):
    """
    Proxy any route to the running vLLM server.
    """
    if current_process is None:
        raise HTTPException(status_code=503, detail="No model loaded (call /switch first)")

    # Avoid forwarding hop-by-hop headers; also avoid forwarding Host to backend
    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "connection", "keep-alive", "proxy-authenticate",
                             "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade")
    }

    async with httpx.AsyncClient() as client:
        url = f"{VLLM_BASE}/{path}"
        resp = await client.request(
            request.method,
            url,
            headers=forward_headers,
            params=request.query_params,
            content=await request.body(),
            timeout=PROXY_REQUEST_TIMEOUT_S,
        )

    # Return proxied response (keep content-type, etc.)
    # Note: some headers like content-encoding may not be appropriate to forward if you transform content (we don't).
    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="AIC vLLM Proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8224, help="Proxy server port (FastAPI)")
    args = parser.parse_args()

    uvicorn.run("aic_vllm_proxy.server:app", host=args.host, port=args.port)
