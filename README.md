# AIC vLLM Proxy

A lightweight **FastAPI-based proxy** that manages the lifecycle of a `vllm serve`
process and forwards all OpenAI-compatible API requests to it.

The proxy allows you to:
- start / stop / switch vLLM models dynamically
- keep vLLM isolated on a GPU node
- interact with the model from notebooks or scripts using an OpenAI-compatible client
- install and run everything via `pip` (no local checkout required)

---

## Architecture overview

```

[Jupyter / Python client]
|
v
AIC vLLM Proxy (FastAPI)
|
v
vllm serve
|
v
GPU

````

The proxy:
- runs on a GPU node
- launches `vllm serve` as a subprocess
- waits until the model is ready
- proxies **all** requests (`/v1/chat/completions`, `/v1/embeddings`, etc.)
- exposes logs and status endpoints

---

## Requirements

- Python **3.10+**
- `vllm` available in the environment (installed separately)
- GPU node for running the server
- Network access between client and server nodes

---

## Installation

You do **not** need to register this package on PyPI.

### Install directly from GitHub

```bash
pip install "aic-vllm-proxy[server,client] @ git+https://github.com/<YOUR_ORG>/aic-vllm-proxy.git"
````

Extras:

* `server` → installs `uvicorn`
* `client` → installs `openai` (for OpenAI-compatible usage)

---

## Running the proxy server (GPU node)

Activate your virtual environment and run:

```bash
aic-vllm-proxy --host 0.0.0.0 --port 8224
```

This starts the FastAPI proxy on port `8224`.

> 💡 The proxy itself does **not** load any model on startup.
> A model is loaded only after calling `/switch`.

---

## API endpoints

### `GET /status`

Returns current proxy / vLLM status.

```json
{
  "running": true,
  "pid": 12345,
  "model_spec": "meta-llama/Llama-2-7b-hf\n--tensor-parallel-size 1",
  "vllm_base": "http://127.0.0.1:8225",
  "log_path": "logs/vllm/2024-01-01T12-00-00_pid-12345.log"
}
```

---

### `POST /switch`

Start or switch the vLLM model.

Example payload:

```json
{
  "model_spec": "meta-llama/Llama-2-7b-hf --tensor-parallel-size 1 --gpu-memory-utilization 0.9"
}
```

* Stops the currently running vLLM instance (if any)
* Starts a new one
* Waits until the model is ready
* Returns once `/v1/models` responds

---

### `GET /logs/current`

Return full current vLLM log.

### `GET /logs/tail?lines=200`

Return last N log lines.

---

### Catch-all proxy

All other paths are forwarded to the running vLLM server:

```
/v1/chat/completions
/v1/embeddings
/v1/models
...
```

---

## Python client usage (e.g. Jupyter notebook)

### Example: switching models

```python
from aic_vllm_proxy.client import VLLMProxyClient

model2spec = {
    "meta-llama/Llama-2-7b-hf": "--tensor-parallel-size 1 --gpu-memory-utilization 0.9",
    "mistralai/Mistral-7B-Instruct-v0.2": "--tensor-parallel-size 1",
}

proxy = VLLMProxyClient(
    proxy_url="http://gpu-node:8224",
    model2spec=model2spec,
)

print(proxy.status())
proxy.switch("meta-llama/Llama-2-7b-hf")
```

---

### Using OpenAI-compatible API

```python
client = proxy.get_openai_client()

resp = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Explain vLLM in one sentence"}],
)

print(resp.choices[0].message.content)
```

The model name here is ignored by vLLM (only one model is loaded), but it is kept
for OpenAI API compatibility.

---

## Typical cluster workflow

1. Allocate GPU node (e.g. via SLURM)
2. Activate virtual environment
3. Run `aic-vllm-proxy`
4. Connect from notebook / login node via HTTP
5. Switch models as needed without restarting jobs

---

## Notes & limitations

* Only **one vLLM instance** is managed at a time
* Proxy is intended for **single-user or controlled multi-user** setups
* `vllm` must be installed separately and available in `$PATH`
* Logs are stored locally on the server node

---

## License

MIT License
