import time
import httpx

from openai import OpenAI


class VLLMProxyClient:
    def __init__(self, proxy_url: str, model2spec: dict[str, str]):
        self.proxy_url = proxy_url.rstrip("/")
        self.model2spec = {k: f"{k}\n{v}" for k, v in model2spec.items()}

    def status(self):
        r = httpx.get(f"{self.proxy_url}/status", timeout=30)
        r.raise_for_status()
        return r.json()

    def switch(self, model_name: str, timeout_s: float = 900.0):
        st = time.time()
        r = httpx.post(
            f"{self.proxy_url}/switch",
            json={"model_spec": self.model2spec[model_name]},
            timeout=timeout_s,
        )
        duration = time.time() - st
        r.raise_for_status()
        return {"response": r.json(), "loaded_in_s": duration}

    def get_model_name(self):
        s = self.status()
        if s.get("running") and s.get("model_spec"):
            return s["model_spec"].split("\n")[0]
        return None

    def get_openai_client(self, timeout_s: float = 3600.0):
        if OpenAI is None:
            raise RuntimeError("openai is not installed")
        return OpenAI(api_key="EMPTY", base_url=f"{self.proxy_url}/v1", timeout=timeout_s)
