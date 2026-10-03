#!/usr/bin/env python3
"""Tiny OpenAI-compatible mock for dry-running the pipeline without a GPU. Deterministic latent utility per item
(hash-seeded N(0,1)); answers /v1/models and /v1/chat/completions with A/B top_logprobs. Logs whether prefill
(continue_final_message) was requested. Usage: mock_openai_server.py [port]"""
import hashlib, json, math, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
def mu(item):  # deterministic pseudo-utility in [-2, 2]
    h = int(hashlib.sha256(item.encode()).hexdigest(), 16) % 10**6 / 10**6
    return 4 * h - 2
PHI = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.endswith("/models"): return self._send({"object": "list", "data": [{"id": "mock", "object": "model"}]})
        self._send({"error": "nope"}, 404)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = next(m["content"] for m in body["messages"] if m["role"] == "user")
        m = re.search(r"A: (.*?) or B: (.*?)\? Answer", user); a, b = m.group(1), m.group(2)
        p_a = PHI((mu(a) - mu(b)) / math.sqrt(2))
        if "negatively" in user: p_a = 1 - p_a
        prefill = body.get("continue_final_message") is True and body["messages"][-1]["role"] == "assistant"
        sys_prompt = any(mm["role"] == "system" for mm in body["messages"])
        tops = [{"token": "A", "logprob": math.log(max(p_a, 1e-9))}, {"token": "B", "logprob": math.log(max(1 - p_a, 1e-9))},
                {"token": "<", "logprob": -9.0}]
        tok = "A" if p_a >= 0.5 else "B"
        content = [{"token": tok, "logprob": tops[0 if tok == "A" else 1]["logprob"], "top_logprobs": tops}]
        if not prefill:  # chat mode: model emits the tag first, letter second
            content = [{"token": "<answer>", "logprob": -0.01, "top_logprobs": [{"token": "<answer>", "logprob": -0.01}]}] + content
        self._send({"id": "x", "object": "chat.completion", "model": body["model"],
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": f"<answer>{tok}</answer>"},
                                 "logprobs": {"content": content}, "finish_reason": "stop"}],
                    "mock": {"prefill": prefill, "system_prompt": sys_prompt, "extra": body.get("chat_template_kwargs")}})
if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print("mock on", port, flush=True); ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
