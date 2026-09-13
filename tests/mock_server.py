"""Threaded mock OpenAI-compatible server for probe tests (stdlib only)."""
from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    server_version = "RigSLOMock/0.1"
    n_tokens = 32          # content deltas to emit
    chunk_delay = 0.001    # seconds between chunks
    ttft_delay = 0.005     # seconds before first token
    fail = False

    def log_message(self, *a):  # quiet
        pass

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.server.fail:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "boom"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(self.server.ttft_delay)
        words = ["one", "two", "three", "four"]
        for i in range(self.server.n_tokens):
            chunk = {"choices": [{"delta": {"content": words[i % 4] + " "},
                                  "finish_reason": None}]}
            if i == self.server.n_tokens - 1:
                chunk["choices"][0]["finish_reason"] = "stop"
                chunk["usage"] = {"prompt_tokens": 64, "completion_tokens": self.server.n_tokens}
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.flush()
            time.sleep(self.server.chunk_delay)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def start(port: int = 0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    srv.n_tokens = _Handler.n_tokens
    srv.chunk_delay = _Handler.chunk_delay
    srv.ttft_delay = _Handler.ttft_delay
    srv.fail = _Handler.fail
    import threading
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]


def stop(srv):
    srv.shutdown()
    srv.server_close()
