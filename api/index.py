
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from common import norm_text


class handler(BaseHTTPRequestHandler):

    def send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/" or path == "/index.html":
            return self.send_json(200, {
                "message": "Uyjoy RAG API is running"
            })

        if path == "/v1/health":
            return self.send_json(200, {
                "status": "ok"
            })

        return self.send_json(404, {
            "error": "Not found",
            "path": path
        })

    def do_POST(self):
        path = urlparse(self.path).path

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            data = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self.send_json(400, {
                "error": "Invalid JSON"
            })

        if path == "/v1/normalize":
            text = data.get("text", "")
            return self.send_json(200, {
                "original": text,
                "normalized": norm_text(text)
            })

        return self.send_json(404, {
            "error": "Not found",
            "path": path
        })

    def log_message(self, format, *args):
        pass
