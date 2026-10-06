"""A small local web page for hand labelling the sample.

Runs on http://localhost:<port> using only the standard library. The model's
predictions are never sent to the page, so labelling stays blind.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from reviews import labels

PAGE = Path(__file__).with_name("label_app.html")


def _payload(cfg: dict) -> dict:
    sample = pd.read_csv(labels.SAMPLE_FILE, dtype={"review_id": str}, keep_default_na=False)
    done = labels.load_labels().set_index("review_id").to_dict("index")
    items = []
    for row in sample.itertuples():
        lab = done.get(row.review_id)
        items.append({
            "id": row.review_id, "score": int(row.review_score), "split": row.split,
            "pt": row.text, "en": row.text_en,
            "label": None if lab is None else {
                "themes": [t for t in lab["themes"].split(";") if t],
                "unsure": lab["unsure"] == "1", "note": lab["note"]},
        })
    return {"themes": cfg["themes"], "items": items}


def serve(cfg: dict) -> None:
    port = cfg["labels"].get("port", 8765)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/items":
                self._send(200, json.dumps(_payload(cfg)).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/label":
                return self._send(404, b"not found", "text/plain")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bad = [t for t in body["themes"] if t not in cfg["themes"]]
            if bad:
                return self._send(400, f"unknown themes {bad}".encode(), "text/plain")
            n = labels.save_label(body["id"], body["themes"], body["unsure"], body.get("note", ""))
            self._send(200, json.dumps({"saved": n}).encode(), "application/json")

        def log_message(self, *args):  # keep the terminal quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"Labelling tool running at {url}  (press Ctrl+C here to stop)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Labels are saved in data/labels/labels.csv")
