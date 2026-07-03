#!/usr/bin/env python3
"""Local web chat UI for ggnpu — spawns ggnpu per request (prototype)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPO_ROOT / "web"

_inference_lock = threading.Lock()


def format_prompt(message: str, history: list[dict]) -> str:
    lines: list[str] = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    lines.append(f"User: {message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def run_ggnpu(
    ggnpu: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temp: float,
    ctx_size: int | None,
    timeout: int,
) -> tuple[str, str, int, float]:
    cmd = [
        ggnpu,
        "-m",
        model,
        "-p",
        prompt,
        "-n",
        str(max_tokens),
        "--temp",
        str(temp),
        "--quiet",
    ]
    if ctx_size is not None and ctx_size > 0:
        cmd.extend(["-c", str(ctx_size)])

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        partial = exc.stdout or ""
        err = (exc.stderr or "").strip() or f"ggnpu timed out after {timeout}s"
        return partial.strip(), err, 1, elapsed

    elapsed = time.monotonic() - start
    return result.stdout.strip(), result.stderr.strip(), result.returncode, elapsed


class ChatHandler(BaseHTTPRequestHandler):
    ggnpu_bin: str = ""
    model_path: str = ""
    max_tokens: int = 128
    temp: float = 0.7
    ctx_size: int | None = None
    timeout: int = 300

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body"})
            return

        message = (data.get("message") or "").strip()
        if not message:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "message is required"})
            return

        history = data.get("history") or []
        if not isinstance(history, list):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "history must be an array"})
            return

        max_tokens = int(data.get("max_tokens", self.max_tokens))
        temp = float(data.get("temp", self.temp))

        if not _inference_lock.acquire(blocking=False):
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "Inference already in progress. Try again shortly."},
            )
            return

        try:
            prompt = format_prompt(message, history)
            stdout, stderr, code, elapsed = run_ggnpu(
                self.ggnpu_bin,
                self.model_path,
                prompt,
                max_tokens,
                temp,
                self.ctx_size,
                self.timeout,
            )
        finally:
            _inference_lock.release()

        if code != 0:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": stderr or f"ggnpu exited with code {code}",
                    "elapsed_s": round(elapsed, 2),
                },
            )
            return

        self._send_json(
            HTTPStatus.OK,
            {"response": stdout, "elapsed_s": round(elapsed, 2)},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="ggnpu local web chat server")
    parser.add_argument("--model", required=True, help="Path to .gguf model file")
    parser.add_argument(
        "--ggnpu",
        default=str(REPO_ROOT / "build-npu" / "ggnpu"),
        help="Path to ggnpu binary (default: ./build-npu/ggnpu)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Default max new tokens")
    parser.add_argument("--temp", type=float, default=0.7, help="Default temperature")
    parser.add_argument("--ctx-size", type=int, default=0, help="Context window override (0 = model default)")
    parser.add_argument("--timeout", type=int, default=300, help="Subprocess timeout in seconds")
    args = parser.parse_args()

    ggnpu = os.path.abspath(args.ggnpu)
    model = os.path.abspath(args.model)

    if not os.path.isfile(ggnpu):
        print(f"Error: ggnpu binary not found: {ggnpu}", file=sys.stderr)
        return 1
    if not os.path.isfile(model):
        print(f"Error: model not found: {model}", file=sys.stderr)
        return 1
    if not (WEB_ROOT / "index.html").is_file():
        print(f"Error: web UI not found: {WEB_ROOT / 'index.html'}", file=sys.stderr)
        return 1

    ChatHandler.ggnpu_bin = ggnpu
    ChatHandler.model_path = model
    ChatHandler.max_tokens = args.max_tokens
    ChatHandler.temp = args.temp
    ChatHandler.ctx_size = args.ctx_size if args.ctx_size > 0 else None
    ChatHandler.timeout = args.timeout

    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"ggnpu chat server at http://{args.host}:{args.port}")
    print(f"  model: {model}")
    print(f"  ggnpu: {ggnpu}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
