#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
duck.ai client, vendored and trimmed from https://github.com/DarkBytee/DuckAi (MIT).
Only the two things F.R.I.D.A.Y. needs are kept: streaming text and image generation.

duck.ai is an undocumented private endpoint with no API key and no SLA. It guards
POST /duckchat/v1/chat with a JavaScript anti-abuse challenge, which jsa_jsdom.js
runs through Node.js + jsdom. So the runner needs `node` on PATH and jsdom installed
(see package.json) before any route here can work.
"""

import base64
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import tempfile
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://duck.ai"
log = logging.getLogger("duckai")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36")

# duck.ai accepts a picture as an input part. WebP keeps the payload small; the
# upstream client compresses the same way before an image ever reaches the wire.
IMG_MAX_DIM = 1568
IMG_QUALITY = 85
IMG_MAX_PAYLOAD = 3_500_000

# gpt-6-luna accepts "low" and rejects "medium"/"high" with HTTP 400, so reasoning is
# on by default at the only level it honours. Anything else stays opt-in per caller.
REASONING_EFFORT = {
    "gpt-6-luna": "low",
    "tinfoil/gpt-oss-120b": "low",
}

# duck.ai's anti-abuse challenge starts answering 418 when requests arrive back to back,
# so consecutive chat calls keep a small gap and a failed challenge waits much longer
# than a normal error before the next attempt.
MIN_GAP = float(os.getenv("DUCKAI_MIN_GAP_SEC") or 4)
_last_post = 0.0


class DuckAIError(RuntimeError):
    """A duck.ai failure. status is None when the request never got an HTTP answer."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def _b64(data):
    return base64.b64encode(data).decode()


def _sha256_b64(text):
    return _b64(hashlib.sha256(text.encode()).digest())


def _b64_json(obj):
    return _b64(json.dumps(obj, separators=(",", ":")).encode())


def image_part(data, mime=""):
    """Bytes -> duck.ai input part. Re-encoded to WebP within the payload cap so a
    full-size article banner still fits; raw bytes are used only when Pillow is absent."""
    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is None:
        if len(data) > IMG_MAX_PAYLOAD:
            raise DuckAIError("image is too large and Pillow is not installed")
        mime = mime or ("image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else
                        "image/webp" if data[8:12] == b"WEBP" else "image/jpeg")
        return {"type": "image", "mimeType": mime,
                "image": "data:" + mime + ";base64," + _b64(data)}
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise DuckAIError("cannot read the reference image: {}".format(e))
    if im.mode in ("RGBA", "P", "LA"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    elif im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    scale = max(w, h) / IMG_MAX_DIM
    if scale > 1:
        im = im.resize((max(1, int(w / scale)), max(1, int(h / scale))),
                       Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC)
    quality = IMG_QUALITY
    while True:
        buf = io.BytesIO()
        im.save(buf, "WEBP", quality=quality)
        encoded = buf.getvalue()
        if len(encoded) <= IMG_MAX_PAYLOAD or quality <= 30:
            break
        quality -= 15
    if len(encoded) > IMG_MAX_PAYLOAD:
        raise DuckAIError("reference image is still too large after compression")
    return {"type": "image", "mimeType": "image/webp",
            "image": "data:image/webp;base64," + _b64(encoded)}


class DuckAI:
    def __init__(self, timeout=60, proxy=None, reasoning="", tools=True):
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
        if proxy:
            self.s.proxies = {"http": proxy, "https": proxy}
        self.reasoning = (reasoning or "").strip().lower()
        self.tools = tools
        self.fe_version = ""
        self.jsa = ""

    # --- anti-abuse handshake ---
    def bootstrap(self):
        r = self.s.get(BASE + "/", timeout=self.timeout)
        r.raise_for_status()
        tag = re.search(r'data-version-tag="([^"]*)"', r.text)
        sha = re.search(r'data-version-sha="([^"]*)"', r.text)
        self.fe_version = "{}-{}".format(tag.group(1) if tag else "dev",
                                         sha.group(1) if sha else "hash")
        return self

    def fetch_challenge(self):
        r = self.s.get(
            BASE + "/duckchat/v1/status",
            headers={"x-vqd-accept": "1", "Cache-Control": "no-store",
                     "x-fe-version": self.fe_version,
                     "Origin": BASE, "Referer": BASE + "/"},
            timeout=self.timeout,
        )
        raw = r.headers.get("x-vqd-hash-1")
        if not raw:
            raise DuckAIError("no challenge received (HTTP {})".format(r.status_code),
                              status=r.status_code)
        return base64.b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8", "replace")

    def solve(self, js_source):
        runner = os.path.join(HERE, "jsa_jsdom.js")
        if not os.path.isfile(runner):
            raise DuckAIError("jsa_jsdom.js not found next to duckai_client.py")
        fd, path = tempfile.mkstemp(suffix=".js")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(js_source)
            try:
                out = subprocess.run(["node", runner, path, UA], capture_output=True,
                                     text=True, timeout=90, cwd=HERE)
            except FileNotFoundError:
                raise DuckAIError("node was not found on PATH (needed for the duck.ai challenge)")
            except subprocess.TimeoutExpired:
                raise DuckAIError("duck.ai challenge timed out")
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if out.returncode != 0 or not out.stdout.strip():
            if "Cannot find module" in out.stderr and "jsdom" in out.stderr:
                raise DuckAIError("jsdom is missing, run: npm install")
            raise DuckAIError("challenge execution failed: {}".format(out.stderr[:300]))
        try:
            result = json.loads(out.stdout)
        except json.JSONDecodeError as e:
            raise DuckAIError("challenge output is not JSON: {}".format(e))
        if "__error" in result:
            raise DuckAIError("error inside challenge: {}".format(result["__error"][:300]))
        return result

    def build_proof(self, solved):
        proof = dict(solved)
        proof["client_hashes"] = [_sha256_b64(x) for x in solved["client_hashes"]]
        meta = dict(solved.get("meta") or {})
        meta.update({
            "origin": BASE,
            "stack": ("Error\nat l (https://duck.ai/dist/duckai-dist/entry.duckai.js:2:2129171)\n"
                      "at async https://duck.ai/dist/duckai-dist/entry.duckai.js:2:1859809"),
            "duration": "58",
        })
        proof["meta"] = meta
        return _b64_json(proof)

    def rehandshake(self):
        self.jsa = ""
        self.bootstrap()
        self.jsa = self.build_proof(self.solve(self.fetch_challenge()))
        return self

    def _effort(self, model, draw):
        # an image call does not reason, and sending a level there would only give the
        # endpoint another chance to reject the request
        if draw:
            return "none"
        return self.reasoning or REASONING_EFFORT.get(model, "none")

    # --- chat ---
    def _post_chat(self, prompt, model, history, draw, images=None, search=False):
        content = [{"type": "text", "text": prompt}]
        if images:
            content.extend(images)
        effort = self._effort(model, draw)
        body = {
            "model": model,
            "messages": (history or []) + [{"role": "user", "content": content}],
            "canUseTools": bool(self.tools),
            "reasoningEffort": effort,
            "canUseApproxLocation": None,
            "canDelegateImageGeneration": None,
            "canShowGreeting": False,
        }
        if draw:
            body["metadata"] = {"toolChoice": {"GenerateImage": True}}
        elif search and self.tools:
            # observed on the live endpoint: this forces the WebSearch tool and the
            # answer comes back with <citation src="N"></citation> markers
            body["metadata"] = {"toolChoice": {"WebSearch": True}}
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Origin": BASE,
            "Referer": BASE + "/",
            "x-fe-version": self.fe_version,
            "x-fe-signals": _b64_json({
                "start": int(time.time() * 1000),
                "events": [{"name": "action", "delta": 1200, "trusted": True}],
                "end": 1500,
            }),
            "X-Vqd-Hash-1": self.jsa,
        }
        return self.s.post(BASE + "/duckchat/v1/chat", headers=headers,
                           data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                           stream=True, timeout=self.timeout)

    def _events(self, prompt, model, history, draw, retries, images=None, search=False):
        """Run one prompt and yield (kind, value). Every attempt starts from a fresh
        handshake: the solved proof is single-use, and reusing one is exactly what the
        endpoint answers with 418 ERR_CHALLENGE, so a retry never inherits stale state."""
        global _last_post
        attempts = max(1, retries)
        last, last_status = "", None
        for attempt in range(attempts):
            gap = MIN_GAP - (time.time() - _last_post)
            if gap > 0:
                time.sleep(min(gap, MIN_GAP))
            try:
                self.rehandshake()
                _last_post = time.time()
                r = self._post_chat(prompt, model, history, draw, images=images, search=search)
                if r.status_code == 200:
                    return self._read_stream(r)
                last = "HTTP {}: {}".format(r.status_code, r.text[:200])
                last_status = r.status_code
                r.close()
                if r.status_code == 400 and self._effort(model, draw) not in ("", "none"):
                    # the endpoint refused this reasoning level for the model: drop it
                    # for the remaining attempts instead of failing the whole reply
                    log.warning("duck.ai rejected reasoning effort %r, retrying without it",
                                self._effort(model, draw))
                    self.reasoning = "none"
            except requests.RequestException as e:
                last, last_status = type(e).__name__, None    # str(e) repeats the URL
            except DuckAIError as e:
                last, last_status = str(e), e.status
            if last_status in (418, 429):
                time.sleep(min(40, 8 + attempt * 8))
            else:
                time.sleep(2 + attempt * 2)
        raise DuckAIError("duck.ai failed after {} attempts: {}".format(attempts, last),
                          status=last_status)

    @staticmethod
    def _read_stream(r):
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.startswith("["):
                continue
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            role = ev.get("role") or ""
            name = ev.get("name") or ""
            # reasoning runs arrive as separate chunks; they are working notes, never
            # part of the answer, so they must not reach the caller's text
            if (role in ("reasoning", "thinking", "thought")
                    or name in ("reasoning", "thinking", "thought")
                    or ev.get("type") in ("reasoning", "thinking")
                    or ev.get("action") in ("reasoning", "thinking")):
                continue
            msg = ev.get("message")
            if isinstance(msg, str) and msg:
                yield ("message", msg)
                continue
            if ev.get("action") == "error":
                raise DuckAIError("duck.ai server error ({}): {}".format(
                    ev.get("status"), ev.get("type", "?")), status=ev.get("status"))
            if role == "tool-invocation":
                yield ("tool", {"toolName": ev.get("toolName"), "state": ev.get("state"),
                                "toolArguments": ev.get("toolArguments")})
                continue
            if role == "ui-component" and name == "generate-image":
                data = ev.get("data") or {}
                if data.get("status") == "success" and data.get("b64Image"):
                    yield ("image", {"b64": data["b64Image"],
                                     "format": data.get("format", "jpeg"),
                                     "width": data.get("width"),
                                     "height": data.get("height")})

    def ask(self, prompt, model="gpt-6-luna", history=None, retries=5, search=False):
        parts = [v for kind, v in self._events(prompt, model, history, False, retries,
                                               search=search)
                 if kind == "message"]
        return "".join(parts)

    def draw(self, prompt, model="gpt-6-luna", retries=5, image=None):
        """Text -> image bytes, or None when duck.ai answered without an image.
        image=(bytes, mime) turns it into an edit: the picture rides along in the
        same user turn, so the generator sees what it is supposed to change."""
        ref = image_part(*image) if image else None
        images = [ref] if ref else None
        found = None
        for kind, val in self._events(prompt, model, None, True, retries, images=images):
            if kind == "image":
                found = val
        return base64.b64decode(found["b64"]) if found else None


# ════════════════════════════════════════════════════════════════════
# GLUE USED BY main.py
# ════════════════════════════════════════════════════════════════════
CLIENT = None


def get_client(timeout=90, reasoning=None, tools=None):
    global CLIENT
    if CLIENT is None:
        proxy = (os.getenv("DUCKAI_PROXY") or "").strip() or None
        if reasoning is None:
            reasoning = (os.getenv("DUCKAI_REASONING_EFFORT") or "").strip()
        if tools is None:
            tools = (os.getenv("DUCKAI_WEB_SEARCH") or "true").strip().lower() not in \
                ("0", "false", "no", "off")
        CLIENT = DuckAI(timeout=timeout, proxy=proxy, reasoning=reasoning, tools=tools)
    return CLIENT


def reset_client():
    global CLIENT
    CLIENT = None


def to_prompt_and_history(messages):
    """duck.ai takes one prompt plus a user/assistant history and has no system role,
    so every system message becomes a prefix on the last user turn."""
    systems, turns = [], []
    for m in messages:
        role = m.get("role")
        text = m.get("content")
        if isinstance(text, list):                      # openai parts shape
            text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
        text = (text or "").strip()
        if not text:
            continue
        if role == "system":
            systems.append(text)
        elif role in ("user", "assistant"):
            turns.append((role, text))
    while turns and turns[-1][0] == "assistant":
        turns.pop()          # duck.ai has no slot for a trailing bot turn: it is not a prompt
    if not turns or turns[-1][0] != "user":
        return None, []
    prompt = turns[-1][1]
    if systems:
        prompt = "\n\n".join(systems) + "\n\n" + prompt
    history = [{"role": r, "content": [{"type": "text", "text": t}] if r == "user" else t}
               for r, t in turns[:-1]]
    return prompt, history
