#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agentic terminal work inside a GitHub Codespace.

The codespace is a separate VM, private to the account, and it never holds a credential:
this module only sends command strings over `gh codespace ssh`, the bot's process
environment is not forwarded over that channel, and every command runs with an empty
environment so the codespace's own GITHUB_TOKEN is not visible to it. Starting is on
demand and stopping happens in a `finally`, so the VM only bills compute while a job is
actually running.

Flow: find (or create) the codespace -> start it -> wait for "Available" ->
run commands over ssh -> stop it again.
"""

import base64
import json
import logging
import os
import shlex
import shutil
import stat
import subprocess
import time

log = logging.getLogger("codespace")

GH = "gh"
SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")
POLL_SEC = 5


class CodespaceError(RuntimeError):
    pass


class Unavailable(CodespaceError):
    """The account cannot run a codespace right now: no quota, no billing, or no gh."""


# ════════════════════════════════════════════════════════════════════
# TRANSPORT (monkeypatched in tests)
# ════════════════════════════════════════════════════════════════════
def _env(token=""):
    # Rebuilt rather than inherited: the ssh channel never needs the bot's secrets.
    keep = {"PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR", "TEMP", "TMP",
            "SYSTEMROOT", "USERPROFILE", "APPDATA", "LOCALAPPDATA"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.setdefault("PATH", os.defpath)
    if token:
        env["GH_TOKEN"] = token
    return env


def _run(args, *, token="", timeout=120, stdin=None):
    try:
        return subprocess.run([GH] + args, capture_output=True, text=True,
                              timeout=timeout, env=_env(token), input=stdin)
    except FileNotFoundError:
        raise Unavailable("gh CLI not found on PATH")
    except subprocess.TimeoutExpired:
        raise CodespaceError("gh %s timed out after %ss" % (" ".join(args[:2]), timeout))


def _api(method, path, payload=None, *, token="", timeout=120):
    args = ["api", "-X", method, path]
    stdin = None
    if payload is not None:
        args += ["--input", "-"]
        stdin = json.dumps(payload)
    r = _run(args, token=token, timeout=timeout, stdin=stdin)
    body = (r.stdout or "").strip()
    if r.returncode != 0:
        err = (r.stderr or body).strip()
        raise _classify(err)
    try:
        return json.loads(body) if body else {}
    except ValueError:
        return {}


def _classify(err):
    low = err.lower()
    for needle in ("quota", "billing", "payment", "spending limit", "not enabled",
                   "codespaces is not available"):
        if needle in low:
            return Unavailable("codespaces unavailable: %s" % err[:200])
    return CodespaceError(err[:300])


# ════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ════════════════════════════════════════════════════════════════════
def _find(label, repo, token):
    """Look the codespace up by display name: the real name is assigned by the API."""
    data = _api("GET", "user/codespaces?per_page=100", token=token)
    for cs in data.get("codespaces") or []:
        if (cs.get("repository") or {}).get("full_name") != repo:
            continue
        if label and label not in (cs.get("display_name"), cs.get("name")):
            continue
        return cs
    return None


def _ensure(label, repo, machine, idle_min, token):
    cs = _find(label, repo, token)
    if cs:
        return cs["name"]
    log.info("creating codespace %s on %s (%s)", label, repo, machine)
    cs = _api("POST", "repos/%s/codespaces" % repo,
              {"ref": "main", "machine": machine, "idle_timeout_minutes": idle_min,
               "display_name": label, "multi_repo_permissions_opt_out": True},
              token=token, timeout=180)
    name = cs.get("name")
    if not name:
        cs = _find(label, repo, token)
        name = (cs or {}).get("name")
    if not name:
        raise CodespaceError("codespace %s was not created (no name in the response)" % label)
    return name


def _wait_available(name, token, budget):
    deadline = time.time() + budget
    state = ""
    while time.time() < deadline:
        cs = _api("GET", "user/codespaces/%s" % name, token=token)
        state = cs.get("state") or ""
        if state == "Available":
            return cs
        if state in ("Failed", "Deleted", "Archived"):
            raise CodespaceError("codespace %s is %s" % (name, state))
        time.sleep(POLL_SEC)
    raise CodespaceError("codespace %s not available after %ss (state=%s)" % (name, budget, state))


def start(name, token, budget=300):
    cs = _api("GET", "user/codespaces/%s" % name, token=token)
    if cs.get("state") != "Available":
        log.info("starting codespace %s (state=%s)", name, cs.get("state"))
        _api("POST", "user/codespaces/%s/start" % name, {}, token=token, timeout=180)
    return _wait_available(name, token, budget)


def stop(name, token):
    try:
        _api("POST", "user/codespaces/%s/stop" % name, {}, token=token, timeout=120)
        log.info("codespace %s stopped", name)
    except Exception as e:                     # never let cleanup mask the real error
        log.warning("could not stop codespace %s: %s", name, str(e)[:200])


def _write_key(private_key):
    if not private_key:
        return
    os.makedirs(os.path.dirname(SSH_KEY_PATH), exist_ok=True)
    if not os.path.exists(SSH_KEY_PATH):
        with open(SSH_KEY_PATH, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(private_key if private_key.endswith("\n") else private_key + "\n")
        os.chmod(SSH_KEY_PATH, stat.S_IRUSR | stat.S_IWUSR)
    # gh only accepts a pair, and reads the public half from <key>.pub.
    pub_path = SSH_KEY_PATH + ".pub"
    if os.path.exists(pub_path) or not shutil.which("ssh-keygen"):
        return
    r = subprocess.run(["ssh-keygen", "-y", "-f", SSH_KEY_PATH],
                       capture_output=True, text=True, env=_env())
    if r.returncode == 0 and (r.stdout or "").strip():
        with open(pub_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(r.stdout.strip() + "\n")


def public_key(private_key):
    """The half that has to exist on the account once (Settings -> SSH keys)."""
    if not private_key:
        return ""
    _write_key(private_key)
    try:
        with open(SSH_KEY_PATH + ".pub", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


class Runner:
    def __init__(self, name, token, workdir, max_chars, timeout):
        self.name = name
        self.token = token
        self.workdir = workdir
        self.max_chars = max_chars
        self.timeout = timeout

    def run(self, command):
        """Run one shell command inside the codespace, in this task's own directory.

        The command gets an empty environment (env -i) and a neutral HOME, so the codespace's
        own GITHUB_TOKEN and git credentials are not visible to anything the AI asks for.
        """
        wd = shlex.quote(self.workdir)
        remote = ("mkdir -p %s && cd %s && env -i HOME=/tmp PATH=/usr/local/bin:/usr/bin:/bin "
                  "LANG=C.UTF-8 sh -c %s") % (wd, wd, shlex.quote(command))
        r = _run(["codespace", "ssh", "-c", self.name, "--", "sh", "-lc", remote],
                 token=self.token, timeout=self.timeout)
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        text = out if out else err
        if len(text) > self.max_chars:
            text = text[:self.max_chars] + "\n…[output truncated]"
        if r.returncode != 0 and not text:
            text = "(exit %s, no output)" % r.returncode
        return text

    def read_file(self, path, max_bytes):
        """Raw bytes of one file from the task directory, for sharing. Unlike run(),
        the payload is not truncated at max_chars, so the caller caps the size first."""
        wd = shlex.quote(self.workdir)
        remote = ("cd %s && f=%s && test -f \"$f\" || exit 3; "
                  "r=$(realpath -- \"$f\") && case \"$r\" in \"$PWD\"/*) ;; *) exit 5;; esac; "
                  "s=$(wc -c < \"$f\") && test \"$s\" -le %d || exit 4; "
                  "base64 -w0 -- \"$f\"") % (wd, shlex.quote(path), int(max_bytes))
        r = _run(["codespace", "ssh", "-c", self.name, "--", "sh", "-lc", remote],
                 token=self.token, timeout=self.timeout)
        if r.returncode == 3:
            raise CodespaceError("file not found in the task folder: %s" % path)
        if r.returncode == 4:
            raise CodespaceError("file is larger than the %d byte share limit" % max_bytes)
        if r.returncode == 5:
            raise CodespaceError("file resolves outside the task folder: %s" % path)
        out = (r.stdout or "").strip()
        if r.returncode != 0 or not out:
            raise CodespaceError((r.stderr or "file could not be read").strip()[:300])
        try:
            return base64.b64decode(out)
        except ValueError:
            raise CodespaceError("file payload was not valid base64")


class _Session:
    def __init__(self, label, token, repo, machine, idle_min, budget,
                 workdir, max_chars, timeout, ssh_key):
        self.label = label
        self.token = token
        self.repo = repo
        self.machine = machine
        self.idle_min = idle_min
        self.budget = budget
        self.workdir = workdir
        self.max_chars = max_chars
        self.timeout = timeout
        self.ssh_key = ssh_key
        self.name = None

    def __enter__(self):
        _write_key(self.ssh_key)
        self.name = _ensure(self.label, self.repo, self.machine, self.idle_min, self.token)
        start(self.name, self.token, self.budget)
        return Runner(self.name, self.token, self.workdir, self.max_chars, self.timeout)

    def __exit__(self, *exc):
        if self.name:
            stop(self.name, self.token)
        return False


def session(label, *, token="", repo="", machine="basicLinux32gb", idle_min=10,
            start_budget=300, workdir="/workspaces", max_chars=3000, timeout=90,
            ssh_key=""):
    return _Session(label, token, repo, machine, idle_min, start_budget,
                    workdir, max_chars, timeout, ssh_key)
