# autoqc/agent/container.py
from __future__ import annotations
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    returncode: int
    output: str


def _subprocess_runner(argv, timeout=None) -> RunResult:
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout, errors="replace")
        return RunResult(p.returncode, (p.stdout or "") + (p.stderr or ""))
    except subprocess.TimeoutExpired:
        return RunResult(124, f"error: command timed out after {timeout}s")
    except (OSError, ValueError) as e:
        return RunResult(127, f"error: could not run {argv[:2]}: {e}")


def docker_available(runner=_subprocess_runner) -> bool:
    return runner(["docker", "version"], timeout=20).returncode == 0


def image_tag(bundle) -> str:
    df = (Path(bundle.root) / "environment/Dockerfile").read_bytes()
    h = hashlib.sha256(df).hexdigest()[:12]
    task = Path(bundle.root).name or "task"
    safe = re.sub(r"[^a-z0-9._-]", "-", task.lower()) or "task"
    return f"autoqc-q06/{safe}:{h}"


class ContainerError(Exception):
    pass


@dataclass
class Limits:
    memory: str = "2g"
    cpus: str = "2"
    pids: int = 256
    scratch_size: str = "512m"
    exec_timeout: float = 120.0
    output_cap: int = 20000
    build_timeout: float = 1800.0
    ttl: str = "2h"


class ContainerSession:
    def __init__(self, bundle, runner=_subprocess_runner, name=None, limits=None):
        self.bundle = bundle
        self.runner = runner
        self.limits = limits or Limits()
        self.tag = image_tag(bundle)
        self.name = name or ("autoqc_q06_" + self.tag.split(":")[-1])
        self.container_id = None

    def ensure_image(self) -> str:
        if self.runner(["docker", "image", "inspect", self.tag], timeout=30).returncode == 0:
            return self.tag
        env = Path(self.bundle.root) / "environment"
        r = self.runner(["docker", "build", "-f", str(env / "Dockerfile"),
                         "-t", self.tag, str(env)], timeout=self.limits.build_timeout)
        if r.returncode != 0:
            raise ContainerError(f"image build failed: {r.output[-800:]}")
        return self.tag

    def start(self) -> str:
        lim = self.limits
        argv = ["docker", "run", "-d", "--name", self.name,
                "--network=none", "--cap-drop=ALL",
                "--security-opt", "no-new-privileges",
                "--read-only", "--tmpfs", f"/scratch:rw,size={lim.scratch_size}",
                "--memory", lim.memory, "--cpus", lim.cpus,
                "--pids-limit", str(lim.pids), "-w", "/testbed",
                self.tag, "sleep", lim.ttl]
        r = self.runner(argv, timeout=60)
        if r.returncode != 0:
            raise ContainerError(f"container start failed: {r.output[-800:]}")
        self.container_id = self.name
        return self.name

    def exec(self, cmd, cwd="/testbed", timeout=None) -> str:
        argv = ["docker", "exec", "-w", cwd,
                "-e", "GOCACHE=/scratch/go-build", "-e", "TMPDIR=/scratch/tmp",
                self.name, "bash", "-lc", cmd]
        out = self.runner(argv, timeout=timeout or self.limits.exec_timeout).output or ""
        cap = self.limits.output_cap
        if len(out) > cap:
            out = out[:cap] + f"\n...[truncated {len(out) - cap} chars]"
        return out

    def stop(self) -> None:
        self.runner(["bash", "-lc",
                     f"(timeout 60 docker stop {self.name} || docker rm -f {self.name}) "
                     f">/dev/null 2>&1 &"], timeout=10)
