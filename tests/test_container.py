# tests/test_container.py
from pathlib import Path
from autoqc.bundle import load_bundle
from autoqc.agent.container import (
    RunResult, docker_available, image_tag, Limits, ContainerError, ContainerSession)


def _bundle(tmp_path: Path):
    (tmp_path / "environment").mkdir(parents=True)
    (tmp_path / "environment/Dockerfile").write_text("FROM busybox\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/rubrics.json").write_text("[]")
    (tmp_path / "task.toml").write_text('schema_version="1.1"\n')
    return load_bundle(tmp_path)


class FakeRunner:
    """Records argv; returns queued RunResults (or a default success)."""
    def __init__(self, results=None):
        self.calls = []
        self._results = list(results or [])
    def __call__(self, argv, timeout=None):
        self.calls.append((argv, timeout))
        return self._results.pop(0) if self._results else RunResult(0, "")


def test_image_tag_is_deterministic_and_hashed(tmp_path):
    b = _bundle(tmp_path)
    tag = image_tag(b)
    assert tag.startswith("autoqc-q06/")
    assert ":" in tag
    assert image_tag(b) == tag  # stable
    (tmp_path / "environment/Dockerfile").write_text("FROM busybox\nRUN true\n")
    assert image_tag(load_bundle(tmp_path)) != tag  # hash changes with Dockerfile


def test_docker_available_reflects_returncode():
    assert docker_available(FakeRunner([RunResult(0, "ok")])) is True
    assert docker_available(FakeRunner([RunResult(1, "cannot connect")])) is False


def test_ensure_image_skips_build_on_cache_hit(tmp_path):
    r = FakeRunner([RunResult(0, "exists")])  # docker image inspect -> hit
    s = ContainerSession(_bundle(tmp_path), runner=r)
    assert s.ensure_image() == s.tag
    assert r.calls[0][0][:3] == ["docker", "image", "inspect"]
    assert not any(c[0][:2] == ["docker", "build"] for c in r.calls)


def test_ensure_image_builds_on_miss(tmp_path):
    r = FakeRunner([RunResult(1, "no such image"), RunResult(0, "built")])
    s = ContainerSession(_bundle(tmp_path), runner=r)
    assert s.ensure_image() == s.tag
    build = [c for c in r.calls if c[0][:2] == ["docker", "build"]][0][0]
    assert "-t" in build and s.tag in build and "-f" in build


def test_ensure_image_raises_on_build_failure(tmp_path):
    r = FakeRunner([RunResult(1, "miss"), RunResult(1, "build error: boom")])
    s = ContainerSession(_bundle(tmp_path), runner=r)
    try:
        s.ensure_image()
        assert False, "expected ContainerError"
    except ContainerError as e:
        assert "boom" in str(e)


def test_start_uses_hardening_flags(tmp_path):
    r = FakeRunner([RunResult(0, "cid")])
    s = ContainerSession(_bundle(tmp_path), runner=r, name="c1")
    assert s.start() == "c1"
    argv = r.calls[0][0]
    for flag in ["--network=none", "--cap-drop=ALL", "--read-only",
                 "--security-opt", "no-new-privileges", "--pids-limit"]:
        assert flag in argv, f"missing {flag}"
    assert "-w" in argv and "/testbed" in argv
    tmpfs = argv[argv.index("--tmpfs") + 1]
    assert tmpfs.startswith("/scratch:")


def test_start_raises_on_failure(tmp_path):
    s = ContainerSession(_bundle(tmp_path), runner=FakeRunner([RunResult(1, "denied")]), name="c1")
    try:
        s.start(); assert False
    except ContainerError as e:
        assert "denied" in str(e)


def test_exec_sets_offline_caches_and_caps_output(tmp_path):
    long = "x" * 50000
    r = FakeRunner([RunResult(0, long)])
    s = ContainerSession(_bundle(tmp_path), runner=r, name="c1", limits=Limits(output_cap=100))
    s.container_id = "c1"
    out = s.exec("go build ./...")
    argv = r.calls[0][0]
    assert argv[:3] == ["docker", "exec", "-w"]
    assert "GOCACHE=/scratch/go-build" in argv and "TMPDIR=/scratch/tmp" in argv
    assert not any("GOMODCACHE" in a for a in argv)  # baked module cache untouched
    assert argv[-3:] == ["bash", "-lc", "go build ./..."]
    assert len(out) < 50000 and "truncated" in out


def test_exec_timeout_surfaces_as_output(tmp_path):
    r = FakeRunner([RunResult(124, "error: command timed out after 5s")])
    s = ContainerSession(_bundle(tmp_path), runner=r, name="c1", limits=Limits(exec_timeout=5))
    s.container_id = "c1"
    assert "timed out" in s.exec("sleep 999")


def test_stop_is_best_effort(tmp_path):
    r = FakeRunner()
    s = ContainerSession(_bundle(tmp_path), runner=r, name="c1")
    s.container_id = "c1"
    s.stop()  # must not raise
    assert any("c1" in " ".join(c[0]) for c in r.calls)
