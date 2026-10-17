"""Converted checkpoints hold only the backbone; the head and processing files come from the release."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clef_mlx import model


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    """Stand in for the HF hub: every download lands in one fake snapshot dir and is recorded."""
    snapshot = tmp_path / "hub" / "snapshots" / "17f0b0ad"
    snapshot.mkdir(parents=True)
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(snapshot)

    monkeypatch.setattr(model, "snapshot_download", snapshot_download)
    return snapshot, calls


def backbone_dir(root: Path, pointer: dict) -> Path:
    path = root / "clef-flash-8bit"
    path.mkdir()
    (path / "config.json").write_text("{}")
    (path / model.RELEASE_POINTER).write_text(json.dumps(pointer))
    return path


def test_a_release_is_its_own_processing_source(tmp_path):
    assert model.release_dir(tmp_path) == tmp_path


def test_a_converted_backbone_fetches_only_processing_files_from_its_hub_release(tmp_path, downloads):
    snapshot, calls = downloads
    path = backbone_dir(tmp_path, {"release": "Cloudflare/clef-flash", "revision": "17f0b0ad"})
    assert model.release_dir(path) == snapshot
    [(repo, kwargs)] = calls
    assert repo == "Cloudflare/clef-flash"
    assert kwargs["revision"] == "17f0b0ad"
    assert {"joint_head.safetensors", "joint_head_config.json", "tokenizer.json"} <= set(kwargs["allow_patterns"])
    assert not any("model" in pattern for pattern in kwargs["allow_patterns"])


def test_a_converted_backbone_can_point_at_a_local_release(tmp_path, downloads):
    release = tmp_path / "release"
    release.mkdir()
    path = backbone_dir(tmp_path, {"release": str(release)})
    assert model.release_dir(path) == release
    assert downloads[1] == []


def fake_mlx_lm_convert(monkeypatch):
    def convert(hf_path, mlx_path, **kwargs):
        out = Path(mlx_path)
        out.mkdir()
        for name in ("config.json", "model.safetensors", "model.safetensors.index.json", "tokenizer.json",
                     "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "README.md"):
            (out / name).write_text("x")

    monkeypatch.setattr("mlx_lm.convert", convert)


def test_convert_keeps_only_the_backbone_and_points_at_the_hub_release(monkeypatch, tmp_path, downloads):
    fake_mlx_lm_convert(monkeypatch)
    out = model.convert("Cloudflare/clef-flash", tmp_path / "clef-flash-8bit", vision=False)
    assert sorted(p.name for p in out.iterdir()) == [
        model.RELEASE_POINTER, "config.json", "model.safetensors", "model.safetensors.index.json"
    ]
    pointer = json.loads((out / model.RELEASE_POINTER).read_text())
    assert pointer == {"release": "Cloudflare/clef-flash", "revision": "17f0b0ad"}


def test_convert_from_a_local_release_points_at_its_absolute_path(monkeypatch, tmp_path, downloads):
    fake_mlx_lm_convert(monkeypatch)
    release = tmp_path / "release"
    release.mkdir()
    monkeypatch.chdir(tmp_path)
    out = model.convert("release", tmp_path / "clef-flash-8bit", vision=False)
    assert json.loads((out / model.RELEASE_POINTER).read_text()) == {"release": str(release.resolve())}
