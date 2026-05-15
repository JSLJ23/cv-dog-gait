"""Dog-relevant checkpoint registry and downloader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


POSE_CHECKPOINT_EXTENSION = ".pt"
POSE_MODEL_NAME_TOKEN = "hrnet"
QUADRUPED_REPO_ID = "mwmathis/" + "Deep" + "Lab" + "CutModelZoo-SuperAnimal-Quadruped"


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    model_name: str | None
    path: Path | None
    kind: str = "pose"

    @property
    def exists(self) -> bool:
        return self.path is not None and self.path.exists()

    @property
    def fine_tunable(self) -> bool:
        return self.kind == "pose"


def dog_gait_default_checkpoints(root: str | Path = "data/default_checkpoints") -> list[CheckpointSpec]:
    """Return default weights relevant to side-view dog gait analysis."""

    root = Path(root)
    return [
        CheckpointSpec(
            label="Quadruped HRNet-W32 pose",
            model_name="superanimal_quadruped_hrnet_w32",
            path=root / "superanimal_quadruped_hrnet_w32.pt",
            kind="pose",
        ),
        CheckpointSpec(
            label="Quadruped Faster R-CNN detector",
            model_name="superanimal_quadruped_fasterrcnn_resnet50_fpn_v2",
            path=root / "superanimal_quadruped_fasterrcnn_resnet50_fpn_v2.pt",
            kind="detector",
        ),
    ]


def local_checkpoint_specs(search_roots: list[str | Path]) -> list[CheckpointSpec]:
    found: dict[str, CheckpointSpec] = {}
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and is_hrnet_pose_checkpoint_path(path):
                resolved = path.resolve()
                found[str(resolved)] = CheckpointSpec(
                    label=f"Local: {path.name}",
                    model_name=None,
                    path=resolved,
                    kind="pose",
                )
    return sorted(found.values(), key=lambda spec: str(spec.path))


def is_hrnet_pose_checkpoint_path(path: str | Path) -> bool:
    """Return true for locally selectable HRNet pose checkpoints.

    Local checkpoint discovery is intentionally filename-based. Fine-tuned
    pose checkpoints keep the HRNet base stem and append a user-provided name,
    while detector checkpoints use names like Faster R-CNN and are not offered
    as pose model choices.
    """

    path = Path(path)
    return path.suffix.lower() == POSE_CHECKPOINT_EXTENSION and POSE_MODEL_NAME_TOKEN in path.stem.lower()


def download_checkpoint(spec: CheckpointSpec) -> Path:
    if spec.model_name is None or spec.path is None:
        raise ValueError("Only registered model-zoo checkpoints can be downloaded.")
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    if spec.path.exists():
        return spec.path

    from huggingface_hub import hf_hub_download

    downloaded_path = Path(
        hf_hub_download(
            repo_id=QUADRUPED_REPO_ID,
            filename=f"{spec.model_name}.pt",
            local_dir=str(spec.path.parent),
        )
    )
    if not spec.path.exists():
        shutil.copy2(downloaded_path, spec.path)
    if not spec.path.exists():
        raise FileNotFoundError(f"Expected downloaded checkpoint at {spec.path}")
    return spec.path
