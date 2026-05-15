from pathlib import Path

import pytest

from dog_gait.data.finetune import fine_tuned_checkpoint_path, safe_checkpoint_name
from dog_gait.pose.checkpoints import is_hrnet_pose_checkpoint_path, local_checkpoint_specs


def test_local_checkpoint_specs_only_includes_hrnet_pt_files(tmp_path):
    included = tmp_path / "superanimal_quadruped_hrnet_w32_walk.pt"
    excluded_detector = tmp_path / "superanimal_quadruped_fasterrcnn_resnet50_fpn_v2.pt"
    excluded_extension = tmp_path / "superanimal_quadruped_hrnet_w32.pth"
    excluded_other = tmp_path / "dog_gait_finetuned.pt"
    for path in (included, excluded_detector, excluded_extension, excluded_other):
        path.write_bytes(b"checkpoint")

    specs = local_checkpoint_specs([tmp_path])

    assert [Path(spec.path).name for spec in specs] == [included.name]
    assert is_hrnet_pose_checkpoint_path(included)
    assert not is_hrnet_pose_checkpoint_path(excluded_detector)
    assert not is_hrnet_pose_checkpoint_path(excluded_extension)


def test_fine_tuned_checkpoint_path_appends_user_name_to_base_stem(tmp_path):
    base = tmp_path / "superanimal_quadruped_hrnet_w32.pt"

    path = fine_tuned_checkpoint_path(tmp_path / "out", base, "post op walk")

    assert path == tmp_path / "out" / "superanimal_quadruped_hrnet_w32_post_op_walk.pt"


def test_safe_checkpoint_name_rejects_blank_names():
    with pytest.raises(ValueError, match="Fine-tuned checkpoint name"):
        safe_checkpoint_name("  !!!  ")
