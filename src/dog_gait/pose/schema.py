"""Quadruped keypoint schema.

The public bodypart set is dog-only. The full quadruped ordering is retained
internally so compatible pretrained checkpoints can be adapted.
"""

from __future__ import annotations

QUADRUPED_BODYPARTS = [
    "nose",
    "upper_jaw",
    "lower_jaw",
    "mouth_end_right",
    "mouth_end_left",
    "right_eye",
    "right_earbase",
    "right_earend",
    "right_antler_base",
    "right_antler_end",
    "left_eye",
    "left_earbase",
    "left_earend",
    "left_antler_base",
    "left_antler_end",
    "neck_base",
    "neck_end",
    "throat_base",
    "throat_end",
    "back_base",
    "back_end",
    "back_middle",
    "tail_base",
    "tail_end",
    "front_left_thigh",
    "front_left_knee",
    "front_left_paw",
    "front_right_thigh",
    "front_right_knee",
    "front_right_paw",
    "back_left_paw",
    "back_left_thigh",
    "back_right_thigh",
    "back_left_knee",
    "back_right_knee",
    "back_right_paw",
    "belly_bottom",
    "body_middle_right",
    "body_middle_left",
]

DOG_EXCLUDED_BODYPARTS = {
    "right_antler_base",
    "right_antler_end",
    "left_antler_base",
    "left_antler_end",
}

BODYPARTS = [bodypart for bodypart in QUADRUPED_BODYPARTS if bodypart not in DOG_EXCLUDED_BODYPARTS]

CHECKPOINT_BODYPARTS = QUADRUPED_BODYPARTS
CHECKPOINT_TO_DOG_INDICES = tuple(CHECKPOINT_BODYPARTS.index(bodypart) for bodypart in BODYPARTS)

BODYPART_DISPLAY_NAMES = {bodypart: bodypart.replace("_", " ").title() for bodypart in BODYPARTS}

BODYPART_GROUPS = {
    "Head and jaw": (
        "nose",
        "upper_jaw",
        "lower_jaw",
        "mouth_end_right",
        "mouth_end_left",
    ),
    "Eyes and ears": (
        "right_eye",
        "left_eye",
        "right_earbase",
        "right_earend",
        "left_earbase",
        "left_earend",
    ),
    "Neck and torso": (
        "neck_base",
        "neck_end",
        "throat_base",
        "throat_end",
        "back_base",
        "back_middle",
        "back_end",
        "belly_bottom",
        "body_middle_right",
        "body_middle_left",
    ),
    "Tail": ("tail_base", "tail_end"),
    "Front left leg": ("front_left_thigh", "front_left_knee", "front_left_paw"),
    "Front right leg": ("front_right_thigh", "front_right_knee", "front_right_paw"),
    "Back left leg": ("back_left_thigh", "back_left_knee", "back_left_paw"),
    "Back right leg": ("back_right_thigh", "back_right_knee", "back_right_paw"),
}

BODY_INDEX = {name: idx for idx, name in enumerate(BODYPARTS)}

SKELETON = [
    ("nose", "neck_base"),
    ("neck_base", "back_middle"),
    ("back_middle", "tail_base"),
    ("tail_base", "tail_end"),
    ("neck_base", "front_left_thigh"),
    ("front_left_thigh", "front_left_knee"),
    ("front_left_knee", "front_left_paw"),
    ("neck_base", "front_right_thigh"),
    ("front_right_thigh", "front_right_knee"),
    ("front_right_knee", "front_right_paw"),
    ("back_middle", "back_left_thigh"),
    ("back_left_thigh", "back_left_knee"),
    ("back_left_knee", "back_left_paw"),
    ("back_middle", "back_right_thigh"),
    ("back_right_thigh", "back_right_knee"),
    ("back_right_knee", "back_right_paw"),
]

LIMB_TRIPLETS = {
    "front_left": ("front_left_thigh", "front_left_knee", "front_left_paw"),
    "front_right": ("front_right_thigh", "front_right_knee", "front_right_paw"),
    "back_left": ("back_left_thigh", "back_left_knee", "back_left_paw"),
    "back_right": ("back_right_thigh", "back_right_knee", "back_right_paw"),
}
