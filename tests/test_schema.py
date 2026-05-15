from dog_gait.pose.schema import BODYPARTS, BODYPART_DISPLAY_NAMES, BODYPART_GROUPS, DOG_EXCLUDED_BODYPARTS, QUADRUPED_BODYPARTS


def test_bodyparts_use_thigh_spelling():
    misspelling = "tha" + "i"

    assert all(misspelling not in bodypart for bodypart in BODYPARTS)
    assert {"front_left_thigh", "front_right_thigh", "back_left_thigh", "back_right_thigh"} <= set(BODYPARTS)


def test_bodyparts_exclude_non_dog_quadruped_parts():
    assert DOG_EXCLUDED_BODYPARTS == {
        "right_antler_base",
        "right_antler_end",
        "left_antler_base",
        "left_antler_end",
    }
    assert DOG_EXCLUDED_BODYPARTS <= set(QUADRUPED_BODYPARTS)
    assert DOG_EXCLUDED_BODYPARTS.isdisjoint(BODYPARTS)


def test_bodypart_groups_cover_each_bodypart_once():
    grouped = [bodypart for group in BODYPART_GROUPS.values() for bodypart in group]

    assert set(grouped) == set(BODYPARTS)
    assert len(grouped) == len(set(grouped))


def test_display_names_distinguish_each_limb_side():
    assert BODYPART_DISPLAY_NAMES["front_left_thigh"] == "Front Left Thigh"
    assert BODYPART_DISPLAY_NAMES["front_right_thigh"] == "Front Right Thigh"
    assert BODYPART_DISPLAY_NAMES["back_left_thigh"] == "Back Left Thigh"
    assert BODYPART_DISPLAY_NAMES["back_right_thigh"] == "Back Right Thigh"
