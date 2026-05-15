from dog_gait.data.annotations import AnnotationStore, FineTuneDatasetStore


def test_annotation_store_round_trip(tmp_path):
    store = AnnotationStore(tmp_path)
    store.save_frame_labels("vid", 3, {"nose": {"x": 10, "y": 20, "likelihood": 1}})

    df = store.load("vid")

    assert len(df) == 1
    assert df.iloc[0]["bodypart"] == "nose"
    assert df.iloc[0]["frame"] == 3


def test_annotation_store_filters_non_dog_bodyparts(tmp_path):
    store = AnnotationStore(tmp_path)
    store.save_frame_labels(
        "vid",
        3,
        {
            "nose": {"x": 10, "y": 20, "likelihood": 1},
            "right_antler_base": {"x": 30, "y": 40, "likelihood": 1},
        },
    )

    df = store.load("vid")

    assert df["bodypart"].tolist() == ["nose"]


def test_finetune_dataset_store_creates_named_dataset(tmp_path):
    source_frame = tmp_path / "source.jpg"
    source_frame.write_bytes(b"fake image bytes")
    store = FineTuneDatasetStore(tmp_path / "annotations")

    copied = store.add_frames("walk study", [source_frame])
    saved = store.save_frame_labels(
        "walk study",
        copied[0],
        12,
        {"nose": {"x": 10, "y": 20, "likelihood": 1.0}},
    )

    assert copied[0].exists()
    assert copied[0].parent.name == "frames"
    assert saved == 1
    assert store.list_datasets() == ["walk_study"]
    assert store.annotations_path("walk study").exists()


def test_finetune_dataset_store_loads_and_resets_frame_labels(tmp_path):
    source_frame = tmp_path / "frame_000012.jpg"
    source_frame.write_bytes(b"fake image bytes")
    store = FineTuneDatasetStore(tmp_path / "annotations")
    copied = store.add_frames("walk study", [source_frame])[0]
    store.save_frame_labels(
        "walk study",
        copied,
        12,
        {
            "nose": {"x": 10, "y": 20, "likelihood": 1.0},
            "tail_base": {"x": 40, "y": 50, "likelihood": 1.0},
        },
    )

    labels = store.load_frame_labels("walk study", copied, 12)
    removed = store.reset_frame_labels("walk study", copied, 12)

    assert labels["nose"] == {"x": 10.0, "y": 20.0, "likelihood": 1.0}
    assert labels["tail_base"] == {"x": 40.0, "y": 50.0, "likelihood": 1.0}
    assert removed == 2
    assert store.load_frame_labels("walk study", copied, 12) == {}
    assert store.annotations_path("walk study").exists()


def test_finetune_dataset_store_filters_non_dog_bodyparts(tmp_path):
    source_frame = tmp_path / "frame_000012.jpg"
    source_frame.write_bytes(b"fake image bytes")
    store = FineTuneDatasetStore(tmp_path / "annotations")
    copied = store.add_frames("walk study", [source_frame])[0]

    saved = store.save_frame_labels(
        "walk study",
        copied,
        12,
        {
            "nose": {"x": 10, "y": 20, "likelihood": 1.0},
            "left_antler_end": {"x": 40, "y": 50, "likelihood": 1.0},
        },
    )
    labels = store.load_frame_labels("walk study", copied, 12)

    assert saved == 1
    assert labels == {"nose": {"x": 10.0, "y": 20.0, "likelihood": 1.0}}
    assert store.count_labels("walk study") == 1
