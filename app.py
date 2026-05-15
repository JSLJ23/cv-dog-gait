from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PIL import Image, ImageDraw
import plotly.express as px
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import torch

from dog_gait.analysis.angles import compute_limb_angles, smooth_angles
from dog_gait.analysis.overlay import render_overlay_videos
from dog_gait.data.annotations import AnnotationStore, FineTuneDatasetStore, safe_dataset_name
from dog_gait.data.finetune import FineTuner, fine_tuned_checkpoint_path, safe_checkpoint_name
from dog_gait.data.frames import extract_frames
from dog_gait.pose.checkpoints import (
    CheckpointSpec,
    dog_gait_default_checkpoints,
    download_checkpoint,
    is_hrnet_pose_checkpoint_path,
    local_checkpoint_specs,
)
from dog_gait.pose.estimator import PoseEstimator
from dog_gait.pose.schema import BODYPARTS, BODYPART_DISPLAY_NAMES, BODYPART_GROUPS


st.set_page_config(page_title="Dog Gait Analysis", page_icon="DG", layout="wide")

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
BROWSER_OUTPUT_DIR = OUTPUT_DIR / "browser"
DOWNLOAD_OUTPUT_DIR = OUTPUT_DIR / "downloads"
ANNOTATION_DIR = DATA_DIR / "annotations"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
DEFAULT_CHECKPOINT_DIR = DATA_DIR / "default_checkpoints"
for path in (UPLOAD_DIR, OUTPUT_DIR, BROWSER_OUTPUT_DIR, DOWNLOAD_OUTPUT_DIR, ANNOTATION_DIR, CHECKPOINT_DIR, DEFAULT_CHECKPOINT_DIR):
    path.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_estimator(device: str) -> PoseEstimator:
    return PoseEstimator(device=device)


def save_upload(uploaded_file) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in " ._-" else "_" for ch in Path(uploaded_file.name).name).strip()
    target = UPLOAD_DIR / (safe_name or "uploaded_video.mp4")
    target.write_bytes(uploaded_file.getbuffer())
    return target


def browser_overlay_output_path(video_path: Path) -> Path:
    return BROWSER_OUTPUT_DIR / f"{video_path.stem}_overlay.webm"


def download_overlay_output_path(video_path: Path) -> Path:
    return DOWNLOAD_OUTPUT_DIR / f"{video_path.stem}_overlay.mov"


def video_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".webm":
        return "video/webm"
    if suffix == ".mov":
        return "video/quicktime"
    return "video/mp4"


def discover_checkpoints() -> list[CheckpointSpec]:
    return dog_gait_default_checkpoints(DEFAULT_CHECKPOINT_DIR) + local_checkpoint_specs([CHECKPOINT_DIR])


def discover_pose_checkpoints() -> list[CheckpointSpec]:
    return [
        spec
        for spec in discover_checkpoints()
        if spec.fine_tunable and spec.path is not None and is_hrnet_pose_checkpoint_path(spec.path)
    ]


def checkpoint_label(spec: CheckpointSpec | None) -> str:
    if spec is None:
        return "No pose checkpoint"
    status = "downloaded" if spec.exists else "not downloaded"
    if spec.model_name is None:
        status = "local"
    return f"{spec.label} ({status})"


def default_pose_checkpoint_index(options: list[CheckpointSpec]) -> int:
    for index, spec in enumerate(options):
        if isinstance(spec, CheckpointSpec) and spec.fine_tunable and spec.exists:
            return index
    return 0


def frame_number_from_path(frame_path: str | Path) -> int:
    return int(Path(frame_path).stem.split("_")[-1])


def clamp_frame_index(index: int, frames: list[str]) -> int:
    if not frames:
        return 0
    return max(0, min(index, len(frames) - 1))


def annotated_frame_image(image_path: str | Path, labels: dict[str, dict[str, float]], active_bodypart: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for bodypart, point in labels.items():
        x = float(point["x"])
        y = float(point["y"])
        radius = 8 if bodypart == active_bodypart else 6
        fill = "#ff3b30" if bodypart == active_bodypart else "#00a676"
        draw.ellipse((x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2), fill="#111111")
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
        if bodypart == active_bodypart:
            draw.text((x + radius + 5, y - radius - 5), BODYPART_DISPLAY_NAMES.get(bodypart, bodypart), fill="#ff3b30")
    return image


def original_image_point(click: dict | None, image_size: tuple[int, int]) -> tuple[int, int] | None:
    if not click or "x" not in click or "y" not in click:
        return None
    width, height = image_size
    displayed_width = max(float(click.get("width", width)), 1.0)
    displayed_height = max(float(click.get("height", height)), 1.0)
    x = round(float(click["x"]) * width / displayed_width)
    y = round(float(click["y"]) * height / displayed_height)
    return max(0, min(int(x), width - 1)), max(0, min(int(y), height - 1))

st.title("Dog Gait Analysis")
st.caption("Video-based dog gait intelligence for pose estimation, annotation, and fine-tuning.")

tab_analysis, tab_annotation, tab_train = st.tabs(["Video analysis", "Annotation", "Fine-tuning"])

with tab_analysis:
    analysis_menu, analysis_window = st.columns([0.32, 0.68])
    with analysis_menu:
        uploaded = st.file_uploader("Upload dog gait video", type=["mp4", "mov", "avi", "mkv"])
        device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        device = st.selectbox(
            "Inference device",
            device_options,
            index=0,
            help="CUDA is selected by default when PyTorch can see the GPU.",
        )
        if device == "cuda":
            st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
        checkpoint_options = discover_pose_checkpoints()
        analysis_checkpoint = st.selectbox(
            "Pose checkpoint",
            checkpoint_options,
            index=default_pose_checkpoint_index(checkpoint_options),
            format_func=checkpoint_label,
            help="Use the downloaded quadruped HRNet-W32 pose checkpoint for pose inference.",
        )
        confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.2, 0.05)
        crop_mode = st.selectbox("Crop mode", ["full-frame", "dog-detector"])
        dog_facing = st.selectbox("Dog facing direction", ["right", "left"])
        max_frames = st.number_input("Max frames for analysis run", min_value=1, max_value=5000, value=240, step=10)
        default_batch_size = 16 if device == "cuda" else 4
        batch_size = st.number_input(
            "Inference batch size",
            min_value=1,
            max_value=64,
            value=default_batch_size,
            step=1,
            help="Higher values improve GPU throughput but use more VRAM.",
        )
        if isinstance(analysis_checkpoint, CheckpointSpec) and analysis_checkpoint.model_name is not None and not analysis_checkpoint.exists:
            if st.button("Download selected pose checkpoint", key="download_analysis_checkpoint"):
                try:
                    with st.spinner(f"Downloading {analysis_checkpoint.label}..."):
                        download_checkpoint(analysis_checkpoint)
                    st.success(f"Downloaded to {analysis_checkpoint.path}")
                except Exception as exc:
                    st.error(f"Download failed: {exc}")
        analysis_checkpoint_ready = isinstance(analysis_checkpoint, CheckpointSpec) and analysis_checkpoint.exists
        run = st.button("Run analysis", type="primary", disabled=uploaded is None or not analysis_checkpoint_ready)

    with analysis_window:
        if run and uploaded is not None:
            video_path = save_upload(uploaded)
            with st.status("Running pose inference...", expanded=True) as status:
                estimator = get_estimator(device)
                st.write(f"Inference device: `{estimator.device}`")
                st.write(f"Inference batch size: `{int(batch_size)}`")
                pose_progress = st.progress(0.0, text="Pose detection: starting")

                def update_pose_progress(processed_frames: int, total_frames: int | None) -> None:
                    if total_frames and total_frames > 0:
                        fraction = min(processed_frames / total_frames, 1.0)
                        pose_progress.progress(fraction, text=f"Pose detection: {processed_frames} / {total_frames} frames")
                    else:
                        pose_progress.progress(0.0, text=f"Pose detection: {processed_frames} frames")

                result = estimator.predict_video(
                    video_path,
                    checkpoint_path=str(analysis_checkpoint.path),
                    confidence=confidence,
                    crop_mode=crop_mode,
                    max_frames=int(max_frames),
                    dog_facing=dog_facing,
                    batch_size=int(batch_size),
                    progress_callback=update_pose_progress,
                )
                pose_progress.progress(1.0, text=f"Pose detection: {result.frame_count} frames complete")

                status.update(label="Computing gait angles...", state="running")
                angle_progress = st.progress(0.0, text="Angle analysis: starting")
                angles = smooth_angles(compute_limb_angles(result.keypoints, confidence), window=5)
                angle_progress.progress(1.0, text="Angle analysis: complete")

                status.update(label="Rendering pose overlay videos...", state="running")
                overlay_progress = st.progress(0.0, text="Overlay rendering: starting")

                def update_overlay_progress(processed_frames: int, total_frames: int | None) -> None:
                    if total_frames and total_frames > 0:
                        fraction = min(processed_frames / total_frames, 1.0)
                        if processed_frames >= total_frames:
                            overlay_progress.progress(fraction, text="Overlay rendering: finalizing video files")
                        else:
                            overlay_progress.progress(fraction, text=f"Overlay rendering: {processed_frames} / {total_frames} frames")
                    else:
                        overlay_progress.progress(0.0, text=f"Overlay rendering: {processed_frames} frames")

                def update_overlay_stage(stage: str) -> None:
                    labels = {
                        "opening-writers": "Preparing labelled video writers...",
                        "rendering": "Rendering pose overlay videos...",
                        "finalizing": "Finalizing labelled videos...",
                        "complete": "Labelled videos ready",
                    }
                    status.update(label=labels.get(stage, "Rendering pose overlay videos..."), state="running")
                    if stage == "finalizing":
                        overlay_progress.progress(1.0, text="Overlay rendering: finalizing video files")

                browser_overlay_path = browser_overlay_output_path(video_path)
                download_overlay_path = download_overlay_output_path(video_path)
                render_overlay_videos(
                    video_path,
                    result.keypoints,
                    [browser_overlay_path, download_overlay_path],
                    confidence,
                    progress_callback=update_overlay_progress,
                    stage_callback=update_overlay_stage,
                )
                overlay_progress.progress(1.0, text="Overlay rendering: complete")
                status.update(label="Analysis complete", state="complete")

            st.session_state["video_path"] = str(video_path)
            st.session_state["keypoints"] = result.keypoints
            st.session_state["angles"] = angles
            st.session_state["browser_overlay_path"] = str(browser_overlay_path)
            st.session_state["download_overlay_path"] = str(download_overlay_path)
            st.success(f"Processed {result.frame_count} frames using {result.inference_source} on {result.device}.")

        angles = st.session_state.get("angles")
        video_path = st.session_state.get("video_path")
        if angles is not None and video_path:
            browser_overlay_path = st.session_state.get("browser_overlay_path") or st.session_state.get("overlay_path")
            download_overlay_path = st.session_state.get("download_overlay_path")
            if browser_overlay_path and Path(browser_overlay_path).exists():
                st.subheader("Pose overlay video")
                browser_overlay_file = Path(browser_overlay_path)
                st.video(browser_overlay_file.read_bytes(), format=video_format(browser_overlay_file))
                if download_overlay_path and Path(download_overlay_path).exists():
                    download_overlay_file = Path(download_overlay_path)
                    st.download_button(
                        "Download MOV overlay",
                        data=download_overlay_file.read_bytes(),
                        file_name=download_overlay_file.name,
                        mime=video_format(download_overlay_file),
                    )

            st.subheader("Limb angles")
            fig = px.line(angles, x="frame", y="angle_smooth", color="limb", labels={"angle_smooth": "angle"})
            fig.update_layout(yaxis_title="Angle (degrees)", xaxis_title="Frame", legend_title_text="Limb")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Upload a video and run analysis to generate a pose-overlay video and limb-angle graph.")

with tab_annotation:
    st.subheader("Create a fine-tuning dataset")
    dataset_store = FineTuneDatasetStore(ANNOTATION_DIR)
    dataset_name_raw = st.text_input("Dataset name", value="dog_gait_dataset")
    try:
        dataset_name = safe_dataset_name(dataset_name_raw)
        st.caption(f"Dataset folder: `data/annotations/{dataset_name}`")
    except ValueError as exc:
        dataset_name = None
        st.error(str(exc))

    setup_left, setup_mid, setup_right = st.columns([0.42, 0.29, 0.29])
    uploaded_annot = setup_left.file_uploader("Upload video for frame extraction", type=["mp4", "mov", "avi", "mkv"], key="ann")
    every_n = setup_mid.number_input("Extract every N frames", min_value=1, max_value=300, value=30)
    max_extract = setup_right.number_input("Maximum extracted frames", min_value=1, max_value=200, value=24)
    if st.button("Extract frames into dataset", disabled=uploaded_annot is None or dataset_name is None):
        video_path = save_upload(uploaded_annot)
        frames = extract_frames(
            video_path,
            dataset_store.frames_dir(dataset_name),
            every_n=int(every_n),
            max_frames=int(max_extract),
        )
        st.session_state["annotation_dataset"] = dataset_name
        st.session_state["extracted_frames"] = [str(p) for p in frames]
        st.session_state["annotation_frame_index"] = 0
        st.session_state["active_bodypart"] = BODYPARTS[0]
        st.success(f"Extracted {len(frames)} frames into `{dataset_name}`.")

    frames = st.session_state.get("extracted_frames", [])
    active_dataset = st.session_state.get("annotation_dataset", dataset_name)
    if frames and active_dataset:
        st.session_state.setdefault("annotation_frame_index", 0)
        st.session_state.setdefault("active_bodypart", BODYPARTS[0])
        st.session_state["annotation_frame_index"] = clamp_frame_index(st.session_state["annotation_frame_index"], frames)

        def step_annotation_frame(delta: int) -> None:
            st.session_state["annotation_frame_index"] = clamp_frame_index(st.session_state["annotation_frame_index"] + delta, frames)

        frame_back, frame_slider_col, frame_next = st.columns([0.08, 0.84, 0.08])
        frame_back.button("<-", use_container_width=True, on_click=step_annotation_frame, args=(-1,), disabled=st.session_state["annotation_frame_index"] == 0)
        frame_next.button("->", use_container_width=True, on_click=step_annotation_frame, args=(1,), disabled=st.session_state["annotation_frame_index"] >= len(frames) - 1)
        frame_index = frame_slider_col.slider(
            "Frame",
            min_value=0,
            max_value=len(frames) - 1,
            key="annotation_frame_index",
            format="%d",
        )

        selected = frames[frame_index]
        frame_number = frame_number_from_path(selected)
        labels = dataset_store.load_frame_labels(active_dataset, selected, frame_number)
        active_bodypart = st.session_state.get("active_bodypart", BODYPARTS[0])

        selector_col, image_col = st.columns([0.24, 0.76], gap="small")
        with selector_col:
            st.markdown("**Point Selector**")
            st.caption(f"{len(labels)} / {len(BODYPARTS)} annotated")
            if st.button("Reset frame", use_container_width=True, disabled=not labels):
                dataset_store.reset_frame_labels(active_dataset, selected, frame_number)
                st.rerun()
            for group_name, group_parts in BODYPART_GROUPS.items():
                with st.expander(group_name, expanded=active_bodypart in group_parts):
                    for bodypart in group_parts:
                        if bodypart not in BODYPARTS:
                            continue
                        marker = "[x]" if bodypart in labels else "[ ]"
                        label = f"{marker} {BODYPART_DISPLAY_NAMES.get(bodypart, bodypart)}"
                        if st.button(
                            label,
                            key=f"select_{group_name}_{bodypart}",
                            help=bodypart,
                            type="primary" if bodypart == active_bodypart else "secondary",
                            use_container_width=True,
                        ):
                            st.session_state["active_bodypart"] = bodypart
                            st.rerun()

        with image_col:
            st.markdown(f"**Frame {frame_index + 1} of {len(frames)}**")
            annotated_image = annotated_frame_image(selected, labels, active_bodypart)
            click = streamlit_image_coordinates(
                annotated_image,
                use_column_width="always",
                click_and_drag=False,
                key=f"annotation_image_{active_dataset}_{frame_index}_{active_bodypart}_{len(labels)}",
                cursor="crosshair",
            )
            selected_point = original_image_point(click, annotated_image.size)
            if selected_point is not None:
                x, y = selected_point
                click_key = f"{active_dataset}:{selected}:{active_bodypart}:{x}:{y}:{click.get('unix_time')}"
                if st.session_state.get("last_annotation_click") != click_key:
                    dataset_store.save_frame_labels(
                        active_dataset,
                        selected,
                        frame_number,
                        {active_bodypart: {"x": float(x), "y": float(y), "likelihood": 1.0}},
                    )
                    st.session_state["last_annotation_click"] = click_key
                    st.rerun()
    else:
        st.info("Extract frames into a dataset to begin annotation.")

with tab_train:
    st.subheader("Fine-tune from local annotations")
    dataset_store = FineTuneDatasetStore(ANNOTATION_DIR)
    dataset_names = dataset_store.list_datasets()
    selected_dataset = st.selectbox(
        "Fine-tuning dataset",
        dataset_names,
        disabled=not dataset_names,
        help="Datasets are created in the Annotation tab under data/annotations/<name>.",
    )
    if not dataset_names:
        st.info("No fine-tuning datasets yet. Create one in the Annotation tab first.")

    checkpoint_options = discover_pose_checkpoints()
    base_checkpoint = st.selectbox(
        "Base checkpoint",
        checkpoint_options,
        format_func=checkpoint_label,
        help="Pose checkpoints only. The dog detector checkpoint is used internally for dog detection/cropping.",
    )
    selected_checkpoint_path = base_checkpoint.path if isinstance(base_checkpoint, CheckpointSpec) and base_checkpoint.exists else None
    if isinstance(base_checkpoint, CheckpointSpec):
        if base_checkpoint.model_name is not None and not base_checkpoint.exists:
            if st.button("Download selected pose checkpoint"):
                try:
                    with st.spinner(f"Downloading {base_checkpoint.label}..."):
                        selected_checkpoint_path = download_checkpoint(base_checkpoint)
                    st.success(f"Downloaded to {selected_checkpoint_path}")
                except Exception as exc:
                    st.error(f"Download failed: {exc}")

    epochs = st.selectbox("Epochs", [25, 50, 100, 200], index=1)
    batch_size = st.selectbox("Batch size", [2, 4, 8], index=1, help="Batch size 4 is a conservative default for an 8GB VRAM GPU.")
    lr = st.selectbox("Learning rate", [0.001, 0.0005, 0.0001, 0.00005], index=0, format_func=lambda value: f"{value:g}")
    fine_tuned_name_raw = st.text_input("Fine-tuned checkpoint name", value=selected_dataset or "dog_gait")
    try:
        fine_tuned_name = safe_checkpoint_name(fine_tuned_name_raw)
        preview_base = base_checkpoint.path if isinstance(base_checkpoint, CheckpointSpec) else None
        preview_path = fine_tuned_checkpoint_path(CHECKPOINT_DIR, preview_base, fine_tuned_name)
        st.caption(f"Checkpoint will be saved as `{preview_path}`")
    except ValueError as exc:
        fine_tuned_name = None
        st.error(str(exc))

    can_train = bool(dataset_names) and fine_tuned_name is not None and isinstance(base_checkpoint, CheckpointSpec) and base_checkpoint.exists
    if st.button("Start fine-tuning", disabled=not can_train):
        try:
            if isinstance(base_checkpoint, CheckpointSpec) and not base_checkpoint.exists:
                raise FileNotFoundError("Download the selected pose checkpoint before fine-tuning.")
            tuner = FineTuner()
            result = tuner.train(
                dataset_store.dataset_dir(selected_dataset),
                str(selected_checkpoint_path),
                int(epochs),
                int(batch_size),
                float(lr),
                fine_tuned_name=fine_tuned_name or "fine_tuned",
            )
            st.success(f"Saved checkpoint: {result.checkpoint_path}")
            st.line_chart(result.history.set_index("epoch"))
        except Exception as exc:
            st.error(str(exc))
