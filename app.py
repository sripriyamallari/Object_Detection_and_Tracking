import os
import tempfile
import subprocess
import cv2
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Object Detection and Tracking",
    page_icon="🎯"
)

st.title("🎯 Object Detection and Tracking")
st.write("Detect and track objects using YOLO + ByteTrack.")

@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "📤 Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:

    input_path = os.path.join(
        tempfile.gettempdir(),
        "input_video.mp4"
    )

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ Video uploaded successfully!")

    st.subheader("🎥 Original Video")
    st.video(input_path)

    if st.button("🚀 Detect & Track Objects"):

        raw_output = os.path.join(
            tempfile.gettempdir(),
            "tracked_raw.mp4"
        )

        final_output = os.path.join(
            tempfile.gettempdir(),
            "tracked_output.mp4"
        )

        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 30

        writer = cv2.VideoWriter(
            raw_output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frame_count = 0

        with st.spinner("⏳ Detecting and tracking..."):

            while True:

                success, frame = cap.read()

                if not success:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=0.25,
                    verbose=False
                )

                result = results[0]

                annotated_frame = result.plot(
                    labels=True,
                    boxes=True
                )

                writer.write(annotated_frame)

                frame_count += 1

        cap.release()
        writer.release()

        if frame_count > 0:

            try:

                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        raw_output,
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        "-an",
                        final_output
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

            except Exception as e:

                st.error("❌ Video conversion failed.")
                st.exception(e)
                st.stop()

            st.success(
                "✅ Object detection and tracking completed!"
            )

            st.subheader("🎯 Tracked Video")

            st.video(final_output)

            with open(final_output, "rb") as f:
                video_bytes = f.read()

            st.download_button(
                label="⬇️ Download Tracked Video",
                data=video_bytes,
                file_name="tracked_output.mp4",
                mime="video/mp4"
            )

        else:

            st.error("❌ No video frames were processed.")

else:

    st.info(
        "👆 Upload a video to start object detection and tracking."
    )

st.markdown("---")
st.caption("Built with YOLO + ByteTrack + Streamlit")