import os
import tempfile
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
        uploaded_file.name
    )

    with open(input_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success("✅ Video uploaded successfully!")

    st.subheader("🎥 Original Video")
    st.video(input_path)

    if st.button("🚀 Detect & Track Objects"):

        output_path = os.path.join(
            tempfile.gettempdir(),
            "tracked_output.mp4"
        )

        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        frame_count = 0

        with st.spinner("⏳ Detecting and tracking objects..."):

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

                annotated_frame = results[0].plot()

                writer.write(annotated_frame)

                frame_count += 1

        cap.release()
        writer.release()

        if os.path.exists(output_path) and frame_count > 0:

            st.success(
                "✅ Object detection and tracking completed!"
            )

            st.subheader("🎯 Tracked Video")

            st.video(output_path)

            with open(output_path, "rb") as file:
                video_bytes = file.read()

            st.download_button(
                label="⬇️ Download Tracked Video",
                data=video_bytes,
                file_name="tracked_output.mp4",
                mime="video/mp4"
            )

        else:

            st.error(
                "❌ Tracking video could not be created."
            )

else:

    st.info(
        "👆 Upload a video to start object detection and tracking."
    )

st.markdown("---")
st.caption("Built with YOLO + ByteTrack + Streamlit")