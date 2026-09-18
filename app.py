import os
import tempfile
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Object Detection and Tracking",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 Object Detection and Tracking")
st.write("Detect and track objects in a video using YOLO and ByteTrack.")

@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")

# Load model
try:
    model = load_model()
    st.success("✅ YOLO model loaded successfully!")
except Exception as e:
    st.error("❌ Could not load YOLO model.")
    st.exception(e)
    st.stop()

# Upload video
uploaded_file = st.file_uploader(
    "📤 Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:

    # Save uploaded video temporarily
    input_path = os.path.join(
        tempfile.gettempdir(),
        uploaded_file.name
    )

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ Video uploaded successfully!")

    # Show original video
    st.subheader("🎥 Original Video")
    st.video(uploaded_file)

    if st.button("🚀 Detect & Track Objects"):

        output_dir = tempfile.mkdtemp()
        output_name = "tracked_video"

        try:
            with st.spinner(
                "⏳ Detecting and tracking objects... Please wait."
            ):
                results = model.track(
                    source=input_path,
                    tracker="bytetrack.yaml",
                    conf=0.25,
                    save=True,
                    project=output_dir,
                    name=output_name,
                    exist_ok=True,
                    stream=False,
                    verbose=False
                )

            # Find generated video
            result_folder = os.path.join(
                output_dir,
                output_name
            )

            output_video = None

            for root, dirs, files in os.walk(result_folder):
                for file in files:
                    if file.lower().endswith(
                        (".mp4", ".avi", ".mov", ".mkv")
                    ):
                        output_video = os.path.join(root, file)
                        break

                if output_video:
                    break

            if output_video and os.path.exists(output_video):

                st.success(
                    "✅ Object detection and tracking completed!"
                )

                st.subheader("🎯 Tracked Video")

                st.video(output_video)

                with open(output_video, "rb") as video_file:
                    video_bytes = video_file.read()

                st.download_button(
                    label="⬇️ Download Tracked Video",
                    data=video_bytes,
                    file_name="tracked_output.mp4",
                    mime="video/mp4"
                )

            else:
                st.error(
                    "❌ Tracked video was not created."
                )

        except Exception as e:
            st.error("❌ An error occurred while processing the video.")
            st.exception(e)

else:
    st.info(
        "👆 Upload an MP4 video above to start object detection and tracking."
    )

st.markdown("---")
st.caption("Built with YOLO + ByteTrack + Streamlit")
