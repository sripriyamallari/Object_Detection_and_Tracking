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

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ Video uploaded successfully!")

    st.subheader("🎥 Original Video")
    st.video(uploaded_file)

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

                if result.boxes is not None:

                    boxes = result.boxes.xyxy.cpu().numpy()

                    if result.boxes.id is not None:
                        track_ids = result.boxes.id.int().cpu().tolist()
                    else:
                        track_ids = [None] * len(boxes)

                    if result.boxes.cls is not None:
                        classes = result.boxes.cls.int().cpu().tolist()
                    else:
                        classes = [0] * len(boxes)

                    for box, track_id, cls in zip(
                        boxes,
                        track_ids,
                        classes
                    ):

                        x1, y1, x2, y2 = map(int, box)

                        class_name = model.names.get(
                            cls,
                            "Object"
                        )

                        if track_id is not None:
                            label = f"{class_name} ID: {track_id}"
                        else:
                            label = f"{class_name} ID: ?"

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            label,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                writer.write(frame)
                frame_count += 1

        cap.release()
        writer.release()

        if os.path.exists(output_path) and frame_count > 0:

            st.success(
                "✅ Object detection and tracking completed!"
            )

            st.subheader("🎯 Tracked Video")

            st.video(output_path)

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            st.download_button(
                "⬇️ Download Tracked Video",
                data=video_bytes,
                file_name="tracked_output.mp4",
                mime="video/mp4"
            )

        else:
            st.error("❌ Tracking video could not be created.")

else:
    st.info("👆 Upload a video to begin.")