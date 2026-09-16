
import streamlit as st
import cv2
from deepface import DeepFace
import time

st.set_page_config(
    page_title="Real-Time Emotion Detection",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Real-Time Emotion Detection")
st.write("This application detects your emotion using your webcam.")

start = st.button("Start Camera")

frame_placeholder = st.empty()

stop = st.button("Stop Camera")

if start:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not open the camera.")
    else:

        while True:

            
            ret, frame = cap.read()

            if not ret:
                st.error("❌ Could not read frame from camera.")
                break

            try:
                
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False
                )

                
                emotion = result[0]['dominant_emotion']

                
                cv2.putText(
                    frame,
                    f"Emotion: {emotion}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 200, 0),
                    2
                )

            except Exception as e:
                print(e)

            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            
            frame_placeholder.image(
                frame,
                channels="RGB"
            )

            
            time.sleep(0.01)

        cap.release()