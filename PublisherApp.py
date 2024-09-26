1pip install paho-mqtt Pillow
import streamlit as st
import paho.mqtt.client as mqtt
import base64
import json
from PIL import Image as PilImage
import io
import time

# MQTT configuration
BROKER = "192.168.4.15"
PORT = 1883
PUBLISH_TOPIC = "image/detection"
SUBSCRIBE_TOPIC = "detection/results"

# Global variable to store the received result
prediction_result = None

# MQTT callback when a message is received
def on_message(client, userdata, msg):
    global prediction_result
    try:
        # Decode the payload
        data = msg.payload.decode()

        # Convert the payload from JSON string to Python dictionary
        prediction_result = json.loads(data)

    except Exception as e:
        st.error(f"Error parsing detection result: {e}")

# Setup MQTT client
client = mqtt.Client()
client.on_message = on_message

def connect_and_subscribe():
    try:
        client.connect(BROKER, PORT, 60)
        client.subscribe(SUBSCRIBE_TOPIC)
        client.loop_start()
        st.success(f"Connected to MQTT broker at {BROKER}:{PORT} and subscribed to {SUBSCRIBE_TOPIC}")
    except Exception as e:
        st.error(f"Failed to connect or subscribe: {e}")

# Encode the image as base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Streamlit app layout
st.title("Water Meter Detection")
st.write("Upload an image to detect objects using YOLO (via MQTT).")

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Upload button to upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = PilImage.open(uploaded_file)

    # Convert image to RGB if it has an alpha channel (RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Encode the image to base64
    base64_image = encode_image(image)

    # Send the image via MQTT
    if st.button("Send Image for Detection"):
        if not client.is_connected():
            connect_and_subscribe()

        # Prepare JSON payload
        payload = json.dumps({"image": base64_image})
        result = client.publish(PUBLISH_TOPIC, payload)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            st.success("Image sent successfully! Waiting for the detection result...")
        else:
            st.error(f"Failed to send image. Return code: {result.rc}")

        # Wait for the result
        with st.spinner("Waiting for detection result..."):
            time.sleep(5)

    # If detection result is received, display it
    if prediction_result:
        # Decode the processed image received from SubscriberApp
        base64_image_result = prediction_result.get('processed_image')
        img_data = base64.b64decode(base64_image_result)
        processed_image = PilImage.open(io.BytesIO(img_data))

        # Display the processed image with bounding boxes
        st.image(processed_image, caption='Detected Image with Bounding Boxes', use_column_width=True)

        # Display formatted detection result
        formatted_output = prediction_result.get('formatted_output', 'No formatted output')
        st.write(f"Detected Numbers: {formatted_output}")

        # Store the uploaded image and formatted output in history
        st.session_state.history.append((processed_image, formatted_output))

else:
    st.write("Please upload an image to begin detection.")

