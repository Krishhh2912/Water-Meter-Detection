import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image as PilImage
import numpy as np
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('best.pt')

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Streamlit app layout
st.title("Water Meter Detection")
st.write("Upload an image to detect objects using the YOLO model.")

# Create tabs for detection and history
tab_detection, tab_history = st.tabs(["Detection", "History"])

with tab_detection:
    # Upload button to upload the image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image
        image = PilImage.open(uploaded_file)

        # Convert image to RGB if it has an alpha channel (RGBA)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Convert PIL image to numpy array (for OpenCV processing)
        img_np = np.array(image)

        # Perform YOLO object detection on the uploaded image
        results = model(img_np)

        # Plotting the image and drawing the bounding boxes
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_np)
        ax.axis('off')

        detected_results = []

        # Draw bounding boxes and detected numbers
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
                conf = box.conf[0].cpu().numpy()   # Confidence score
                cls = box.cls[0].cpu().numpy()     # Class ID

                # Draw rectangle around the object
                ax.add_patch(plt.Rectangle(
                    (xyxy[0], xyxy[1]),
                    xyxy[2] - xyxy[0],
                    xyxy[3] - xyxy[1],
                    fill=False,
                    edgecolor='red',
                    linewidth=2
                ))

                # Add label text with class ID and confidence
                ax.text(xyxy[0], xyxy[1] - 10,
                        f'ID: {int(cls)} Conf: {conf:.2f}',
                        color='white', fontsize=12,
                        bbox=dict(facecolor='red', alpha=0.5))

                # Append detected class and coordinates
                detected_results.append((cls, xyxy))

        st.pyplot(fig)

        # Sort the detected results by coordinates
        detected_results.sort(key=lambda x: (x[1][0], x[1][1], x[1][2], x[1][3]))

        # Display detected numbers sorted by coordinates
        sorted_numbers = [int(cls) for cls, _ in detected_results]

        # Format the output to show the last three numbers with a dot
        if len(sorted_numbers) > 3:
            # Create a string of the detected numbers
            number_string = ''.join(map(str, sorted_numbers))
            # Insert a dot before the last three digits
            formatted_numbers = number_string[:-3] + '.' + number_string[-3:]
        else:
            # If there are 3 or fewer digits, just join them without a dot
            formatted_numbers = ','.join(map(str, sorted_numbers))

        # Use print(f) format for displaying numbers
        st.write(f"Detected Numbers: {formatted_numbers}")

        # Store the uploaded image and detected numbers in history
        st.session_state.history.append((image, formatted_numbers))

    else:
        st.write("Please upload an image to begin detection.")

with tab_history:
    st.write("Detection History")
    if st.session_state.history:
        for idx, (img, numbers) in enumerate(st.session_state.history):
            st.image(img, caption=f"Detected Image {idx + 1}", use_column_width=True)
            st.write(f"Detected Numbers: {numbers}")
    else:
        st.write("No history available.")
