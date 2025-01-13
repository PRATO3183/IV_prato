# Step 1: Environment Setup
# pip install numpy opencv-python gtts torch torchvision matplotlib ultralytics

from ultralytics import YOLO
import cv2
# import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Step 2: Image Input and Preprocessing
image_path = "image7.jpg"  # Path to the input image
image = cv2.imread(image_path)
# cv2.imshow("Input Image", image)
# cv2.waitKey(1)
# cv2.destroyAllWindows()


# Step 3: Object Detection with YOLO


# Perform object detection
results = model(image)

# Draw bounding boxes and labels
# Iterate over detected objects
for box in results[0].boxes:
    # Extract box coordinates, confidence, and class
    x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordinates
    conf = box.conf[0].item()             # Confidence
    cls = int(box.cls[0].item())          # Class ID
    label = results[0].names[cls]         # Class name

    # # Only mark "stairs" if detected
    # if label == "stairs":
    
    # Draw bounding box and label on the image
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, f"{label} ({conf:.2f})", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # Display the image
# cv2.imshow("Detected Objects", image)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()


# Step 4: Depth Estimation
import torch
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas.eval()

# Preprocess image for depth estimation
transform = Compose([Resize((384, 384)), ToTensor()])
input_image = transform(Image.open(image_path)).unsqueeze(0)

# Predict depth
with torch.no_grad():
    depth_map = midas(input_image)

# Display depth map
depth_array = depth_map.squeeze().numpy()
distance = depth_array.mean()  # Placeholder: actual calculation may vary
distance=int(distance)
print(f"Average Distance: {distance} centimeters")


# Step 5: Text-to-Speech for Feedback
from gtts import gTTS
import os

# Convert detected information to speech
text = f"Stairs detected at {distance} centimeters."
tts = gTTS(text, lang='en')
tts.save("output.mp3")
os.system("start output.mp3")


# Step 6: Annotate and Save the Image
cv2.putText(image, f"{text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.imwrite("output.jpg", image)
print("Annotated image saved as output.jpg.")

image = cv2.resize(image, (1080, 720))  # Resize to 1080x720

# Display the image
cv2.imshow("Detected Objects", image)
cv2.waitKey(10000)
cv2.destroyAllWindows()

# # Delete the generated files
# if os.path.exists("output.jpg"):
#     os.remove("output.jpg")
# if os.path.exists("output.mp3"):
#     os.remove("output.mp3")

print("Temporary files deleted successfully.")



# Final Workflow :
# 1. Load an image using OpenCV.
# 2. Detect objects using YOLO.
# 3. Estimate distances using MiDaS.
# 4. Generate auditory feedback using gTTS.33
# 5. Save and display the annotated image.


# Future Improvements :
# 1. Integrate segmentation models for better obstacle understanding.
# 2. Use more accurate depth models like DPT for fine-grained distance estimation.
# 3. Incorporate a GUI for easier interaction (e.g., using PyQt or Tkinter).