import cv2
import torch
import numpy as np
from torchvision import transforms
from custom_module import CustomVGG19  # Replace with the correct import statement

# Load the trained VGG19 model
model = CustomVGG19()
model.load_state_dict(torch.load('custom_vgg19_model.pth'))
model.eval()

# Set up the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, adjust as needed

# Define a transform for preprocessing the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG19-compatible image size
    transforms.ToTensor(),
])

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    # Preprocess the frame
    input_image = transform(frame)
    input_image = input_image.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(input_image)

    _, predicted_class = torch.max(outputs, 1)

    # Display the result on the frame
    label = "Danger" if predicted_class.item() == 1 else "Safe"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if predicted_class.item() == 1 else (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-time Detection', frame)

    # Exit the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
