import cv2
import mediapipe as mp
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Placeholder for hand sign mapping (Define unique hand gestures for each character A-Z, 0-9, and spacebar)
hand_sign_mapping = {chr(i): None for i in range(65, 91)}  # A-Z
hand_sign_mapping.update({str(i): None for i in range(10)})  # 0-9
hand_sign_mapping['space'] = None

# Variable to hold the constructed message
constructed_message = ""
assigning_mode = True
assigned_signs = set()
display_message = False  # Flag to control when the message is displayed

# Function to assign hand gestures manually
def assign_hand_sign(character, landmarks):
    if hand_sign_mapping[character] is not None:
        print(f"Error: The hand sign for '{character}' is already assigned. Please assign a different gesture.")
        return False
    hand_sign_mapping[character] = [lm for lm in landmarks.landmark]
    assigned_signs.add(character)
    print(f"Assigned hand sign for: {character}")
    return True

# Function to delete a hand sign assignment
def delete_hand_sign(character):
    if hand_sign_mapping[character] is None:
        print(f"Error: No hand sign assigned for '{character}' to delete.")
        return False
    hand_sign_mapping[character] = None
    assigned_signs.remove(character)
    print(f"Deleted hand sign for: {character}")
    return True

# Function to identify hand signs based on landmarks
def identify_hand_sign(landmarks):
    for sign, stored_landmarks in hand_sign_mapping.items():
        if stored_landmarks:
            if all(abs(landmarks.landmark[i].y - stored_landmarks[i].y) < 0.1 for i in range(21)):
                return sign
    return None

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame to avoid mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks and identify signs
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_sign = identify_hand_sign(hand_landmarks)
            if detected_sign:
                if display_message:  # Display the message only if the flag is True
                    constructed_message += detected_sign if detected_sign != 'space' else ' '
                cv2.putText(frame, f'Sign: {detected_sign}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Assign hand sign on pressing 'a' to 'z', '0' to '9', or 'space'
            key = cv2.waitKey(1) & 0xFF
            if assigning_mode:
                if key in range(97, 123):  # 'a' to 'z'
                    if not assign_hand_sign(chr(key).upper(), hand_landmarks):
                        continue
                elif key in range(48, 58):  # '0' to '9'
                    if not assign_hand_sign(chr(key), hand_landmarks):
                        continue
                elif key == ord(' '):  # space
                    if not assign_hand_sign('space', hand_landmarks):
                        continue
                elif key == ord('d'):  # Delete assigned hand sign
                    character_to_delete = input("Enter the character to delete the hand sign for: ").strip()
                    if character_to_delete in hand_sign_mapping:
                        delete_hand_sign(character_to_delete)

    # Check if all characters are assigned and stop assignment mode
    if len(assigned_signs) == 37:  # 26 letters + 10 digits + space
        cv2.putText(frame, 'All signs assigned. Press "Enter" to start the message.', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the constructed message if the flag is True
    if display_message:
        cv2.putText(frame, f'Message: {constructed_message}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with hand tracking
    cv2.imshow('Hand Tracking with MediaPipe', frame)

    # Check for user input to stop assignment mode or start displaying the message
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') and len(assigned_signs) == 37:
        if assigning_mode:
            assigning_mode = False
            print("Finished assigning hand signs. Now you can enter the message.")
        else:
            break

    if key == 13:  # Enter key
        display_message = True
        print("Now displaying the message...")

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()

# Print the final message
print(f"Final Message: {constructed_message}")