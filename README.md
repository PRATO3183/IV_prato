# IV_prato

Note this is an underdevelopment project
items remaining:

1. blobs in detection
2. distance of calculation from the point or origin
3. output with the blob insted of top left corner


Explainations remaining:

1. line 58
2. line 65
3. line 54

Enhancements to do :
1. Video capture:
```
    cap = cv2.VideoCapture(0)  # Replace 0 with the video file path for pre-recorded input
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    # Perform object detection and depth estimation here
    cv2.imshow("Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()
```
