import cv2
import matplotlib.pyplot as plt
import time

# Load the video
video = cv2.VideoCapture(0)

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Record positions and times
positions = []
frame_times = []

# Start the global timer
start_time = time.time()

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Apply background subtractor
    fgmask = fgbg.apply(frame)

    # Find contours of the moving object
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Track the largest moving contour
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)
            center = (x + w//2, y + h//2)
            positions.append(center)

            # Use total elapsed time since video started
            elapsed_time = time.time() - start_time
            frame_times.append(elapsed_time)

            # Draw on the frame
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Motion Tracking', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC key to quit
        break

video.release()
cv2.destroyAllWindows()

# Plotting
x_vals = [pos[0] for pos in positions]
y_vals = [pos[1] for pos in positions]

plt.figure(figsize=(8, 6))
plt.plot(frame_times, x_vals, marker='o', label='X Position', color='blue')
plt.plot(frame_times, y_vals, marker='x', label='Y Position', color='red')
plt.title("Object Motion Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Position (pixels)")
plt.legend()
plt.grid(True)
plt.show()
