"""
Real-Time Finger Counter Application Entry Point.

Captures live webcam feed frame-by-frame, processes hand landmarks using HandDetector,
and overlays live finger count metrics with a sleek UI interface.
"""

import time
import cv2
from hand_detector import HandDetector


def main() -> None:
    """
    Main function establishing video capture pipeline and real-time visualization loop.
    """
    # 1. Initialize webcam feed (Index 0 is standard default camera)
    cap = cv2.VideoCapture(0)
    
    # Configure capture resolution (1280x720 HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Could not access webcam. Please check hardware connection and permissions.")
        return

    print("[INFO] Webcam initialized successfully. Starting Hand Tracker...")
    print("[INFO] Press 'q' on the video display window to quit.")

    # 2. Instantiate detector engine
    detector = HandDetector(max_hands=2, detection_confidence=0.7, tracking_confidence=0.7)

    p_time = 0.0  # Previous time for FPS calculation

    while True:
        success, frame = cap.read()
        if not success:
            print("[WARNING] Failed to capture video frame. Skipping...")
            continue

        # Flip frame horizontally for intuitive mirror reflection
        frame = cv2.flip(frame, 1)

        # Process frame to detect hands and draw skeleton landmark joints
        frame = detector.find_hands(frame, draw=True)

        total_fingers = 0
        finger_states = []

        # Evaluate finger count if at least one hand is detected
        if detector.results and detector.results.hand_landmarks:
            total_fingers, finger_states = detector.count_open_fingers(frame, hand_index=0)

        # 3. Calculate frames per second (FPS)
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        # 4. Render sleek HUD Overlay on Video Window
        # Draw background panel card for metrics display
        cv2.rectangle(frame, (20, 20), (320, 140), (20, 20, 20), cv2.FILLED)
        cv2.rectangle(frame, (20, 20), (320, 140), (0, 255, 128), 2)  # Green border

        # Display Live Finger Count
        cv2.putText(
            frame,
            f"Fingers: {total_fingers}",
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        # Display FPS metric
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (40, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )

        # Display individual finger state badges if detected
        if finger_states:
            labels = ["T", "I", "M", "R", "P"]
            start_x = 350
            for idx, state in enumerate(finger_states):
                color = (0, 255, 0) if state == 1 else (50, 50, 200)
                cv2.rectangle(frame, (start_x + idx * 45, 20), (start_x + idx * 45 + 40, 60), color, cv2.FILLED)
                cv2.putText(
                    frame,
                    labels[idx],
                    (start_x + idx * 45 + 12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # 5. Show final rendering in window
        cv2.imshow("Computer Vision - Finger Counter", frame)

        # Exit application cleanly when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting application upon user request...")
            break

    # Release webcam resources and destroy open windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
