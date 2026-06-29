"""
Real-Time Finger Counter Application Entry Point.

Captures live webcam feed frame-by-frame, processes hand landmarks using HandDetector,
and overlays live finger count metrics for single or multiple hands with a sleek UI interface.
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

    # 2. Instantiate detector engine supporting up to 2 hands simultaneously
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

        grand_total_fingers = 0
        all_hand_states = []

        # 3. Evaluate finger count across ALL detected hands
        if detector.results and detector.results.hand_landmarks:
            num_detected = len(detector.results.hand_landmarks)
            for h_idx in range(num_detected):
                count, states = detector.count_open_fingers(frame, hand_index=h_idx)
                grand_total_fingers += count
                all_hand_states.append((h_idx, count, states))

        # 4. Calculate frames per second (FPS)
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        # 5. Render sleek HUD Overlay on Video Window
        # Draw background panel card for metrics display
        cv2.rectangle(frame, (20, 20), (340, 140), (20, 20, 20), cv2.FILLED)
        cv2.rectangle(frame, (20, 20), (340, 140), (0, 255, 128), 2)  # Green border

        # Display Total Live Finger Count
        cv2.putText(
            frame,
            f"Fingers: {grand_total_fingers}",
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

        # Display individual finger state badges for detected hands
        if all_hand_states:
            labels = ["T", "I", "M", "R", "P"]
            for h_idx, count, states in all_hand_states:
                start_x = 370
                start_y = 20 + (h_idx * 50)
                
                # Hand label identifier
                cv2.putText(
                    frame,
                    f"H{h_idx + 1}:",
                    (start_x, start_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                for idx, state in enumerate(states):
                    color = (0, 255, 0) if state == 1 else (50, 50, 200)
                    box_x = start_x + 55 + (idx * 45)
                    cv2.rectangle(frame, (box_x, start_y), (box_x + 40, start_y + 40), color, cv2.FILLED)
                    cv2.putText(
                        frame,
                        labels[idx],
                        (box_x + 12, start_y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        # 6. Show final rendering in window
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
