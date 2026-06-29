"""
Hand Detection Module using MediaPipe Framework.

This module provides an enterprise-grade HandDetector class to extract hand landmarks
and determine open/closed status for fingers in real-time visual streams.
"""

import cv2
import mediapipe as mp
from typing import List, Tuple, Dict, Any, Optional


class HandDetector:
    """
    A class wrapper for MediaPipe Hands solution providing landmark tracking
    and logical state determination for finger status.
    """

    # MediaPipe landmark indices for finger tips and PIP/IP joints
    FINGER_TIPS = [8, 12, 16, 20]      # Index, Middle, Ring, Pinky tips
    FINGER_PIPS = [6, 10, 14, 18]      # Index, Middle, Ring, Pinky PIP joints
    THUMB_TIP = 4
    THUMB_IP = 3

    def __init__(
        self,
        mode: bool = False,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ):
        """
        Initialize the MediaPipe Hands configuration.

        :param mode: Static image mode (False for video streams).
        :param max_hands: Maximum number of hands to detect simultaneously.
        :param detection_confidence: Minimum detection confidence threshold.
        :param tracking_confidence: Minimum tracking confidence threshold.
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        self.results: Optional[Any] = None

    def find_hands(self, img: cv2.Mat, draw: bool = True) -> cv2.Mat:
        """
        Process an RGB image to detect hands and visually annotate joints/landmarks.

        :param img: Input frame from OpenCV (BGR).
        :param draw: Boolean flag indicating whether to draw landmark nodes.
        :return: Annotated image frame.
        """
        # Convert frame color space from BGR to RGB required by MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Draw joint landmarks and connections with stylized colors
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style(),
                )

        return img

    def find_positions(self, img: cv2.Mat, hand_index: int = 0) -> List[Tuple[int, int, int]]:
        """
        Extract landmark positions into pixel coordinates.

        :param img: Input image frame.
        :param hand_index: Index of the detected hand (default 0 for primary hand).
        :return: List of tuples containing (landmark_id, x_pixel, y_pixel).
        """
        landmark_list: List[Tuple[int, int, int]] = []
        h, w, c = img.shape

        if self.results and self.results.multi_hand_landmarks:
            if hand_index < len(self.results.multi_hand_landmarks):
                target_hand = self.results.multi_hand_landmarks[hand_index]
                for lm_id, lm in enumerate(target_hand.landmark):
                    # Convert normalized coordinate ratios [0, 1] into pixel values
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append((lm_id, cx, cy))

        return landmark_list

    def get_hand_label(self, hand_index: int = 0) -> str:
        """
        Identify whether the detected hand is 'Left' or 'Right'.

        :param hand_index: Index of the hand.
        :return: Hand classification string ('Left' or 'Right').
        """
        if self.results and self.results.multi_handedness:
            if hand_index < len(self.results.multi_handedness):
                return self.results.multi_handedness[hand_index].classification[0].label
        return "Right"

    def count_open_fingers(self, img: cv2.Mat, hand_index: int = 0) -> Tuple[int, List[int]]:
        """
        Logical condition algorithm to evaluate which fingers are open or closed.

        :param img: Input image frame.
        :param hand_index: Target hand index.
        :return: Tuple containing (total open fingers count, binary list of finger states [T, I, M, R, P]).
        """
        landmarks = self.find_positions(img, hand_index=hand_index)
        if not landmarks:
            return 0, []

        finger_states: List[int] = []
        hand_label = self.get_hand_label(hand_index=hand_index)

        # 1. Thumb State Logic
        # For thumb, check lateral (x-axis) displacement relative to IP joint depending on hand side
        thumb_tip_x = landmarks[self.THUMB_TIP][1]
        thumb_ip_x = landmarks[self.THUMB_IP][1]

        if hand_label == "Right":
            # Right hand thumb tip is open when positioned further left (smaller x) than IP joint
            if thumb_tip_x < thumb_ip_x:
                finger_states.append(1)
            else:
                finger_states.append(0)
        else:
            # Left hand thumb tip is open when positioned further right (greater x) than IP joint
            if thumb_tip_x > thumb_ip_x:
                finger_states.append(1)
            else:
                finger_states.append(0)

        # 2. Four Fingers State Logic (Index, Middle, Ring, Pinky)
        # Check vertical (y-axis) coordinate: open if tip is higher (smaller y value) than PIP joint
        for tip_id, pip_id in zip(self.FINGER_TIPS, self.FINGER_PIPS):
            if landmarks[tip_id][2] < landmarks[pip_id][2]:
                finger_states.append(1)
            else:
                finger_states.append(0)

        total_count = sum(finger_states)
        return total_count, finger_states
