"""
Hand Detection Module using MediaPipe Tasks API.

This module provides an enterprise-grade HandDetector class to extract hand landmarks
and determine open/closed status for fingers in real-time visual streams.
Supports multi-hand tracking and rotation/orientation-invariant thumb detection.
"""

import os
import math
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Tuple, Optional, Any


class HandDetector:
    """
    A class wrapper for MediaPipe HandLandmarker Tasks solution providing
    landmark tracking and logical state determination for finger status.
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = "hand_landmarker.task"

    # Connections defining hand skeleton
    HAND_CONNECTIONS = [
        (0, 1), (1, 5), (9, 13), (13, 17), (5, 9), (0, 17),
        (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20)
    ]

    FINGER_TIPS = [8, 12, 16, 20]      # Index, Middle, Ring, Pinky tips
    FINGER_PIPS = [6, 10, 14, 18]      # Index, Middle, Ring, Pinky PIP joints
    THUMB_TIP = 4
    THUMB_IP = 3
    PINKY_MCP = 17
    INDEX_MCP = 5

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ):
        """
        Initialize the MediaPipe HandLandmarker configuration.

        :param max_hands: Maximum number of hands to detect simultaneously.
        :param detection_confidence: Minimum detection confidence threshold.
        :param tracking_confidence: Minimum tracking confidence threshold.
        """
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Auto-download model asset if not present locally
        self._ensure_model_exists()

        # Build HandLandmarker options
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_confidence,
            min_hand_presence_confidence=self.tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.results: Optional[Any] = None

    def _ensure_model_exists(self) -> None:
        """Downloads the pretrained hand_landmarker.task file if missing."""
        if not os.path.exists(self.MODEL_PATH):
            print(f"[INFO] Downloading pretrained model from Google Storage...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print("[INFO] Model downloaded successfully.")

    def find_hands(self, img: cv2.Mat, draw: bool = True) -> cv2.Mat:
        """
        Process an image frame to detect hands and visually annotate joints/landmarks.

        :param img: Input frame from OpenCV (BGR).
        :param draw: Boolean flag indicating whether to draw landmark nodes.
        :return: Annotated image frame.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        self.results = self.landmarker.detect(mp_image)

        if self.results and self.results.hand_landmarks and draw:
            h, w, _ = img.shape
            for hand_idx, hand_landmarks in enumerate(self.results.hand_landmarks):
                pixel_pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                # Distinct colors for multiple hands (Green/Cyan for Hand 1, Yellow/Orange for Hand 2)
                line_color = (0, 255, 128) if hand_idx == 0 else (0, 215, 255)
                node_color = (255, 0, 128) if hand_idx == 0 else (255, 128, 0)

                for start_idx, end_idx in self.HAND_CONNECTIONS:
                    cv2.line(img, pixel_pts[start_idx], pixel_pts[end_idx], line_color, 2, cv2.LINE_AA)

                for pt in pixel_pts:
                    cv2.circle(img, pt, 5, node_color, cv2.FILLED, cv2.LINE_AA)
                    cv2.circle(img, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)

        return img

    def find_positions(self, img: cv2.Mat, hand_index: int = 0) -> List[Tuple[int, int, int]]:
        """
        Extract landmark positions into pixel coordinates.

        :param img: Input image frame.
        :param hand_index: Index of the detected hand.
        :return: List of tuples containing (landmark_id, x_pixel, y_pixel).
        """
        landmark_list: List[Tuple[int, int, int]] = []
        h, w, _ = img.shape

        if self.results and self.results.hand_landmarks:
            if hand_index < len(self.results.hand_landmarks):
                target_landmarks = self.results.hand_landmarks[hand_index]
                for lm_id, lm in enumerate(target_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append((lm_id, cx, cy))

        return landmark_list

    def get_hand_label(self, hand_index: int = 0) -> str:
        """
        Identify whether the detected hand is 'Left' or 'Right'.

        :param hand_index: Index of the hand.
        :return: Hand classification string ('Left' or 'Right').
        """
        if self.results and self.results.handedness:
            if hand_index < len(self.results.handedness):
                return self.results.handedness[hand_index][0].category_name
        return "Right"

    @staticmethod
    def _euclidean_distance(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
        """Calculate 2D Euclidean distance between two points."""
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    def count_open_fingers(self, img: cv2.Mat, hand_index: int = 0) -> Tuple[int, List[int]]:
        """
        Logical condition algorithm to evaluate which fingers are open or closed.
        Uses scale and orientation-invariant distance geometry for thumb detection.

        :param img: Input image frame.
        :param hand_index: Target hand index.
        :return: Tuple containing (total open fingers count, binary list of finger states [T, I, M, R, P]).
        """
        landmarks = self.find_positions(img, hand_index=hand_index)
        if not landmarks:
            return 0, []

        finger_states: List[int] = []

        # 1. Orientation-Invariant Thumb State Logic
        # Calculate distance from thumb tip (4) to pinky MCP base (17) vs thumb IP joint (3) to pinky MCP base (17)
        thumb_tip_pt = (landmarks[self.THUMB_TIP][1], landmarks[self.THUMB_TIP][2])
        thumb_ip_pt = (landmarks[self.THUMB_IP][1], landmarks[self.THUMB_IP][2])
        pinky_mcp_pt = (landmarks[self.PINKY_MCP][1], landmarks[self.PINKY_MCP][2])

        dist_tip_to_pinky = self._euclidean_distance(thumb_tip_pt, pinky_mcp_pt)
        dist_ip_to_pinky = self._euclidean_distance(thumb_ip_pt, pinky_mcp_pt)

        # When thumb is extended/open, the tip is significantly further from pinky base than the IP joint is
        if dist_tip_to_pinky > dist_ip_to_pinky:
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
