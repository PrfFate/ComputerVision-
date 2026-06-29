# Real-Time Hand Tracking & Finger Counter Application

An enterprise-grade Computer Vision (CV) application that captures live webcam video streams, detects human hands using MediaPipe landmarks, and accurately counts open fingers in real-time.

---

## 1. Theoretical Knowledge (Section 3.1.1)

> [!NOTE]
> Below are concise responses addressing core Computer Vision concepts, limited strictly to 3 sentences total.

1. **What is Computer Vision, and what are its primary use cases in the modern tech industry?**  
   Computer Vision (CV) is an interdisciplinary field of artificial intelligence that enables software systems to interpret and extract meaningful information from digital visual inputs, powering modern applications such as autonomous driving, medical diagnostics, facial authentication, and augmented reality.

2. **What is the key difference between image classification and object detection?**  
   While image classification assigns a single categorical label to an entire image, object detection identifies, categorizes, and localizes multiple distinct objects within an image frame using spatial bounding boxes.

3. **Why do software engineers often prefer using pre-trained frameworks (like MediaPipe) instead of training a custom neural network from scratch for simple tracking tasks?**  
   Software engineers prefer pre-trained frameworks like MediaPipe for standard tracking tasks because they deliver highly optimized, cross-platform inference pipelines out-of-the-box, saving significant time and compute resources required for large-scale data curation and neural network architecture training.

---

## 2. Setup & Installation

### Requirements
- Python 3.8+
- Webcam / Camera access

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/PrfFate/ComputerVision-.git
   cd ComputerVision-
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. How to Run

Execute the main real-time finger counter script:
```bash
python main.py
```

Press **`q`** while viewing the video window to safely exit the application.
