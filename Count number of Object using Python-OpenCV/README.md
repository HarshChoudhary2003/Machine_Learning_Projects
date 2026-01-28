<div align="center">

# 🔢 Object Counter using OpenCV

### *Computer Vision for Real-Time Object Counting*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/Domain-Computer%20Vision-purple?style=flat-square" />
<img src="https://img.shields.io/badge/Type-Image%20Processing-blue?style=flat-square" />

---

*Automatically count objects in images using Python and OpenCV image processing techniques.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Usage](#-usage)
- [Applications](#-applications)

---

## 🎯 Overview

This project demonstrates **computer vision** techniques to count objects in images automatically. Using OpenCV's powerful image processing capabilities, the system can detect and count distinct objects in various scenarios.

### 🌟 Key Features
- 🖼️ **Image Processing** - Advanced filtering and segmentation
- 🔍 **Contour Detection** - Accurate object boundary detection
- 🔢 **Automatic Counting** - No manual intervention needed
- ⚡ **Fast Processing** - Efficient algorithms

---

## 🔬 How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Input Image   │ --> │  Preprocessing  │ --> │  Grayscale      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐     ┌─────────────────┐     ┌───────▼─────────┐
│   Object Count  │ <-- │    Contours     │ <-- │  Thresholding   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Processing Pipeline
1. **Load Image** - Read input image
2. **Convert to Grayscale** - Simplify processing
3. **Apply Threshold** - Binary segmentation
4. **Find Contours** - Detect object boundaries
5. **Count Objects** - Return total count

---

## 📁 Project Structure

```
Count number of Object using Python-OpenCV/
├── main.py           # Main counting script
└── images.jpg        # Sample test image
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/Count number of Object using Python-OpenCV"

# Install dependencies
pip install opencv-python numpy
```

---

## 💻 Usage

```bash
# Run the object counter
python main.py
```

### Code Example
```python
import cv2

# Load image
img = cv2.imread('images.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply threshold
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Count objects
print(f"Objects found: {len(contours)}")
```

---

## 🏭 Applications

| Domain | Use Case |
|--------|----------|
| 🏭 **Manufacturing** | Count products on assembly line |
| 🌾 **Agriculture** | Count fruits, seeds, plants |
| 🚗 **Traffic** | Vehicle counting |
| 🔬 **Medical** | Cell counting |
| 📦 **Inventory** | Stock management |

---

## 🔧 Customization

- Adjust threshold values for different lighting
- Modify contour area filters for size-based counting
- Add color filtering for specific object detection

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
