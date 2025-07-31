<img src="https://combilift.com/wp-content/uploads/2023/09/combilfit-25.png" width="480"/>

<img src="https://github.com/combiCaolan/chicken-camera/blob/main/.github/example.drawio.png" />

# Counting live poultry on a conveyor belt using computer vision

## Abstract

This project aims to develop reliable solutions for counting live chickens on a moving conveyor belt using computer vision techniques. The system will be designed for industrial poultry processing environments where accurate, real-time counting is critical for operational efficiency, inventory management, and quality control.

The planned solution will combine multiple computer vision approaches including traditional OpenCV methods, deep learning models (YOLO), and hybrid detection systems to achieve high accuracy in challenging industrial conditions such as varying lighting, overlapping birds, and high-speed conveyor movement.

## Project Goals

- Develop real-time detection and counting algorithms for live poultry
- Create robust solutions for industrial environments
- Implement multiple detection approaches for reliability comparison
- Design confidence scoring systems for quality assurance
- Build integration-ready components for existing factory systems

## Proposed Technical Approach

The development will explore several approaches:

1. **Traditional Computer Vision**
   - Color-based segmentation for white/cream chickens
   - Contour detection and geometric validation
   - Motion tracking on conveyor belts

2. **Deep Learning Methods**
   - YOLO object detection fine-tuning
   - Custom dataset creation and training
   - Transfer learning from existing animal detection models

3. **Hybrid Systems**
   - Combining multiple detection methods
   - Confidence scoring and validation
   - Failure mode analysis and redundancy

### Target Specifications:

**Camera System:**
- Industrial-grade cameras with minimum 1080p resolution
- Frame rate: 30-60 FPS for accurate motion capture
- IP65/IP67 rating for dust and moisture protection
- Adjustable focus and exposure for varying lighting conditions

**Processing Unit:**
- GPU acceleration capability (NVIDIA RTX series or equivalent)
- Minimum 8GB RAM, 16GB+ preferred for video processing
- Industrial PC with appropriate environmental ratings
- Temperature range suitable for factory environments

```python
# Planned project structure
chicken-camera/
├── approaches/
│   ├── count/               # Python approaches to counting chickens
│   └── judge health/        # Python approaches to gauging the chickens health
├── test-data/
│   ├── photos/          # photos for test data
│   └── videos/          # videos for test data
├── outputs/             # Videos for each approach named for appraoch - mp4 files
├── docs/                 # Documentation
└── tests/                # Unit and integration tests

```

## Contact

Software Developer:
caolan.maguire@combilift.com

## Languages & Frameworks
**Languages:** Python

**Frameworks/Libraries**: Opencv, numpy, matplotlib, ultralytics, sklearn
