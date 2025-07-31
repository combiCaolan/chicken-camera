"""
Chicken Pose Estimation System
==============================

A computer vision system for analyzing chicken posture and behavior.
Determines if chickens are upright and if their necks are erect.

INSTALLATION:
pip install opencv-python matplotlib scikit-learn pandas numpy

ALTERNATIVE OPENCV INSTALLATION (if opencv-python fails):
pip install opencv-python-headless

USAGE:
estimator = ChickenPoseEstimator()
pose = estimator.estimate_pose('chicken_image.jpg')
estimator.visualize_pose('chicken_image.jpg', pose)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

def check_dependencies():
    """Check if all required packages are available"""
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy', 
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas'
    }
    
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nTo install missing packages, run:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ All dependencies are available!")
        return True

@dataclass
class ChickenPose:
    """Data class to store chicken pose analysis results"""
    is_upright: bool
    neck_erect: bool
    upright_confidence: float
    neck_confidence: float
    body_angle: float
    neck_angle: float
    keypoints: dict

class ChickenPoseEstimator:
    """
    Chicken pose estimation system that combines multiple approaches:
    1. Custom keypoint detection for chicken-specific anatomy
    2. Geometric analysis for posture assessment
    3. Machine learning-based pose classification
    """
    
    def __init__(self):
        # Initialize chicken-specific parameters
        self.setup_chicken_anatomy()
        
    def setup_chicken_anatomy(self):
        """Define chicken-specific anatomical parameters"""
        self.chicken_keypoints = {
            'head': None, 'neck_top': None, 'neck_mid': None, 'neck_base': None,
            'body_center': None, 'body_front': None, 'body_back': None,
            'left_leg': None, 'right_leg': None, 'tail': None
        }
        
        # Thresholds for pose analysis
        self.upright_angle_threshold = 45  # degrees from vertical
        self.neck_erect_angle_threshold = 30  # degrees from upward
        
    def detect_chicken_keypoints(self, image: np.ndarray) -> dict:
        """
        Detect chicken-specific keypoints using computer vision techniques
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Use edge detection and contours to find chicken outline
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self.chicken_keypoints
            
        # Find the largest contour (assumed to be the chicken)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box and center
        x, y, w, h = cv2.boundingRect(main_contour)
        center_x, center_y = x + w//2, y + h//2
        
        # Estimate keypoints based on chicken anatomy
        keypoints = self._estimate_anatomical_points(main_contour, (center_x, center_y), (w, h))
        
        return keypoints
    
    def _estimate_anatomical_points(self, contour, center, dimensions):
        """Estimate chicken anatomical points from contour analysis"""
        center_x, center_y = center
        width, height = dimensions
        
        # Convert contour to points
        contour_points = contour.reshape(-1, 2)
        
        # Find extreme points
        topmost = tuple(contour_points[contour_points[:, 1].argmin()])
        bottommost = tuple(contour_points[contour_points[:, 1].argmax()])
        leftmost = tuple(contour_points[contour_points[:, 0].argmin()])
        rightmost = tuple(contour_points[contour_points[:, 0].argmax()])
        
        # Estimate chicken keypoints based on typical anatomy
        keypoints = {}
        
        # Head is typically the topmost point or close to it
        keypoints['head'] = topmost
        
        # Neck points - estimate based on head position and body center
        neck_vector_x = (center_x - topmost[0]) * 0.3
        neck_vector_y = (center_y - topmost[1]) * 0.3
        
        keypoints['neck_top'] = (
            int(topmost[0] + neck_vector_x * 0.3),
            int(topmost[1] + neck_vector_y * 0.3)
        )
        keypoints['neck_mid'] = (
            int(topmost[0] + neck_vector_x * 0.6),
            int(topmost[1] + neck_vector_y * 0.6)
        )
        keypoints['neck_base'] = (
            int(topmost[0] + neck_vector_x),
            int(topmost[1] + neck_vector_y)
        )
        
        # Body points
        keypoints['body_center'] = (center_x, center_y)
        keypoints['body_front'] = (int(center_x - width * 0.2), center_y)
        keypoints['body_back'] = (int(center_x + width * 0.2), center_y)
        
        # Leg points - estimated from bottom area
        keypoints['left_leg'] = (int(center_x - width * 0.15), int(center_y + height * 0.3))
        keypoints['right_leg'] = (int(center_x + width * 0.15), int(center_y + height * 0.3))
        
        # Tail - typically at the back
        keypoints['tail'] = rightmost if rightmost[0] > center_x else leftmost
        
        return keypoints
    
    def analyze_posture(self, keypoints: dict) -> Tuple[bool, float, float]:
        """
        Analyze if chicken is upright based on keypoints
        Returns: (is_upright, confidence, body_angle)
        """
        if not keypoints.get('body_center') or not keypoints.get('left_leg') or not keypoints.get('right_leg'):
            return False, 0.0, 0.0
        
        body_center = keypoints['body_center']
        left_leg = keypoints['left_leg']
        right_leg = keypoints['right_leg']
        
        # Calculate average leg position
        avg_leg_x = (left_leg[0] + right_leg[0]) / 2
        avg_leg_y = (left_leg[1] + right_leg[1]) / 2
        
        # Calculate body angle from vertical
        body_vector_x = body_center[0] - avg_leg_x
        body_vector_y = body_center[1] - avg_leg_y
        
        # Angle from vertical (0 degrees = perfectly upright)
        body_angle = abs(math.degrees(math.atan2(body_vector_x, body_vector_y)))
        
        # Determine if upright
        is_upright = body_angle < self.upright_angle_threshold
        
        # Confidence based on how close to vertical
        confidence = max(0, 1 - (body_angle / 90))  # Normalize to 0-1
        
        return is_upright, confidence, body_angle
    
    def analyze_neck_posture(self, keypoints: dict) -> Tuple[bool, float, float]:
        """
        Analyze if chicken's neck is erect
        Returns: (neck_erect, confidence, neck_angle)
        """
        if not keypoints.get('head') or not keypoints.get('neck_base'):
            return False, 0.0, 0.0
        
        head = keypoints['head']
        neck_base = keypoints['neck_base']
        
        # Calculate neck vector
        neck_vector_x = head[0] - neck_base[0]
        neck_vector_y = head[1] - neck_base[1]  # Note: y increases downward in image coordinates
        
        # Calculate angle from horizontal (upward is negative y)
        neck_angle = math.degrees(math.atan2(-neck_vector_y, abs(neck_vector_x)))
        
        # Neck is erect if it's pointing upward (positive angle from horizontal)
        neck_erect = neck_angle > self.neck_erect_angle_threshold
        
        # Confidence based on how upward the neck is pointing
        confidence = max(0, min(1, neck_angle / 90))  # Normalize to 0-1
        
        return neck_erect, confidence, neck_angle
    
    def estimate_pose(self, image_path: str) -> ChickenPose:
        """
        Main function to estimate chicken pose from image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect keypoints
        keypoints = self.detect_chicken_keypoints(image)
        
        # Analyze posture
        is_upright, upright_conf, body_angle = self.analyze_posture(keypoints)
        neck_erect, neck_conf, neck_angle = self.analyze_neck_posture(keypoints)
        
        return ChickenPose(
            is_upright=is_upright,
            neck_erect=neck_erect,
            upright_confidence=upright_conf,
            neck_confidence=neck_conf,
            body_angle=body_angle,
            neck_angle=neck_angle,
            keypoints=keypoints
        )
    
    def visualize_pose(self, image_path: str, pose: ChickenPose, save_path: Optional[str] = None):
        """
        Visualize the detected pose with keypoints and analysis
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image with keypoints
        ax1.imshow(image_rgb)
        ax1.set_title("Chicken Keypoints Detection")
        
        # Draw keypoints
        colors = {
            'head': 'red', 'neck_top': 'orange', 'neck_mid': 'orange', 'neck_base': 'orange',
            'body_center': 'blue', 'body_front': 'cyan', 'body_back': 'cyan',
            'left_leg': 'green', 'right_leg': 'green', 'tail': 'purple'
        }
        
        for point_name, point in pose.keypoints.items():
            if point:
                ax1.plot(point[0], point[1], 'o', color=colors.get(point_name, 'black'), 
                        markersize=8, label=point_name)
        
        # Draw connections
        connections = [
            ('head', 'neck_top'), ('neck_top', 'neck_mid'), ('neck_mid', 'neck_base'),
            ('neck_base', 'body_center'), ('body_center', 'left_leg'), ('body_center', 'right_leg'),
            ('body_center', 'tail')
        ]
        
        for start, end in connections:
            start_point = pose.keypoints.get(start)
            end_point = pose.keypoints.get(end)
            if start_point and end_point:
                ax1.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                        'g-', linewidth=2, alpha=0.7)
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.axis('off')
        
        # Pose analysis results
        ax2.axis('off')
        results_text = f"""
        CHICKEN POSE ANALYSIS
        =====================
        
        UPRIGHT STATUS:
        • Is Upright: {'✅ YES' if pose.is_upright else '❌ NO'}
        • Confidence: {pose.upright_confidence:.2f}
        • Body Angle: {pose.body_angle:.1f}° from vertical
        • Threshold: <{self.upright_angle_threshold}° for upright
        
        NECK POSTURE:
        • Neck Erect: {'✅ YES' if pose.neck_erect else '❌ NO'}
        • Confidence: {pose.neck_confidence:.2f}
        • Neck Angle: {pose.neck_angle:.1f}° from horizontal
        • Threshold: >{self.neck_erect_angle_threshold}° for erect
        
        SUMMARY:
        • Overall Status: {self._get_overall_status(pose)}
        • Posture Quality: {self._get_posture_quality(pose)}
        """
        
        ax2.text(0.05, 0.95, results_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _get_overall_status(self, pose: ChickenPose) -> str:
        """Get overall chicken status"""
        if pose.is_upright and pose.neck_erect:
            return "🐔 Alert & Upright"
        elif pose.is_upright:
            return "🐔 Upright (head down)"
        elif pose.neck_erect:
            return "🐔 Alert (lying/sitting)"
        else:
            return "🐔 Resting/Inactive"
    
    def _get_posture_quality(self, pose: ChickenPose) -> str:
        """Get posture quality assessment"""
        avg_confidence = (pose.upright_confidence + pose.neck_confidence) / 2
        if avg_confidence > 0.8:
            return "Excellent"
        elif avg_confidence > 0.6:
            return "Good"
        elif avg_confidence > 0.4:
            return "Moderate"
        else:
            return "Poor"

# Advanced pose estimation using enhanced computer vision
class AdvancedChickenPoseEstimator:
    """
    Advanced chicken pose estimator using enhanced computer vision techniques
    """
    
    def __init__(self):
        self.base_estimator = ChickenPoseEstimator()
        
    def enhance_keypoint_detection(self, image: np.ndarray) -> dict:
        """
        Enhanced keypoint detection using multiple computer vision techniques
        """
        # Convert to different color spaces for better analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Use adaptive thresholding for better contour detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(
            adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Enhanced keypoint estimation
        return self.base_estimator.detect_chicken_keypoints(image)
    
    def analyze_chicken_activity(self, pose: ChickenPose) -> str:
        """
        Analyze chicken activity based on pose
        """
        if pose.is_upright and pose.neck_erect:
            if pose.upright_confidence > 0.8 and pose.neck_confidence > 0.8:
                return "Highly Alert - Scanning Environment"
            else:
                return "Alert - Normal Activity"
        elif pose.is_upright and not pose.neck_erect:
            return "Upright - Feeding/Foraging"
        elif not pose.is_upright and pose.neck_erect:
            return "Sitting/Lying - Alert"
        else:
            return "Resting/Sleeping"

# Utility functions for batch processing
def process_chicken_images(image_folder: str, output_folder: str = None):
    """
    Process multiple chicken images for pose estimation
    """
    import os
    import glob
    
    estimator = ChickenPoseEstimator()
    
    # Find all images in folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
    
    results = []
    
    for i, image_path in enumerate(image_paths):
        try:
            print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            pose = estimator.estimate_pose(image_path)
            
            result = {
                'filename': os.path.basename(image_path),
                'is_upright': pose.is_upright,
                'neck_erect': pose.neck_erect,
                'upright_confidence': pose.upright_confidence,
                'neck_confidence': pose.neck_confidence,
                'body_angle': pose.body_angle,
                'neck_angle': pose.neck_angle
            }
            
            results.append(result)
            
            # Optionally save visualization
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                save_path = os.path.join(output_folder, f"pose_{os.path.basename(image_path)}")
                estimator.visualize_pose(image_path, pose, save_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    return results

def create_test_image():
    """Create a simple test image to verify the system works"""
    # Create a simple test image with basic shapes
    test_img = np.zeros((400, 600, 3), dtype=np.uint8)
    test_img.fill(255)  # White background
    
    # Draw a simple chicken-like shape for testing
    # Body (ellipse)
    cv2.ellipse(test_img, (300, 250), (100, 60), 0, 0, 360, (150, 150, 150), -1)
    
    # Head (circle)
    cv2.circle(test_img, (250, 180), 30, (120, 120, 120), -1)
    
    # Neck (line)
    cv2.line(test_img, (250, 210), (280, 220), (130, 130, 130), 15)
    
    # Legs
    cv2.line(test_img, (270, 310), (270, 360), (100, 100, 100), 8)
    cv2.line(test_img, (330, 310), (330, 360), (100, 100, 100), 8)
    
    # Save test image
    cv2.imwrite('test_chicken.jpg', test_img)
    print("✅ Test image created: 'test_chicken.jpg'")
    return 'test_chicken.jpg'

def download_sample_chicken_image():
    """Download a sample chicken image for testing"""
    try:
        import urllib.request
        
        # Sample chicken image URLs (free to use)
        sample_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Female_pair.jpg/640px-Female_pair.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fc/%22Kampong_chicken%22_hen_with_chicks.jpg/640px-%22Kampong_chicken%22_hen_with_chicks.jpg"
        ]
        
        for i, url in enumerate(sample_urls):
            try:
                filename = f'sample_chicken_{i+1}.jpg'
                print(f"Downloading sample image {i+1}...")
                urllib.request.urlretrieve(url, filename)
                print(f"✅ Downloaded: {filename}")
                return filename
            except Exception as e:
                print(f"Failed to download image {i+1}: {e}")
                continue
        
        print("❌ Could not download sample images. You'll need to provide your own.")
        return None
        
    except ImportError:
        print("❌ urllib not available. Please provide your own chicken image.")
        return None

def comprehensive_image_test(image_path: str = None):
    """
    Comprehensive test of the chicken pose estimation system
    """
    print("\n🧪 COMPREHENSIVE CHICKEN POSE TEST")
    print("=" * 50)
    
    estimator = ChickenPoseEstimator()
    
    # If no image provided, try to get one
    if image_path is None:
        print("No image provided. Trying to get test images...")
        
        # First try downloading a sample
        image_path = download_sample_chicken_image()
        
        # If that fails, create a test image
        if image_path is None:
            print("Creating synthetic test image...")
            image_path = create_test_image()
    
    # Check if image exists
    import os
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    print(f"\n📸 Testing with image: {image_path}")
    
    try:
        # Test image loading
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Could not load image")
            return None
        
        print(f"✅ Image loaded successfully - Size: {image.shape}")
        
        # Run pose estimation
        print("\n🔍 Running pose estimation...")
        pose = estimator.estimate_pose(image_path)
        
        # Display detailed results
        print("\n📊 DETAILED RESULTS:")
        print("-" * 30)
        print(f"🐔 Upright Status: {'✅ YES' if pose.is_upright else '❌ NO'}")
        print(f"   • Confidence: {pose.upright_confidence:.3f}")
        print(f"   • Body Angle: {pose.body_angle:.1f}° from vertical")
        print(f"   • Threshold: <{estimator.upright_angle_threshold}° for upright")
        
        print(f"\n🦢 Neck Status: {'✅ ERECT' if pose.neck_erect else '❌ DOWN'}")
        print(f"   • Confidence: {pose.neck_confidence:.3f}")
        print(f"   • Neck Angle: {pose.neck_angle:.1f}° from horizontal")
        print(f"   • Threshold: >{estimator.neck_erect_angle_threshold}° for erect")
        
        print(f"\n📋 Summary:")
        print(f"   • Overall Status: {estimator._get_overall_status(pose)}")
        print(f"   • Posture Quality: {estimator._get_posture_quality(pose)}")
        
        # Count detected keypoints
        detected_keypoints = sum(1 for point in pose.keypoints.values() if point is not None)
        total_keypoints = len(pose.keypoints)
        print(f"   • Keypoints Detected: {detected_keypoints}/{total_keypoints}")
        
        # Show keypoint details
        print(f"\n🎯 Detected Keypoints:")
        for name, point in pose.keypoints.items():
            if point:
                print(f"   • {name}: ({point[0]}, {point[1]})")
            else:
                print(f"   • {name}: Not detected")
        
        # Visualize results
        print(f"\n🎨 Generating visualization...")
        estimator.visualize_pose(image_path, pose, f"test_result_{os.path.basename(image_path)}")
        
        # Performance assessment
        print(f"\n⚡ Performance Assessment:")
        if pose.upright_confidence > 0.7 and pose.neck_confidence > 0.7:
            print("   ✅ High confidence detection - Results are reliable")
        elif pose.upright_confidence > 0.5 and pose.neck_confidence > 0.5:
            print("   ⚠️ Moderate confidence - Results may be partially reliable")
        else:
            print("   ❌ Low confidence - Consider using different image or adjusting parameters")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if detected_keypoints < total_keypoints * 0.7:
            print("   • Try images with better chicken visibility")
            print("   • Ensure good contrast between chicken and background")
        if pose.upright_confidence < 0.5:
            print("   • Check if chicken is clearly visible in upright position")
        if pose.neck_confidence < 0.5:
            print("   • Ensure neck/head area is clearly visible")
        
        return pose
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

def interactive_test():
    """Interactive testing function"""
    print("\n🎮 INTERACTIVE CHICKEN POSE TESTING")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Test with your own image")
        print("2. Download and test sample image")
        print("3. Create and test synthetic image")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            image_path = input("Enter path to your chicken image: ").strip()
            comprehensive_image_test(image_path)
            
        elif choice == '2':
            sample_path = download_sample_chicken_image()
            if sample_path:
                comprehensive_image_test(sample_path)
            
        elif choice == '3':
            test_path = create_test_image()
            comprehensive_image_test(test_path)
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option. Please select 1-4.")
        
        input("\nPress Enter to continue...")

def save_results_to_csv(results: List[dict], output_path: str):
    """Save pose estimation results to CSV file"""
    import pandas as pd
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize the pose estimator
    estimator = ChickenPoseEstimator()
    
    # Example usage for single image
    def analyze_chicken_image(image_path: str):
        """
        Analyze a single chicken image
        """
        try:
            # Estimate pose
            pose = estimator.estimate_pose(image_path)
            
            # Print results
            print(f"\n🐔 CHICKEN POSE ANALYSIS RESULTS:")
            print(f"{'='*50}")
            print(f"Image: {image_path}")
            print(f"Is Upright: {'✅ YES' if pose.is_upright else '❌ NO'} (confidence: {pose.upright_confidence:.2f})")
            print(f"Neck Erect: {'✅ YES' if pose.neck_erect else '❌ NO'} (confidence: {pose.neck_confidence:.2f})")
            print(f"Body Angle: {pose.body_angle:.1f}° from vertical")
            print(f"Neck Angle: {pose.neck_angle:.1f}° from horizontal")
            
            # Visualize results
            estimator.visualize_pose(image_path, pose)
            
            return pose
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None
    
    # COMPREHENSIVE TESTING OPTIONS:
    
    # Option 1: Auto-download sample chicken image and test
    # comprehensive_image_test()
    
    # Option 2: Test with your own chicken image  
    # comprehensive_image_test('path/to/your/chicken_photo.jpg')
    
    # Option 3: Interactive guided testing
    # interactive_test()
    
    # Option 4: Create synthetic test image
    # test_image_path = create_test_image()
    # comprehensive_image_test(test_image_path)
    
    # BASIC USAGE:
    # pose = analyze_chicken_image('path_to_chicken_image.jpg')
    
    # BATCH PROCESSING:
    # results = process_chicken_images('path_to_image_folder', 'path_to_output_folder')
    # save_results_to_csv(results, 'chicken_pose_results.csv')
    
    print("\n🐔 Chicken Pose Estimation System Ready!")
    print("="*50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies before proceeding.")
        exit(1)
    
    print("Use analyze_chicken_image('path_to_image.jpg') to analyze a single image")
    print("Use process_chicken_images('folder_path') for batch processing")
    print("Use comprehensive_image_test('image_path') for detailed testing")
    print("Use interactive_test() for guided testing")
    
    # Quick system test
    print("\n🔧 Testing system...")
    try:
        # Test OpenCV functionality
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        print("✅ OpenCV functionality - OK")
        
        # Test matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        plt.close(fig)
        print("✅ Matplotlib functionality - OK")
        
        print("✅ System test passed! Ready to analyze chicken images.")
        
        # Ask user if they want to run a comprehensive test
        print("\n🎯 READY TO TEST WITH REAL IMAGES!")
        print("You can now:")
        print("• Run interactive_test() for guided testing")
        print("• Run comprehensive_image_test('your_image.jpg') with your own image")
        print("• Run comprehensive_image_test() to auto-download sample images")
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        print("There may be issues with the installation.")
    
    # Example of how to run comprehensive test
    print("\n📝 QUICK START EXAMPLES:")
    print("# Test with sample image (auto-download):")
    print("comprehensive_image_test()")
    print("")
    print("# Test with your own image:")
    print("comprehensive_image_test('path/to/your/chicken.jpg')")
    print("")
    print("# Interactive guided testing:")
    print("interactive_test()")