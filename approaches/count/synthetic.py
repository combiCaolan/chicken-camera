#!/usr/bin/env python3
"""
Generate synthetic test video for chicken counting system using real chicken images
Creates moving chicken images that simulate chickens on a conveyor belt
"""

import cv2
import numpy as np
import random
import math
import os
import urllib.request
from pathlib import Path

def download_chicken_images():
    """
    Download sample chicken images for testing
    You can replace these URLs with your own chicken images
    """
    # Create images directory
    img_dir = Path("chicken_images")
    img_dir.mkdir(exist_ok=True)
    
    # Sample chicken image URLs (free stock images)
    # Replace these with your own chicken images for better results
    chicken_urls = [
        "https://images.unsplash.com/photo-1548550023-2bdb3c5beed7?w=400",  # Chicken 1
        "https://images.unsplash.com/photo-1612817288484-6f916006741a?w=400",  # Chicken 2
        "https://images.unsplash.com/photo-1599429044984-1ec8ac1ce6d0?w=400",  # Chicken 3
        "https://images.unsplash.com/photo-1516467508483-a7212febe31a?w=400",  # Chicken 4
    ]
    
    downloaded_images = []
    
    for i, url in enumerate(chicken_urls):
        filename = img_dir / f"chicken_{i+1}.jpg"
        
        if not filename.exists():
            try:
                print(f"Downloading chicken image {i+1}...")
                urllib.request.urlretrieve(url, filename)
                print(f"Downloaded: {filename}")
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                continue
        
        if filename.exists():
            downloaded_images.append(str(filename))
    
    return downloaded_images

def load_and_process_chicken_images(image_paths, target_sizes=[(80, 60), (100, 75), (120, 90)]):
    """
    Load chicken images and prepare them for use in the video
    
    Args:
        image_paths: List of paths to chicken images
        target_sizes: List of (width, height) tuples for different chicken sizes
    
    Returns:
        List of processed chicken image variants
    """
    processed_chickens = []
    
    for img_path in image_paths:
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not load image: {img_path}")
                continue
            
            # Create multiple size variants of each chicken
            for target_size in target_sizes:
                # Resize image
                resized = cv2.resize(img, target_size)
                
                # Optional: Remove background (simple method)
                # This is basic - you might want to use more sophisticated background removal
                processed = remove_simple_background(resized)
                
                # Create different orientations
                for angle in [0, -5, 5, -10, 10]:  # Slight rotations for variety
                    rotated = rotate_image(processed, angle)
                    processed_chickens.append({
                        'image': rotated,
                        'size': target_size,
                        'original_path': img_path
                    })
                    
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Processed {len(processed_chickens)} chicken variants")
    return processed_chickens

def remove_simple_background(image, threshold=50):
    """
    Simple background removal - assumes background is relatively uniform
    For better results, use more sophisticated methods or pre-processed images
    """
    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for non-background areas (this is very basic)
    # You might want to adjust these values based on your images
    lower_bound = np.array([0, 30, 30])
    upper_bound = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Create 4-channel image (BGR + Alpha)
    result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask  # Set alpha channel
    
    return result

def rotate_image(image, angle):
    """Rotate image by given angle while maintaining size"""
    if angle == 0:
        return image
        
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate image
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_TRANSPARENT)
    
    return rotated

def overlay_image_with_alpha(background, overlay, x, y):
    """
    Overlay an image with alpha channel onto background
    """
    if overlay.shape[2] != 4:  # No alpha channel
        h, w = overlay.shape[:2]
        if 0 <= y < background.shape[0] - h and 0 <= x < background.shape[1] - w:
            background[y:y+h, x:x+w] = overlay
        return background
    
    # Handle alpha channel
    h, w = overlay.shape[:2]
    
    # Ensure overlay fits within background
    if y + h > background.shape[0] or x + w > background.shape[1] or x < 0 or y < 0:
        return background
    
    # Extract alpha channel
    alpha = overlay[:, :, 3] / 255.0
    
    # Blend images
    for c in range(3):  # BGR channels
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c] + 
            (1 - alpha) * background[y:y+h, x:x+w, c]
        )
    
    return background

def create_test_video_with_images(chicken_images, output_path='realistic_chicken_test.mp4', duration=30, fps=30):
    """
    Create synthetic test video using real chicken images
    
    Args:
        chicken_images: List of processed chicken image dictionaries
        output_path: Output video file path
        duration: Video duration in seconds
        fps: Frames per second
    """
    if not chicken_images:
        print("No chicken images available! Please download or provide chicken images.")
        return
    
    width, height = 1280, 720
    total_frames = duration * fps
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create chicken objects with random properties
    chickens = []
    for i in range(random.randint(8, 15)):  # 8-15 chickens
        chicken_variant = random.choice(chicken_images)
        
        chicken = {
            'id': i,
            'x': random.randint(-200, -50),  # Start off-screen left
            'y': random.randint(height//4, 3*height//4),
            'speed_x': random.uniform(1.0, 3.5),  # Pixels per frame
            'speed_y': random.uniform(-0.3, 0.3),  # Slight vertical variation
            'image_data': chicken_variant,
            'wobble_freq': random.uniform(0.02, 0.08),  # For natural movement
            'spawn_frame': random.randint(0, total_frames//3),  # When to appear
            'scale_factor': random.uniform(0.8, 1.3),  # Size variation
            'vertical_offset': random.uniform(-10, 10)  # Slight height variation
        }
        chickens.append(chicken)
    
    print(f"Creating realistic chicken video with {len(chickens)} chickens...")
    print(f"Using {len(chicken_images)} chicken image variants")
    print(f"Duration: {duration}s, FPS: {fps}, Total frames: {total_frames}")
    
    for frame_num in range(total_frames):
        # Create conveyor belt background
        frame = create_conveyor_background(width, height, frame_num)
        
        # Add chickens
        chickens_in_frame = 0
        for chicken in chickens:
            # Only show chicken after its spawn time
            if frame_num < chicken['spawn_frame']:
                continue
                
            # Update position with wobble
            chicken['x'] += chicken['speed_x']
            wobble = math.sin(frame_num * chicken['wobble_freq']) * 2
            chicken['y'] += chicken['speed_y'] + wobble + chicken['vertical_offset'] * 0.01
            
            # Skip if completely off-screen (right side)
            if chicken['x'] > width + 200:
                continue
            
            # Get chicken image
            chicken_img = chicken['image_data']['image'].copy()
            
            # Scale image if needed
            if chicken['scale_factor'] != 1.0:
                h, w = chicken_img.shape[:2]
                new_w = int(w * chicken['scale_factor'])
                new_h = int(h * chicken['scale_factor'])
                chicken_img = cv2.resize(chicken_img, (new_w, new_h))
            
            # Calculate position
            img_h, img_w = chicken_img.shape[:2]
            pos_x = int(chicken['x'] - img_w // 2)
            pos_y = int(chicken['y'] - img_h // 2)
            
            # Only draw if at least partially visible
            if pos_x < width and pos_x + img_w > 0 and pos_y < height and pos_y + img_h > 0:
                # Overlay chicken image onto frame
                frame = overlay_image_with_alpha(frame, chicken_img, pos_x, pos_y)
                
                # Add debug info (chicken ID)
                label_pos = (max(0, pos_x), max(20, pos_y - 5))
                cv2.putText(frame, f"C{chicken['id']}", label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                chickens_in_frame += 1
        
        # Add frame information
        info_text = [
            f"Frame: {frame_num}",
            f"Time: {frame_num/fps:.1f}s",
            f"Chickens visible: {chickens_in_frame}",
            f"Using real chicken images"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw counting line
        line_y = height // 2
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 3)
        cv2.putText(frame, "COUNTING LINE", (10, line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        out.write(frame)
        
        # Progress indicator
        if frame_num % (fps * 5) == 0:  # Every 5 seconds
            print(f"Progress: {frame_num/total_frames*100:.1f}%")
    
    out.release()
    print(f"Realistic chicken test video created: {output_path}")

def create_conveyor_background(width, height, frame_num):
    """Create a moving conveyor belt background"""
    # Base color (industrial gray)
    frame = np.ones((height, width, 3), dtype=np.uint8) * 65
    
    # Add conveyor belt texture (moving horizontal lines)
    belt_speed = 2  # pixels per frame
    line_spacing = 25
    
    for y in range(0, height, line_spacing):
        # Moving lines to simulate belt movement
        offset = (frame_num * belt_speed) % line_spacing
        adjusted_y = y + offset
        
        if 0 <= adjusted_y < height:
            # Lighter lines for belt texture
            cv2.line(frame, (0, int(adjusted_y)), (width, int(adjusted_y)), 
                    (85, 85, 85), 1)
    
    # Add belt edges
    belt_top = height // 6
    belt_bottom = 5 * height // 6
    cv2.line(frame, (0, belt_top), (width, belt_top), (40, 40, 40), 3)
    cv2.line(frame, (0, belt_bottom), (width, belt_bottom), (40, 40, 40), 3)
    
    return frame

def create_fallback_chicken_images():
    """
    Create simple programmatic chicken images if no real images are available
    """
    print("Creating fallback chicken images...")
    
    fallback_chickens = []
    colors = [
        (180, 140, 100),  # Brown
        (200, 160, 120),  # Light brown
        (160, 120, 80),   # Dark brown
        (220, 180, 140),  # Cream
    ]
    
    sizes = [(80, 60), (100, 75), (120, 90)]
    
    for color in colors:
        for size in sizes:
            # Create chicken-shaped image
            img = np.zeros((size[1], size[0], 4), dtype=np.uint8)  # BGRA
            
            # Main body (ellipse)
            center = (size[0]//2, size[1]//2)
            axes = (size[0]//3, size[1]//3)
            cv2.ellipse(img, center, axes, 0, 0, 360, color + (255,), -1)
            
            # Head (circle)
            head_center = (size[0]//3, size[1]//3)
            head_radius = size[1]//4
            cv2.circle(img, head_center, head_radius, color + (255,), -1)
            
            # Beak (triangle)
            beak_points = np.array([
                [head_center[0] - head_radius//2, head_center[1]],
                [head_center[0] - head_radius, head_center[1] - 5],
                [head_center[0] - head_radius, head_center[1] + 5]
            ], np.int32)
            cv2.fillPoly(img, [beak_points], (50, 50, 50, 255))
            
            fallback_chickens.append({
                'image': img,
                'size': size,
                'original_path': 'fallback'
            })
    
    return fallback_chickens

if __name__ == "__main__":
    print("=== Chicken Test Video Generator with Real Images ===")
    
    # Try to download chicken images
    print("Step 1: Downloading chicken images...")
    image_paths = download_chicken_images()
    
    # Process chicken images
    chicken_images = []
    if image_paths:
        print("Step 2: Processing downloaded images...")
        chicken_images = load_and_process_chicken_images(image_paths)
    
    # Fallback to programmatic chickens if no real images
    if not chicken_images:
        print("Step 2: No real images available, creating fallback chickens...")
        chicken_images = create_fallback_chicken_images()
    
    if not chicken_images:
        print("ERROR: No chicken images available!")
        exit(1)
    
    # Create test videos
    print("Step 3: Creating test videos...")
    
    # Basic test with real chicken images
    create_test_video_with_images(
        chicken_images, 
        'realistic_chicken_test.mp4', 
        duration=30, 
        fps=30
    )
    
    # Longer test with more variety
    create_test_video_with_images(
        chicken_images, 
        'long_chicken_test.mp4', 
        duration=60, 
        fps=30
    )
    
    print("\n=== Test Videos Created! ===")
    print("Files created:")
    print("- realistic_chicken_test.mp4 (30 seconds)")
    print("- long_chicken_test.mp4 (60 seconds)")
    print("\nTo test with your chicken counter:")
    print("python chicken_counter.py --input realistic_chicken_test.mp4 --output counted_realistic.mp4")
    print("\nTip: Replace images in 'chicken_images/' folder with your own chicken photos for better results!")