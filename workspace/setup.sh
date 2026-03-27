#!/bin/bash
# 🚀 Face Recognition System Quick Setup
# Run this script to verify and set up your face recognition system

echo "🎯 FACE RECOGNITION SYSTEM SETUP"
echo "=================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ "$2" = "OK" ]; then
        echo -e "${GREEN}✅ $1${NC}"
    elif [ "$2" = "WARNING" ]; then
        echo -e "${YELLOW}⚠️  $1${NC}"
    elif [ "$2" = "ERROR" ]; then
        echo -e "${RED}❌ $1${NC}"
    else
        echo -e "${BLUE}ℹ️  $1${NC}"
    fi
}

# Check Python version
echo -e "\n📋 CHECKING SYSTEM REQUIREMENTS"
echo "--------------------------------"

python_version=$(python3 --version 2>/dev/null | cut -d" " -f2)
if [ $? -eq 0 ]; then
    print_status "Python version: $python_version" "OK"
else
    print_status "Python 3 not found - please install Python 3.8+" "ERROR"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    print_status "pip3 is available" "OK"
else
    print_status "pip3 not found - please install pip" "ERROR"
    exit 1
fi

# Check and install required packages
echo -e "\n📦 CHECKING PYTHON PACKAGES"
echo "-----------------------------"

packages=("opencv-python" "numpy" "ultralytics" "face-recognition")
missing_packages=()

for package in "${packages[@]}"; do
    if python3 -c "import ${package//-/_}" 2>/dev/null; then
        print_status "$package installed" "OK"
    else
        print_status "$package missing" "WARNING"
        missing_packages+=("$package")
    fi
done

# Install missing packages
if [ ${#missing_packages[@]} -gt 0 ]; then
    echo -e "\n🔧 INSTALLING MISSING PACKAGES"
    echo "-------------------------------"
    for package in "${missing_packages[@]}"; do
        echo "Installing $package..."
        pip3 install "$package"
        if [ $? -eq 0 ]; then
            print_status "$package installed successfully" "OK"
        else
            print_status "Failed to install $package" "ERROR"
        fi
    done
fi

# Check project structure
echo -e "\n📁 CHECKING PROJECT STRUCTURE"
echo "------------------------------"

required_dirs=("src" "family_images")
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        print_status "Directory $dir exists" "OK"
    else
        print_status "Creating directory $dir" "WARNING"
        mkdir -p "$dir"
    fi
done

# Check for YOLO model
if [ -f "src/yolov11n-face.pt" ] || [ -f "src/src/yolov11n-face.pt" ]; then
    print_status "YOLO model file found" "OK"
else
    print_status "YOLO model file missing - please ensure yolov11n-face.pt is in src/" "WARNING"
fi

# Check required Python scripts
echo -e "\n🐍 CHECKING PYTHON SCRIPTS"
echo "---------------------------"

required_scripts=(
    "src/improved_hybrid_recognition.py"
    "src/face_learning_mode.py" 
    "src/launcher.py"
    "src/optimize_encodings.py"
)

# If scripts not found in expected location, try alternate path
for script in "${required_scripts[@]}"; do
    if [ ! -f "$script" ]; then
        alt_script="src/$(basename $script)"
        if [ -f "$alt_script" ]; then
            required_scripts=("${required_scripts[@]/$script/$alt_script}")
        fi
    fi
done

for script in "${required_scripts[@]}"; do
    if [ -f "$script" ]; then
        print_status "$(basename $script) found" "OK"
    else
        print_status "$(basename $script) missing" "ERROR"
    fi
done

# Check dataset
echo -e "\n👥 CHECKING DATASET"
echo "-------------------"

if [ -d "family_images" ]; then
    person_count=$(find family_images -mindepth 1 -maxdepth 1 -type d | wc -l)
    if [ "$person_count" -gt 0 ]; then
        print_status "Found $person_count people in dataset" "OK"
        
        # Count images for each person
        for person_dir in family_images/*/; do
            if [ -d "$person_dir" ]; then
                person_name=$(basename "$person_dir")
                image_count=$(find "$person_dir" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
                if [ "$image_count" -gt 10 ]; then
                    print_status "$person_name: $image_count images" "OK"
                elif [ "$image_count" -gt 0 ]; then
                    print_status "$person_name: $image_count images (recommend 20+)" "WARNING"
                else
                    print_status "$person_name: no images found" "ERROR"
                fi
            fi
        done
    else
        print_status "No people found in dataset - use learning mode to add" "WARNING"
    fi
else
    print_status "Dataset directory missing" "ERROR"
fi

# Check camera access
echo -e "\n📷 CHECKING CAMERA ACCESS"
echo "-------------------------"

if python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('Camera accessible')
    cap.release()
    exit(0)
else:
    print('Camera not accessible')
    exit(1)
" 2>/dev/null; then
    print_status "Camera is accessible" "OK"
else
    print_status "Camera not accessible - check permissions/connections" "WARNING"
fi

# Performance test
echo -e "\n⚡ PERFORMANCE TEST"
echo "------------------"

if python3 -c "
import time
import numpy as np
# Simple performance test
start = time.time()
for i in range(1000):
    arr = np.random.random((100, 100))
    result = np.mean(arr)
end = time.time()
print(f'Performance test: {end-start:.3f}s')
" 2>/dev/null; then
    print_status "System performance test passed" "OK"
else
    print_status "Performance test failed" "WARNING"
fi

# Final recommendations
echo -e "\n🎯 SETUP COMPLETE - NEXT STEPS"
echo "==============================="

echo -e "${BLUE}📖 Quick Start Guide:${NC}"
echo "1. 📚 Add training data:"
echo "   python src/face_learning_mode.py"
echo ""
echo "2. ⚡ Optimize performance:"
echo "   python src/optimize_encodings.py"
echo ""
echo "3. 🚀 Start recognition:"
echo "   python src/launcher.py"
echo "   # OR directly:"
echo "   python src/improved_hybrid_recognition.py"
echo ""

echo -e "${BLUE}📋 Important Notes:${NC}"
echo "• Collect 20-50 images per person for best accuracy"
echo "• Use good lighting and clear face shots"
echo "• Run optimization after adding new people"
echo "• Check README.md for detailed usage instructions"

echo -e "\n${GREEN}🎉 Setup verification completed!${NC}"
echo "Run 'python src/launcher.py' to get started."