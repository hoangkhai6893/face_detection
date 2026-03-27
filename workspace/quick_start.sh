#!/bin/bash
# 🎯 Quick Start - Face Recognition System
# Simple launcher script for the face recognition system

echo "🎯 FAMILY FACE RECOGNITION SYSTEM"
echo "=================================="
echo ""

# Change to correct directory
cd /home/dkhai/workspace/src

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}📁 Current directory: $(pwd)${NC}"
echo ""

# Check if main script exists
if [ -f "improved_hybrid_recognition.py" ]; then
    echo -e "${GREEN}✅ Main system found${NC}"
    echo ""
    
    echo -e "${YELLOW}🚀 AVAILABLE COMMANDS:${NC}"
    echo "1️⃣  python improved_hybrid_recognition.py  # Main recognition system"
    echo "2️⃣  python face_learning_mode.py          # Train new people"
    echo "3️⃣  python launcher.py                    # System menu"
    echo "4️⃣  python optimize_encodings.py          # Speed optimization"
    echo ""
    
    echo -e "${BLUE}📖 Quick Usage:${NC}"
    echo "• For daily use: Run command 1️⃣"
    echo "• To add people: Run command 2️⃣" 
    echo "• For easy menu: Run command 3️⃣"
    echo "• To speed up: Run command 4️⃣"
    echo ""
    
    echo -e "${GREEN}🎯 Starting main recognition system in 3 seconds...${NC}"
    echo "Press Ctrl+C to cancel"
    
    sleep 3
    python improved_hybrid_recognition.py
    
else
    echo "❌ System files not found in expected location"
    echo "📁 Checking current directory contents:"
    ls -la *.py 2>/dev/null || echo "No Python files found"
    echo ""
    echo "💡 Tip: Make sure you're in the correct directory"
    echo "Try: cd /home/dkhai/workspace/src"
fi