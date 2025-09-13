#!/bin/bash
set -e

# Simplified Flashlight build script targeting the specific trie limit file
# Usage: ./build_flashlight_custom_simple.sh [TRIE_LIMIT]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/flashlight_custom_build"
TRIE_LIMIT=${1:-50}  # Default to 50 if not specified

echo "🔧 Building custom Flashlight with trie label limit: $TRIE_LIMIT"
echo "📁 Build directory: $BUILD_DIR"

# Clean up previous build
if [ -d "$BUILD_DIR" ]; then
    echo "🧹 Cleaning up previous build..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone Flashlight text repository
echo "📥 Cloning Flashlight text repository..."
git clone --recursive https://github.com/flashlight/text.git
cd text

# Target the specific file you found
TARGET_FILE="flashlight/lib/text/decoder/Trie.h"
echo "🔍 Looking for trie limit in: $TARGET_FILE"

if [ ! -f "$TARGET_FILE" ]; then
    echo "❌ Could not find $TARGET_FILE"
    echo "   Directory structure:"
    find . -name "Trie.h" -type f 2>/dev/null | head -5
    exit 1
fi

echo "📝 Found target file: $TARGET_FILE"

# Show original content
echo "📄 Original trie limit definition:"
grep -n "constexpr.*kTrieMaxLabel" "$TARGET_FILE" || echo "   (Pattern not found, showing context)"
grep -A2 -B2 "kTrieMaxLabel\|6" "$TARGET_FILE" | head -10

# Backup and modify the file
echo "🔧 Modifying trie limit..."
cp "$TARGET_FILE" "$TARGET_FILE.backup"

# Use the exact pattern you found: constexpr int kTrieMaxLabel = 6;
sed -i.tmp "s/constexpr int kTrieMaxLabel = 6;/constexpr int kTrieMaxLabel = $TRIE_LIMIT;/g" "$TARGET_FILE"
rm -f "$TARGET_FILE.tmp"

# Verify the change
echo "📄 Modified trie limit definition:"
grep -n "constexpr.*kTrieMaxLabel" "$TARGET_FILE"

if ! grep -q "kTrieMaxLabel = $TRIE_LIMIT" "$TARGET_FILE"; then
    echo "❌ Failed to modify trie limit. Manual check required."
    echo "   File location: $PWD/$TARGET_FILE"
    exit 1
fi

echo "✅ Successfully modified trie limit to $TRIE_LIMIT"

# Check if we have the required dependencies
echo "🔍 Checking build dependencies..."

# Check for cmake
if ! command -v cmake &> /dev/null; then
    echo "❌ cmake is required but not installed"
    echo "   Install with: brew install cmake"
    exit 1
fi

# Check for python development headers
python3 -c "import pybind11" 2>/dev/null || {
    echo "❌ pybind11 is required but not installed"
    echo "   Install with: pip install pybind11"
    exit 1
}

# Install KenLM first
echo "📦 Installing KenLM..."
if ! command -v kenlm &> /dev/null; then
    # Clone and build KenLM
    echo "   Cloning KenLM repository..."
    git clone https://github.com/kpu/kenlm.git ../kenlm
    cd ../kenlm
    
    echo "   Building KenLM..."
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
    make -j$(nproc 2>/dev/null || echo 4)
    
    # Install KenLM system-wide (requires sudo)
    echo "   Installing KenLM (may require password)..."
    sudo make install
    
    # Return to text directory
    cd ../../text
else
    echo "   KenLM already installed"
fi

# Create build directory and configure
echo "🏗️  Configuring build..."
mkdir -p build
cd build

# Configure with cmake (with KenLM support)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DFL_TEXT_BUILD_PYTHON=ON \
    -DFL_TEXT_BUILD_STANDALONE=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DFL_TEXT_USE_KENLM=ON

# Build the project
echo "🔨 Building Flashlight text (this may take several minutes)..."
make -j$(nproc 2>/dev/null || echo 4)

# Install the Python package
echo "📦 Installing custom Flashlight text..."
cd bindings/python
pip install -e .

echo "🎉 Custom Flashlight text with trie limit $TRIE_LIMIT built and installed!"
echo "💡 Test with: python -c \"from flashlight.lib.text.decoder import Trie; print('Success!')\""

# Clean up build artifacts to save space (optional)
echo "🧹 Cleaning up build artifacts..."
cd "$BUILD_DIR"
rm -rf text/build

echo "✅ Build complete! Custom Flashlight is ready to use."
