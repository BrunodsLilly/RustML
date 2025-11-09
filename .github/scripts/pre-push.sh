#!/bin/bash
# Pre-push validation script
# Run this before pushing to catch issues early

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track timing
SCRIPT_START=$(date +%s)

echo -e "${BLUE}🚀 RustML Pre-Push Validation${NC}"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}❌ Error: Cargo.toml not found. Run this script from the project root.${NC}"
    exit 1
fi

# Function to print step headers
print_step() {
    echo ""
    echo -e "${BLUE}▶ $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Step 1: Format check
print_step "1/6 Checking code formatting..."
START=$(date +%s)
if cargo fmt --all -- --check; then
    END=$(date +%s)
    print_success "Code formatting is correct ($(($END - $START))s)"
else
    echo -e "${RED}❌ Formatting check failed. Run: cargo fmt --all${NC}"
    exit 1
fi

# Step 2: Clippy linting
print_step "2/6 Running Clippy linter..."
START=$(date +%s)
if cargo clippy --all-targets --all-features -- -D warnings; then
    END=$(date +%s)
    print_success "Clippy checks passed ($(($END - $START))s)"
else
    echo -e "${RED}❌ Clippy found issues. Run: cargo clippy --fix${NC}"
    exit 1
fi

# Step 3: Build all packages
print_step "3/6 Building all packages..."
START=$(date +%s)
if cargo build --workspace; then
    END=$(date +%s)
    print_success "Build successful ($(($END - $START))s)"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

# Step 4: Run all tests
print_step "4/6 Running all tests..."
START=$(date +%s)
if cargo test --workspace; then
    END=$(date +%s)
    print_success "All tests passed ($(($END - $START))s)"
else
    echo -e "${RED}❌ Tests failed${NC}"
    exit 1
fi

# Step 5: WASM build check
print_step "5/6 Building WASM application..."
START=$(date +%s)
cd web

# Check if dioxus CLI is installed
if ! command -v dx &> /dev/null; then
    print_warning "Dioxus CLI not found. Install with: cargo install dioxus-cli"
    print_warning "Skipping WASM build check..."
    cd ..
else
    if dx build --platform web --release; then
        END=$(date +%s)

        # Check bundle size
        WASM_FILE=$(find target/dx/web/release/web/public -name "*.wasm" 2>/dev/null | head -n 1)
        if [ -n "$WASM_FILE" ]; then
            WASM_SIZE=$(ls -lh "$WASM_FILE" | awk '{print $5}')
            WASM_BYTES=$(ls -l "$WASM_FILE" | awk '{print $5}')

            print_success "WASM build successful ($(($END - $START))s) - Bundle size: $WASM_SIZE"

            # Warn if bundle is large
            if [ "$WASM_BYTES" -gt 5242880 ]; then
                print_warning "WASM bundle exceeds 5MB ($WASM_SIZE)"
            fi
        else
            print_success "WASM build successful ($(($END - $START))s)"
        fi
    else
        echo -e "${RED}❌ WASM build failed${NC}"
        cd ..
        exit 1
    fi
    cd ..
fi

# Step 6: Security audit
print_step "6/6 Running security audit..."
START=$(date +%s)

# Check if cargo-audit is installed
if ! command -v cargo-audit &> /dev/null; then
    print_warning "cargo-audit not found. Install with: cargo install cargo-audit"
    print_warning "Skipping security audit..."
else
    if cargo audit; then
        END=$(date +%s)
        print_success "Security audit passed ($(($END - $START))s)"
    else
        echo -e "${YELLOW}⚠️  Security vulnerabilities found. Review and update dependencies.${NC}"
        # Don't fail on security audit, just warn
    fi
fi

# Summary
SCRIPT_END=$(date +%s)
TOTAL_TIME=$(($SCRIPT_END - $SCRIPT_START))

echo ""
echo "=================================="
echo -e "${GREEN}🎉 All checks passed!${NC}"
echo "Total time: ${TOTAL_TIME}s"
echo ""
echo -e "${BLUE}Safe to push! 🚀${NC}"
echo ""
