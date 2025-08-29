# EGGO C++ Interface

This directory contains a C++ interface to the EGGO Julia package that allows calling EGGO functions from C++ with optimal performance.

## Features

✅ **Load models once** - Heavy models (Green's functions, basis functions, neural networks) are loaded once and cached  
✅ **Fast predictions** - Subsequent predictions are extremely fast (~1ms each)  
✅ **Julia integration** - Uses existing, tested Julia implementation  
✅ **Identical results** - Same algorithms and accuracy as Julia version  

## Performance

- **Model loading:** ~5-6 seconds (one-time cost)
- **First prediction:** ~1.8 seconds (includes JIT compilation)  
- **Subsequent predictions:** ~1ms each ⚡

## Requirements

- Julia installed and in PATH
- EGGO package available (`using EGGO` works)
- CMake 3.16+
- C++17 compatible compiler

## Quick Start

```bash
# Build
mkdir build && cd build
cmake ..
make

# Run example
./run_example
```

## Example Output

```
Loading EGGO models (one-time initialization)...
✅ Models loaded and cached in Julia globals
Model loading took: 5362 ms

=== Running multiple predictions with cached models ===
Running prediction for shot 168830 (using cached models)...
✅ Prediction completed!
   PSI coefficients: 46 elements
   1D profile coefficients: 35 elements
Prediction 1 took: 1751 ms

Running prediction for shot 168831 (using cached models)...
✅ Prediction completed!
Prediction 2 took: 1 ms

Running prediction for shot 168832 (using cached models)...
✅ Prediction completed!  
Prediction 3 took: 1 ms
```

## Usage in Your Code

```cpp
#include "run_example.cpp" // Or create a header/library

// Initialize once
OptimizedJuliaEGGO eggo;
eggo.load_models_once();  // Heavy operation, do once

// Fast predictions
auto [y_psi, y1d] = eggo.run_prediction(
    shot, expsi, fwtsi, expmp2, fwtmp2, fcurrt, ecurrt, Ip
);
```

## Architecture

1. **Julia Runtime Embedding** - Embeds Julia in C++ process
2. **Model Caching** - Stores models as Julia `const` globals
3. **Type Conversion** - Automatic conversion between C++ vectors and Julia arrays
4. **Error Handling** - Catches and reports Julia exceptions

## Files

- `run_example.cpp` - Complete C++ interface with example
- `CMakeLists.txt` - Build configuration  
- `README.md` - This documentation

Perfect for:
- **Batch processing** multiple shots
- **Real-time applications** requiring fast predictions
- **Web services** serving multiple requests  
- **Parameter studies** with different diagnostic data