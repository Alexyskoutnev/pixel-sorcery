# NAFNet Mobile Inference - Debugging Guide

## Overview

This document captures all issues, investigations, and solutions encountered while implementing NAFNet real estate image enhancement on iOS using Flutter and ONNX Runtime.

**Goal**: Run NAFNet model inference on-device for real estate photo enhancement with two modes:
- **1080p Mode**: Fast processing (target max 768px)
- **Full Scale Mode**: Best quality (target max 1024px)

**Tech Stack**:
- Flutter + Cupertino (iOS)
- `flutter_onnxruntime` (1.6.0) for model inference
- `image` package (`^4.3.0`, currently resolving to `4.6.0`) for image manipulation
- NAFNet model: Trained for real estate image denoising/enhancement

---

## Timeline of Issues & Solutions

### Issue 1: Memory Crash on Large Images (SOLVED)

**Symptom**:
```
Memory limit exceeded: 6144 MB
```
App crashed when processing images larger than ~2MP without tiling.

**Root Cause**:
- Full image tensor requires ~72.4 MB for a 1206x2622 portrait image
- Model execution + intermediate buffers exceeded iOS memory limits
- Mobile devices have strict memory constraints (3-8GB total RAM, app limit ~1-2GB)

**Solution**:
✅ **Implemented tiled inference**
- Break image into overlapping tiles (e.g., 512x512 with 96px overlap)
- Process each tile independently through the model
- Blend tiles back together with smooth weighting at edges
- Dispose native ONNX tensors per tile to avoid memory growth over long runs

**Code Location**: `lib/services/onnx_inference_service.dart::_processWithTiles()`

**Configuration**:
```dart
// Per-mode settings
1080p Mode:
  - maxDimension: 768 (or null for unlimited)
  - enableTiling: false (or true for large images)
  - tileSize: 256-512

Full Scale Mode:
  - maxDimension: 1024 (or null for unlimited)
  - enableTiling: true
  - tileSize: 384-512
  - tileOverlap: 32-96
```

---

### Issue 2: ONNX Output Type Casting Error (SOLVED)

**Symptom**:
```
type 'List<dynamic>' is not a subtype of type 'num'
```

**Root Cause**:
Early implementation used `OrtValue.asList()`, which returns a nested structure for multi-dimensional tensors. Treating it like a flat numeric list caused type errors.

**Solution**:
✅ **Preferred fix: use `asFlattenedList()`**
- `OrtValue.asFlattenedList()` returns the tensor data as a single flat buffer
- On iOS, this comes back as `Float32List`, so we can avoid reshape/flatten overhead entirely

✅ **Additional hardening: dispose ORT tensors**
- When tiling, each tile produces input/output `OrtValue`s; if they are not disposed, native memory grows across tiles

**Code Location**: `lib/services/onnx_inference_service.dart::_runInference()`

---

### Issue 3: Tiling Artifacts on Vertical Lines (REPORTED BY USER)

**Symptom**:
User reported jitter/artifacts on vertical lines when using tiling, especially at lower resolutions.

**Root Cause**:
- Insufficient tile overlap causing visible seams
- Tile size too small relative to image features
- Blending mode not smooth enough

**Solutions Attempted**:
✅ Increased tile sizes (256 → 512px)
✅ Increased overlap (32 → 96px)
✅ Added configurable blend modes (linear, cosine, gaussian)
✅ Made tiling/overlap per-mode configurable

**Configuration Options**:
```dart
enum BlendMode {
  none,     // No blending (hard edges)
  linear,   // Linear interpolation
  cosine,   // Cosine interpolation (smooth)
  gaussian, // Gaussian-like smoothstep
}
```

**User Feedback**: Still testing optimal settings for different image types.

---

### Issue 4: Runtime Settings Not Applied (SOLVED)

**Symptom**:
User changed settings in UI (e.g., enabled tiling, switched model), but processing still used old settings.

**Root Cause**:
State management issue - settings were being passed by reference but not explicitly during processing.

**Solution**:
✅ **Explicit settings passing**
```dart
// In ImageEnhancementProvider.processImage():
final outputBytes = await _inference.processImage(
  File(image.inputPath),
  image.mode,
  onProgress: (progress) { ... },
  runtimeSettings: settingsProvider,  // ← Explicit pass
);
```

✅ **Connected providers in main.dart**:
```dart
inferenceService.settings = settingsProvider;
imageProvider.settingsProvider = settingsProvider;
settingsProvider.onModelChanged = () async {
  await inferenceService.switchModel(settingsProvider.modelType);
};
```

**Code Locations**:
- `lib/main.dart` - Provider wiring
- `lib/providers/image_provider.dart::processImage()` - Explicit settings pass
- `lib/services/onnx_inference_service.dart::processImage()` - Settings consumption

---

### Issue 5: uint16/HDR Input Corruption (FIX IMPLEMENTED - RETEST)

**Symptom**:
Some iOS photos (commonly portrait, but not portrait-specific) produce severely corrupted output:
- Green tint / inverted colors
- Distorted / scrambled pixels
- Visible seams between tiles
- Over-dark / wrong color response (even when tensor shapes look correct)

**Key Evidence (from device logs)**:
The decoded `image` can be **high bit depth** (uint16/HDR). Example:
- `Sample input pixel (0,0): R=43423, G=39967, B=35455`  (values **> 255**)

**Root Cause**:
Preprocessing previously assumed **uint8** pixels:
- It copied pixels into a new default `img.Image(...)` (which is `Format.uint8`)
- It used `setPixel(...)` to copy pixels from the decoded image
- When the decoded image is `Format.uint16` (or other high bit depth), that copy step can **truncate/wrap** channel values into 0–255
- The tensor was then normalized by dividing by 255, producing a plausible 0–1 range but representing the *wrong* colors/values

**Fix Implemented**:
✅ `_preprocessImage(...)` now writes the input tensor **directly** from the decoded source image using:
- `pixel.rNormalized`, `pixel.gNormalized`, `pixel.bNormalized` (always 0–1 regardless of bit depth)
- Reflection padding is applied in tensor space (no intermediate uint8 padded image)

**How to Verify the Fix**:
In logs for the same image:
1. `Input image format ... max=...` should show `max=65535` (or other >255) for these photos
2. `Sample input pixel ... (normalized: ...)` should be ~0–1
3. `Input tensor pixel (0,0)` should match the normalized sample pixel closely

**Current Status**:
Fix is implemented in code; re-test the same portrait image with tiling OFF then ON to confirm visual quality.

**Important Note**:
If you are re-processing the same image ID repeatedly, you may be looking at a stale cached decode. See **Issue 6**.

**Mitigations Implemented (may affect portrait edge cases)**:
✅ Apply `img.bakeOrientation(image)` immediately after decode so EXIF rotation is consistently applied before tiling/inference.

---

### Issue 6: Output Looks Unchanged After Re-processing (SOLVED)

**Symptom**:
You fix preprocessing/inference bugs (logs improve), hit **Retry**, but the UI still shows the *old* corrupted output.

This can make it look like nothing changed even though inference is now producing better results.

**Root Cause**:
Flutter caches decoded images in memory via `ImageCache`.
- `Image.file(...)` uses `FileImage`
- `FileImage` cache key is **(file path + scale)**, not file contents
- If you overwrite the same file path (e.g. `.../outputs/<id>_enhanced.jpg`), Flutter may keep showing the previously cached decode

**Solution Implemented**:
✅ **Version output filenames per processing run**
- Every processing run writes to a new file name: `outputs/<id>_enhanced_<timestamp>.jpg`
- This forces a new `FileImage` cache key, so the UI always loads the latest bytes
- The previous output file is deleted after a successful new run to avoid disk bloat

**Code Locations**:
- `lib/services/image_cache_service.dart::saveOutputImage()` - timestamped output filenames
- `lib/providers/image_provider.dart::processImage()` - deletes the previous output after saving the new one

**Manual Workarounds (if you suspect caching)**:
- Force quit + relaunch the app (clears memory caches)
- Delete the image entry and re-import it (new ID/new output path)
- Clear app storage / Documents cache (nuclear option)

---

### Issue 7: Flutter UI Jank / Freeze During Inference (SOLVED)

**Symptom**:
While inference is running (especially long full-res runs), the UI becomes unresponsive:
- Animations stop
- Buttons don’t respond
- Progress UI may not repaint until inference finishes

**Root Cause**:
The upstream `flutter_onnxruntime` plugin ran `session.run(...)` on the platform main thread:
- iOS: `FlutterOnnxruntimePlugin.swift::handleRunInference`
- Android: `FlutterOnnxruntimePlugin.kt` `"runInference"` handler

Long ONNX runs on the platform main thread cause visible Flutter jank/freezes.

**Solution Implemented**:
✅ Vendored `flutter_onnxruntime` into the repo and patched it:
- iOS runs inference on a dedicated background queue
- Android runs inference on a dedicated background executor

✅ Wired via `dependency_overrides` so the app uses the vendored plugin.

**Code Locations**:
- `packages/flutter_onnxruntime/ios/Classes/FlutterOnnxruntimePlugin.swift`
- `packages/flutter_onnxruntime/android/src/main/kotlin/com/masicai/flutteronnxruntime/FlutterOnnxruntimePlugin.kt`
- `pubspec.yaml` (`dependency_overrides`)

**How to Verify**:
- Start an enhancement run and try scrolling the list / opening settings while processing.
- UI should remain responsive while inference continues in the background.

---

## Deep Dive: Portrait Image Corruption

### Investigation History

#### Phase 1: Initial Diagnosis (Tensor Format Suspected)

**Hypothesis**: ONNX output might be in NHWC (channels interleaved) instead of NCHW (planar).

**Test**: Added format detection heuristic
```dart
bool _detectNHWC(Float32List data, int width, int height) {
  // Sample pixels and check if RGB channels are more correlated
  // in NHWC vs NCHW interpretation
}
```

**Result**: ❌ Unreliable - detected different formats for different tiles
- Some 512x512 tiles detected as NCHW
- Some 374x512 tiles detected as NHWC
- Bottom row tiles detected as NHWC
- Inconsistent results suggest heuristic was flawed

**Conclusion**: Format detection approach abandoned.

---

#### Phase 2: Tile Size Pattern Discovery

**Observation**: Output value ranges showed a clear pattern:

| Tile Size | Padded Size | Output Range | Status |
|-----------|-------------|--------------|--------|
| 512x512 | 512x512 (no padding) | [-1191, 1923] | ❌ BROKEN |
| 374x512 | 376x512 (2px width pad) | [0.30, 1.03] | ✅ WORKING |
| 512x126 | 512x128 (2px height pad) | [0.34, 1.08] | ✅ WORKING |

**Pattern**: Tiles that **don't need padding** produce garbage output with extreme values.
Tiles that **do need padding** produce correct output with normal value ranges.

**Root Cause Identified (at the time)**:
The original working theory was tile-size/padding dependent behavior (cropping/output extraction differences). We tried forcing a consistent preprocessing path to stabilize tensor ranges.

**Solution Attempted (historical)**:
✅ Always create a fresh `img.Image` buffer in preprocessing:
```dart
Float32List _preprocessImage(img.Image image, int paddedWidth, int paddedHeight) {
  // Always create a fresh image buffer to ensure consistent memory layout
  final paddedImage = img.Image(width: paddedWidth, height: paddedHeight);

  // Copy pixels (even if no padding needed)
  for (int y = 0; y < paddedHeight; y++) {
    for (int x = 0; x < paddedWidth; x++) {
      if (x < image.width && y < image.height) {
        paddedImage.setPixel(x, y, image.getPixel(x, y));
      } else {
        // Reflection padding for edges
        ...
      }
    }
  }
  ...
}
```

**Result (then)**: ✅ Output values moved into reasonable ranges
**BUT**: ❌ Image still looked corrupted

**Important**:
This approach creates a default `Format.uint8` buffer. After discovering some iOS photos decode as `Format.uint16`/HDR, we realized copying via `setPixel(...)` can truncate/wrap high-bit-depth channel values. Current preprocessing does **not** copy into a uint8 buffer.

---

#### Phase 3: High Bit Depth Input Discovered (Key Breakthrough)

**Device Log Evidence**:
```
Sample input pixel (0,0): R=43423, G=39967, B=35455
Input tensor sample [0,1,2]: 0.623529..., 0.121568..., 0.623529...
```

**Interpretation**:
- The decoded image is not uint8 (values are >255).
- The tensor values looked “reasonable” only because truncation/wrap produced 0–255 values that still normalize to 0–1.
- In other words: shapes and ranges can look correct while the underlying colors/values are wrong.

---

#### Phase 4: Current Fix (Normalized Channels)

**Fix**:
Preprocessing now fills the tensor directly from the decoded image using `pixel.rNormalized/gNormalized/bNormalized`, so uint8/uint16/HDR all map correctly to 0–1.

**Expected Result**:
The model should now receive correct inputs for iOS HDR/high-bit-depth photos, eliminating the green/inverted/scrambled look caused by wrapped channels.

---

## Technical Details

### Tensor Format: NCHW (Channels First)

NAFNet and most PyTorch/ONNX models use **NCHW** format:
- **N**: Batch size (always 1 for us)
- **C**: Channels (3 for RGB)
- **H**: Height (image rows)
- **W**: Width (image columns)

**Memory Layout** (C-contiguous / row-major):
```
For tensor shape [1, 3, 512, 512]:
- Indices 0-262143:       Channel 0 (R), all pixels row-by-row
- Indices 262144-524287:  Channel 1 (G), all pixels row-by-row
- Indices 524288-786431:  Channel 2 (B), all pixels row-by-row

Within each channel:
- First 512 values: Row 0, columns 0-511
- Next 512 values:  Row 1, columns 0-511
- ...

Pixel at (x, y) in channel c:
index = c * (H * W) + y * W + x
```

### Our Implementation

**Preprocessing** (Image → Tensor):
```dart
final planeSize = paddedHeight * paddedWidth;
final tensorData = Float32List(1 * 3 * planeSize);

int reflect(int i, int size) =>
    i < size ? i : (2 * size - i - 2).clamp(0, size - 1);

for (int y = 0; y < paddedHeight; y++) {
  final srcY = reflect(y, image.height);
  for (int x = 0; x < paddedWidth; x++) {
    final srcX = reflect(x, image.width);
    final pixel = image.getPixel(srcX, srcY);
    final idx = y * paddedWidth + x;  // Row-major index within each channel plane

    tensorData[0 * planeSize + idx] = pixel.rNormalized.toDouble();  // R
    tensorData[1 * planeSize + idx] = pixel.gNormalized.toDouble();  // G
    tensorData[2 * planeSize + idx] = pixel.bNormalized.toDouble();  // B
  }
}
```

**Postprocessing** (Tensor → Image):
```dart
final planeSize = paddedHeight * paddedWidth;

for (int y = 0; y < height; y++) {
  for (int x = 0; x < width; x++) {
    final idx = y * paddedWidth + x;  // Row-major index

    final r = (tensorData[0 * planeSize + idx].clamp(0.0, 1.0) * 255).round();
    final g = (tensorData[1 * planeSize + idx].clamp(0.0, 1.0) * 255).round();
    final b = (tensorData[2 * planeSize + idx].clamp(0.0, 1.0) * 255).round();

    outputImage.setPixelRgb(x, y, r, g, b);
  }
}
```

**Indexing**:
- `idx = y * paddedWidth + x` - Row-major within a channel plane
- Assumes W (width) varies fastest, H (height) varies next
- This is standard for NCHW row-major layout

### Potential Issues Still Being Investigated

1. **Column-Major vs Row-Major**:
   - If ONNX expects column-major (Fortran order), our row-major indexing would be wrong
   - Test: Try `idx = x * paddedHeight + y` instead of `y * paddedWidth + x`

2. **Channel Order**:
   - Some models use BGR instead of RGB
   - Test: Swap R and B channels in pre/postprocessing

3. **Image Package Format**:
   - `image` 4.x changed APIs significantly
   - `Pixel.r/g/b` might return different value ranges for different formats
   - May need explicit format specification: `img.Image(format: Format.uint8, numChannels: 3)`

4. **Coordinate System**:
   - Verify (x, y) conventions match between `image` package and tensor indexing
   - Image package: x=column (left to right), y=row (top to bottom)
   - Tensor: W=columns, H=rows

5. **Tile Blending**:
   - Even if individual tiles are correct, blending might introduce artifacts
   - Weighted blending with cosine falloff at edges

---

## Configuration Files & Key Code Locations

### Core Inference Logic
- `lib/services/onnx_inference_service.dart`
  - `processImage()` - Main entry point
  - `_processWithTiles()` - Tiled inference implementation
  - `_processSingleTile()` - Process one tile through model
  - `_preprocessImage()` - Image → Tensor conversion
  - `_postprocessOutput()` - Tensor → Image conversion
  - `_runInference()` - ONNX model execution
  - `_flattenList()` - Nested list flattening
  - `_smoothBlend()` - Tile edge blending

### State Management
- `lib/providers/settings_provider.dart`
  - Runtime configuration (model, resolution, tiling, blending)
  - Preset configurations (YOLO, 4K, 2K, 1080p, etc.)
  - `getConfigSummary()` - Human-readable config string

- `lib/providers/image_provider.dart`
  - Image processing orchestration
  - Connects settings to inference service
  - Database persistence

### UI
- `lib/screens/settings_screen.dart`
  - Full settings UI with all configurable parameters
  - Quick presets, per-mode settings, debug toggles

- `lib/widgets/cards/image_card.dart`
  - Displays processed images with config summary
  - Shows what settings were used for each image

### Data Models
- `lib/models/enhanced_image.dart`
  - `ProcessingMode` enum (hd1080p, fullScale)
  - `ProcessingStatus` enum (pending, processing, completed, failed)
  - `EnhancedImage` class with config tracking fields:
    - `modelType` - fp32 or fp16
    - `maxDimension` - resolution cap used
    - `tilingEnabled` - whether tiling was used
    - `tileSize` - tile size if tiling
    - `blendMode` - blending mode if tiling

### Configuration
- `lib/core/config/inference_config.dart`
  - Compile-time default configuration
  - Used as fallback if no runtime settings

---

## Testing Configurations

### Known Working (Landscape/Square Images)

**Landscape Images (width > height)**:
- ✅ Without tiling (< 1MP)
- ✅ With tiling (any size)
- ✅ FP32 and FP16 models
- ✅ All blend modes

**Settings Example**:
```dart
ModelType: fp16
Max Dimension: null (unlimited) or 1024
Tiling: true
Tile Size: 512
Tile Overlap: 96
Blend Mode: cosine
```

### Previously Broken (High Bit Depth iOS Photos)

**iOS Photos Decoded as `Format.uint16` / HDR (`maxChannelValue > 255`)**:
- Previously produced corrupted output due to uint8 truncation during preprocessing
- Expected to be fixed by normalized-channel preprocessing (retest on device)

**How to Detect**:
- Check logs for `Input image format: ... max=...`
- If `max=65535` (or other >255), the decoded image is not uint8

---

## Diagnostic Approach

### Step 1: Isolate Tiling vs Core Inference

**Test A**: Process portrait image **without tiling**
```dart
Settings:
- Mode: 1080p
- Max Dimension: 768 (to avoid memory crash)
- Tiling: OFF
```

**Outcomes**:
- If works ✅: Problem is in tiling/blending code
- If fails ❌: Problem is in core tensor handling

### Step 2: Verify Landscape Still Works

**Test B**: Process landscape image with current code
```dart
Settings:
- Mode: Full Scale
- Tiling: ON
- Same settings as portrait test
```

**Outcomes**:
- If works ✅: Problem is portrait-specific
- If fails ❌: Recent changes broke everything

### Step 3: Debug Output Analysis

**Added Debug Logging**:
```dart
// In _processSingleTile():
final samplePixel = image.getPixel(0, 0);
debugPrint('Input image format: ${image.format}, channels=${image.numChannels}, max=${image.maxChannelValue}');
debugPrint('Sample input pixel (0,0): R=${samplePixel.r}, G=${samplePixel.g}, B=${samplePixel.b} '
  '(normalized: R=${samplePixel.rNormalized}, G=${samplePixel.gNormalized}, B=${samplePixel.bNormalized})');

debugPrint('Input tensor sample [0,1,2]: ${inputData[0]}, ${inputData[1]}, ${inputData[2]}');
final planeSize = paddedHeight * paddedWidth;
debugPrint('Input tensor pixel (0,0): R=${inputData[0]}, G=${inputData[planeSize]}, B=${inputData[2 * planeSize]}');

final outPixel = result.getPixel(0, 0);
debugPrint('Sample output pixel (0,0): R=${outPixel.r}, G=${outPixel.g}, B=${outPixel.b}');
```

**What to Look For**:
1. **Input pixel values**:
   - Raw values may be 0–255 (uint8) or 0–65535 (uint16/HDR)
   - Normalized values should always be ~0.0–1.0
2. **Tensor values**: Should be 0.0-1.0 range (normalized)
3. **Output pixel values**: Should be 0-255 range (uint8)

**Corruption Diagnosis**:
- If input pixels are wrong → Image loading/cropping issue
- If tensor values are wrong → Preprocessing (image→tensor) issue
- If output pixels are wrong → Postprocessing (tensor→image) or ONNX issue

---

## Model Information

### Files
- **FP32 Model**: `assets/models/nafnet_realestate.onnx` (111.5 MB)
- **FP16 Model**: `assets/models/nafnet_realestate_fp16.onnx` (55.9 MB)
- **BEST Model (External Data)**:
  - `assets/models/best_model.onnx` (~77 KB)
  - `assets/models/best_model.onnx.data` (~48 MB)

**Note (External Data Models)**:
Some ONNX exports store weights in a separate `.onnx.data` file. On-device, that `.onnx.data` file must be copied next to the `.onnx` file in the app's Documents `models/` directory or session creation will fail.

### Model Characteristics
- **Architecture**: NAFNet (Nonlinear Activation Free Network)
- **Purpose**: Real estate image denoising and enhancement
- **Input**: RGB image, any size (padded to multiple of 8)
- **Output**: Enhanced RGB image, same size as input
- **Format**: ONNX, NCHW layout
- **I/O Names**: Read from `OrtSession.inputNames/outputNames` (not hardcoded)
- **Normalization**: Input pixels normalized to [0, 1], output may exceed [0, 1] (clamp in postprocessing)

### Quantization
FP16 model created with:
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    'nafnet_realestate.onnx',
    'nafnet_realestate_fp16.onnx',
    weight_type=QuantType.QFloat16
)
```

**Benefits**:
- 50% smaller file size (111.5 MB → 55.9 MB)
- Faster inference on mobile
- Slightly lower quality (usually imperceptible)

---

## Memory Constraints

### iOS Memory Limits
- **Total Device RAM**: 3-8 GB (varies by iPhone model)
- **App Memory Limit**: ~1.2 GB (before iOS kills app)
- **Safe Working Memory**: ~800 MB (with safety margin)

### Tensor Memory Usage

**Formula**: `bytes = batch * channels * height * width * bytes_per_value * 2`
- `* 2` because we need both input and output tensors in memory

**Examples**:
| Image Size | FP32 Input | FP32 Output | Total | Safe? |
|------------|------------|-------------|-------|-------|
| 512x512    | 3.1 MB     | 3.1 MB      | 6.2 MB | ✅ Yes |
| 1024x1024  | 12.6 MB    | 12.6 MB     | 25.2 MB | ✅ Yes |
| 1206x2622  | 38.1 MB    | 38.1 MB     | 76.2 MB | ⚠️ Borderline |
| 2048x2048  | 50.3 MB    | 50.3 MB     | 100.6 MB | ⚠️ Borderline |
| 4032x3024  | 147 MB     | 147 MB      | 294 MB | ❌ Too large |

**Recommendation**: Use tiling for images > 1.5 MP (megapixels).

---

## Tiling Strategy

### Tile Size Selection

**Trade-offs**:
- **Larger tiles**: Better quality, more context, fewer seams, more memory
- **Smaller tiles**: Less memory, more tiles, more seams, less context

**Recommendations**:
- **128-256**: Very memory constrained, many artifacts
- **384-512**: Good balance ✅ (recommended)
- **768-1024**: Best quality, high memory usage

### Overlap Selection

**Purpose**: Smooth blending between tiles to avoid visible seams.

**Trade-offs**:
- **More overlap**: Smoother blending, slower (more redundant processing)
- **Less overlap**: Faster, more visible seams

**Recommendations**:
- **16-32**: Minimal, faster, may show seams
- **64-96**: Good balance ✅ (recommended)
- **128+**: Excellent blending, slower

**Formula**: `overlap = tileSize * 0.15` to `0.25` is a good rule of thumb.

### Blend Modes

**None**: Hard edges, visible seams
```dart
weight = 1.0  // No falloff
```

**Linear**: Simple linear interpolation
```dart
weight = t  // where t is distance ratio
```

**Cosine**: Smooth S-curve (recommended)
```dart
weight = 0.5 - 0.5 * cos(t * π)
```

**Gaussian**: Smoothstep (very smooth)
```dart
weight = t * t * (3.0 - 2.0 * t)
```

---

## Database Schema

### Version 1 (Initial)
```sql
CREATE TABLE enhanced_images (
  id TEXT PRIMARY KEY,
  inputPath TEXT NOT NULL,
  outputPath TEXT,
  inputWidth INTEGER NOT NULL,
  inputHeight INTEGER NOT NULL,
  outputWidth INTEGER,
  outputHeight INTEGER,
  mode INTEGER NOT NULL,
  status INTEGER NOT NULL,
  createdAt INTEGER NOT NULL,
  completedAt INTEGER,
  processingDurationMs INTEGER,
  errorMessage TEXT
)
```

### Version 2 (Config Tracking Added)
```sql
ALTER TABLE enhanced_images ADD COLUMN modelType TEXT;
ALTER TABLE enhanced_images ADD COLUMN maxDimension INTEGER;
ALTER TABLE enhanced_images ADD COLUMN tilingEnabled INTEGER;
ALTER TABLE enhanced_images ADD COLUMN tileSize INTEGER;
ALTER TABLE enhanced_images ADD COLUMN blendMode TEXT;
```

**Migration**: Handled automatically by `sqflite` via `onUpgrade` callback.

---

## Next Steps for Debugging

### Immediate Actions

1. **Run diagnostic tests**:
   ```
   Test 1: Portrait without tiling (max 768px)
   Test 2: Landscape with tiling (verify still works)
   Test 3: Collect debug output with pixel value logging
   ```

2. **Analyze debug output**:
   - Check if pixel values are in expected ranges
   - Verify tensor data is normalized correctly
   - Confirm output pixels are reasonable

3. **Test alternative indexing**:
   - Try column-major: `idx = x * height + y`
   - Try channel swap: BGR instead of RGB
   - Try different image formats

### Longer-Term Solutions

1. **Save intermediate images**:
   - Export preprocessed tensor as image (verify preprocessing)
   - Export raw ONNX output as image (verify model output)
   - Export individual tiles before blending (verify tiling)

2. **Test with simple patterns**:
   - Create synthetic test images (gradients, checkerboards)
   - See if patterns reveal systematic issues

3. **Compare with Python implementation**:
   - Run same image through NAFNet in Python
   - Compare intermediate tensor values
   - Verify Flutter implementation matches

4. **Profile memory usage**:
   - Use Xcode Instruments to track actual memory consumption
   - Identify if there are memory spikes causing corruption

5. **Consider alternative approaches**:
   - Try different ONNX Runtime settings
   - Test with TensorFlow Lite instead of ONNX
   - Investigate CoreML conversion for native iOS inference

---

## Open Questions

1. **Why do portrait images fail but landscape works?**
   - Aspect ratio specific?
   - Dimension ordering issue?
   - Model training bias?

2. **Why did tiles without padding produce extreme values before the fix?**
   - What exactly is different about `img.copyCrop()` memory layout?
   - Is this a bug in the `image` package?

3. **Is the model itself causing issues?**
   - Was NAFNet trained on landscape images only?
   - Does it have built-in assumptions about aspect ratios?
   - Should we test with a different model?

4. **What is `flutter_onnxruntime` doing internally?**
   - How does `OrtValue.fromList()` interpret the data?
   - Is there documentation on expected data layout?
   - Are there examples of working NCHW tensors in Flutter?

---

## Useful Commands

### Build & Run
```bash
flutter clean
flutter pub get
flutter run -d seb
```

### Hot Reload During Development
```bash
r    # Hot reload (fast)
R    # Hot restart (full reload)
```

### Check for Issues
```bash
flutter pub outdated
flutter analyze
```

### Database Reset (If Schema Changes)
```bash
# Uninstall app from device to clear database
# Then rebuild and reinstall
```

---

## Contact Points for Help

### flutter_onnxruntime Package
- **GitHub**: https://github.com/gtbluesky/flutter_onnxruntime
- **Issues**: Check for similar data layout problems
- **Examples**: Look for NCHW tensor examples

### Dart image Package
- **GitHub**: https://github.com/brendan-duncan/image
- **Docs**: API documentation for v4.x
- **Migration Guide**: v3 → v4 breaking changes

### ONNX Runtime
- **Docs**: https://onnxruntime.ai/docs/
- **Tensor Formats**: https://onnxruntime.ai/docs/get-started/with-python.html#data-types

---

## Document Version

**Last Updated**: 2025-12-14
**Status**: Fix implemented for uint16/HDR preprocessing; needs on-device re-test
**Next Review**: After re-testing the same images and confirming tensor normalization matches `pixel.*Normalized`

---

## Summary

**What Works**:
- ✅ Model loading (FP32 and FP16)
- ✅ Model hot-swapping at runtime
- ✅ Runtime settings UI and state management
- ✅ Tiled inference for memory management
- ✅ Landscape/square image processing
- ✅ Database persistence with config tracking
- ✅ Preprocessing creates correct tensor shapes and sizes
- ✅ Tensor values are in reasonable ranges

**What's Broken**:
- ❓ Some iOS photos previously produced corrupted output (uint16/HDR decode) — fix implemented, re-test needed
- ❓ Tiling seams/artifacts may still require overlap/blend tuning (separate from core correctness)

**Current Theory**:
The primary corruption came from a bit-depth mismatch:
- Some iOS photos decode as `Format.uint16`/HDR (`maxChannelValue > 255`)
- Copying those pixels into a default uint8 buffer truncated/wrapped channels
- The model then received incorrect normalized inputs (despite shapes/ranges looking plausible)

**Blocking Issue**: Needs confirmation that portrait/HDR iOS photos now process cleanly.

**Priority**: HIGH - This is a critical feature for mobile real estate photography.
