# ppu-uniface

TypeScript, type-safe, opinionated port of Python's Uniface: A comprehensive library for face detection, recognition, ~~landmark analysis, age, and gender detection~~.

The code pattern is highly inspired by Uniface, however we do not offer model variations. We stick to predetermined opinionated models and functionality to achieve a minimum footprint.

## Features

1. **Face Detection**: Using RetinaNet (like Uniface) with basic 5 landmark points
2. **Face Recognition**: Using FaceNet512 (unlike Uniface, we opted for this, ported from Deepface)
3. **Face Verification**: Using Cosine similarity (unlike Deepface which offers euclidean, euclideanL2, and angular)
4. **Face Alignment**: Automatic face alignment based on eye landmarks
5. **Anti-Spoofing Face**: (WIP) Following Deepface implementation

Customization will be added as we go along. Feel free to open an issue for feature requests.

## Installation

```bash
bun add ppu-uniface
```

or

```bash
npm install ppu-uniface
```

## Usage

### Basic Face Verification

```typescript
import { Uniface } from "ppu-uniface";

const uniface = new Uniface();
await uniface.initialize();

const image1 = await Bun.file("path/to/image1.jpg").arrayBuffer();
const image2 = await Bun.file("path/to/image2.jpg").arrayBuffer();

const result = await uniface.verify(image1, image2);
console.log(result);

await uniface.destroy();
```

### Compact Result (Default)

```typescript
const result = await uniface.verify(image1, image2, { compact: true });
// Returns: { multipleFaces, spoofing, verified, similarity }
```

### Full Result

```typescript
const result = await uniface.verify(image1, image2, { compact: false });
// Returns: { detection, recognition, verification }
```

### Face Detection Only

```typescript
const detection = await uniface.detect(imageBuffer);
console.log(detection);
// Returns: { box, confidence, landmarks, multipleFaces, spoofing }
```

### Face Recognition Only

```typescript
const recognition = await uniface.recognize(imageBuffer);
console.log(recognition.embedding);
// Returns: { embedding: Float32Array(512) }
```

### Direct Embedding Comparison

```typescript
const face1 = await uniface.recognize(image1);
const face2 = await uniface.recognize(image2);

const verification = await uniface.verifyEmbedding(
  face1.embedding,
  face2.embedding
);
console.log(verification);
// Returns: { similarity, verified, threshold }
```

### Enable Verbose Logging

```typescript
import { LoggerConfig } from "ppu-uniface";

LoggerConfig.verbose = true;
```

### Advanced Usage with Individual Components

```typescript
import {
  RetinaNetDetection,
  FaceNet512Recognition,
  CosineVerification,
  alignAndCropFace,
} from "ppu-uniface";

const detector = new RetinaNetDetection();
await detector.initialize();

const recognizer = new FaceNet512Recognition();
await recognizer.initialize();

const verifier = new CosineVerification();

const detection = await detector.detect(imageBuffer);
if (detection) {
  const alignedFace = await alignAndCropFace(imageBuffer, detection);
  const embedding = await recognizer.recognize(alignedFace);
  console.log(embedding);
}

await detector.destroy();
await recognizer.destroy();
```

## API Reference

### `Uniface`

Main service class for face detection, recognition, and verification.

#### Methods

- `initialize(): Promise<void>` - Initializes all models
- `detect(image: ArrayBuffer | Canvas): Promise<DetectionResult | null>` - Detects face in image
- `recognize(image: ArrayBuffer | Canvas): Promise<RecognitionResult>` - Generates face embedding
- `verify(image1, image2, options?): Promise<UnifaceFullResult | UnifaceCompactResult>` - Verifies if two images contain the same person
- `verifyEmbedding(embedding1, embedding2): Promise<VerificationResult>` - Compares two embeddings directly
- `destroy(): Promise<void>` - Releases all model resources

### Types

#### `DetectionResult`

```typescript
{
  box: { x: number; y: number; width: number; height: number };
  confidence: number;
  landmarks: number[][]; // 5 points: left eye, right eye, nose, left mouth, right mouth
  multipleFaces: boolean;
  spoofing: boolean;
}
```

#### `RecognitionResult`

```typescript
{
  embedding: Float32Array; // 512-dimensional vector
}
```

#### `VerificationResult`

```typescript
{
  similarity: number; // 0-1
  verified: boolean;
  threshold: number; // Default: 0.7
}
```

#### `UnifaceCompactResult`

```typescript
{
  multipleFaces: { face1: boolean | null; face2: boolean | null };
  spoofing: { face1: boolean | null; face2: boolean | null };
  verified: boolean;
  similarity: number;
}
```

#### `UnifaceFullResult`

```typescript
{
  detection: { face1: DetectionResult | null; face2: DetectionResult | null };
  recognition: { face1: RecognitionResult; face2: RecognitionResult };
  verification: VerificationResult;
}
```

## Models

The library automatically downloads and caches the following models on first use:

- **RetinaFace MobileNet V2** (~1.7MB) - Face detection
- **FaceNet512** (~23MB) - Face recognition

Models are cached in `~/.cache/ppu-uniface/` for faster subsequent loads.

## Benchmark

```sh
bun run benchmark
```

Feel free to compare it to Deepface.

## Testing

```sh
bun test
```

The library includes comprehensive unit tests covering:
- Face detection with RetinaNet
- Face recognition with FaceNet512
- Cosine similarity verification
- Full integration tests

## Performance

Typical performance on modern hardware:
- **Detection**: ~50-150ms per image
- **Recognition**: ~30-80ms per aligned face
- **Full Verification**: ~100-300ms for two images

## Requirements

- **Runtime**: Bun or Node.js 18+
- **Dependencies**: 
  - `onnxruntime-node` - ONNX model inference
  - `ppu-ocv` - Computer vision utilities

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT

## Credits

- [Uniface](https://github.com/yakhyo/uniface) - Original Python implementation
- [Deepface](https://github.com/serengil/deepface) - Face recognition framework
- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis toolkit

## Roadmap

- [ ] Anti-spoofing detection
- [ ] Age and gender detection
- [ ] Additional verification metrics (Euclidean, Euclidean L2)
- [ ] Model customization options
- [ ] Browser support (ONNX Web)
- [ ] Performance optimizations
