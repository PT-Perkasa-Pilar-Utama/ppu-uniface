import type { InferenceSession } from "onnxruntime-common";
import type { CoreCanvas, PlatformProvider } from "../core/platform.js";
import { Base } from "../global.interface.js";

/**
 * Configuration options for face detection models
 */
export interface DetectionModelOptions {
  /** Threshold values for detection */
  threshold: {
    /** Minimum confidence score for face detection */
    confidence: number;
    /** IoU threshold for non-maximum suppression */
    nonMaximumSuppression: number;
  };
  /** Top-K filtering options */
  topK: {
    /** Maximum detections before NMS */
    preNonMaximumSuppression: number;
    /** Maximum detections after NMS */
    postNonMaxiumSuppression: number;
  };
  /** Model input size */
  size: {
    /** Input dimensions [height, width] */
    input: [number, number];
  };
}

/**
 * Options for detect method
 */
export interface DetectOptions {
  /** Threshold values for detection */
  threshold?: {
    /** Minimum confidence score for face detection (default: 0.7) */
    confidence?: number;
    /** IoU threshold for non-maximum suppression (default: 0.4) */
    nonMaximumSuppression?: number;
  };
}

/**
 * Face detection result
 */
export interface DetectionResult {
  /** Bounding box coordinates and dimensions */
  box: { x: number; y: number; width: number; height: number };
  /** Confidence score of the detection */
  confidence: number;
  /** Facial landmarks (5 points: left eye, right eye, nose, left mouth, right mouth) */
  landmarks: number[][];
  /** Whether multiple faces were detected in the image */
  multipleFaces: boolean;
}

/**
 * Base class for face detection models
 */
export abstract class BaseDetection extends Base {
  /** Class name for logging */
  protected abstract override className: string;
  /** Path to the model file */
  protected abstract modelPath: string;
  /** Detection configuration options */
  protected abstract detectionOptions: DetectionModelOptions;
  /** ONNX inference session */
  protected abstract session: InferenceSession | null;

  constructor(platform?: PlatformProvider) {
    super(platform);
  }

  /** Initializes the detection model */
  abstract initialize(): Promise<void>;

  /**
   * Detects faces in an image
   * @param image - Input image as ArrayBuffer or Canvas
   * @param options - Optional detection options
   * @returns Detection result or null if no face detected
   */
  abstract detect(
    image: ArrayBuffer | CoreCanvas,
    options?: DetectOptions,
  ): Promise<DetectionResult | null>;

  /**
   * Preprocesses image for model inference
   * @param image - Input canvas
   * @param height - Image height
   * @param width - Image width
   * @returns Preprocessed tensor
   */
  abstract preprocess(
    image: CoreCanvas,
    height: number,
    width: number,
  ): Float32Array;

  /**
   * Runs model inference
   * @param tensor - Preprocessed input tensor
   * @param shape - Input shape [height, width]
   * @returns Model outputs
   */
  abstract inference(
    tensor: Float32Array,
    shape: [number, number],
  ): Promise<InferenceSession.OnnxValueMapType>;

  /**
   * Postprocesses model outputs
   * @param outputs - Raw model outputs
   * @returns Processed boxes, scores, and landmarks
   */
  abstract postprocess(outputs: InferenceSession.OnnxValueMapType): {
    boxes: Float32Array;
    scores: Float32Array;
    landmarks: Float32Array;
  };

  /** Releases model resources */
  abstract destroy(): Promise<void>;

  /**
   * Checks if the model has been initialized
   * @returns True if initialized
   */
  protected isInitialized(): boolean {
    return this.session !== null;
  }

  /**
   * Generates anchor boxes for face detection
   * @param imageSize - Image dimensions [height, width]
   * @returns Flat array of anchor boxes [cx, cy, w, h, ...]
   */
  protected generateAnchors(
    imageSize: [number, number] = [640, 640],
  ): Float32Array {
    const steps = [8, 16, 32];
    const minSizes = [
      [16, 32],
      [64, 128],
      [256, 512],
    ];

    const anchors: number[] = [];
    const featureMaps = steps.map((step) => [
      Math.ceil(imageSize[0] / step),
      Math.ceil(imageSize[1] / step),
    ]);

    for (let k = 0; k < featureMaps.length; k++) {
      const [mapHeight, mapWidth] = featureMaps[k]!;
      const step = steps[k]!;

      for (let i = 0; i < mapHeight!; i++) {
        for (let j = 0; j < mapWidth!; j++) {
          for (const minSize of minSizes[k]!) {
            const s_kx = minSize / imageSize[1];
            const s_ky = minSize / imageSize[0];

            const cx = ((j + 0.5) * step) / imageSize[1];
            const cy = ((i + 0.5) * step) / imageSize[0];

            anchors.push(cx, cy, s_kx, s_ky);
          }
        }
      }
    }

    return new Float32Array(anchors);
  }

  /**
   * Decode locations from predictions using priors to undo
   * the encoding done for offset regression at train time.
   */
  protected decodeBoxes(
    loc: Float32Array,
    priors: Float32Array,
    variances: [number, number] = [0.1, 0.2],
  ): Float32Array {
    const numPriors = loc.length / 4;
    const boxes = new Float32Array(loc.length);

    for (let i = 0; i < numPriors; i++) {
      const idx = i * 4;

      const cx = priors[idx]! + loc[idx]! * variances[0] * priors[idx + 2]!;
      const cy =
        priors[idx + 1]! + loc[idx + 1]! * variances[0] * priors[idx + 3]!;

      const w = priors[idx + 2]! * Math.exp(loc[idx + 2]! * variances[1]);
      const h = priors[idx + 3]! * Math.exp(loc[idx + 3]! * variances[1]);

      boxes[idx] = cx - w / 2;
      boxes[idx + 1] = cy - h / 2;
      boxes[idx + 2] = cx + w / 2;
      boxes[idx + 3] = cy + h / 2;
    }

    return boxes;
  }

  /**
   * Decode landmark predictions using prior boxes.
   */
  protected decodeLandmarks(
    predictions: Float32Array,
    priors: Float32Array,
    variances: [number, number] = [0.1, 0.2],
  ): Float32Array {
    const numPriors = predictions.length / 10;
    const landmarks = new Float32Array(predictions.length);

    for (let i = 0; i < numPriors; i++) {
      const priorIdx = i * 4;
      const landmarkIdx = i * 10;

      const priorCx = priors[priorIdx]!;
      const priorCy = priors[priorIdx + 1]!;
      const priorW = priors[priorIdx + 2]!;
      const priorH = priors[priorIdx + 3]!;

      for (let j = 0; j < 5; j++) {
        const idx = landmarkIdx + j * 2;

        landmarks[idx] = priorCx + predictions[idx]! * variances[0] * priorW;
        landmarks[idx + 1] =
          priorCy + predictions[idx + 1]! * variances[0] * priorH;
      }
    }

    return landmarks;
  }

  /**
   * Apply Non-Maximum Suppression (NMS) to reduce overlapping bounding boxes.
   */
  protected nonMaximumSuppression(
    dets: Float32Array,
    numDets: number,
    threshold: number,
  ): number[] {
    if (numDets === 0) return [];

    const x1 = new Float32Array(numDets);
    const y1 = new Float32Array(numDets);

    const x2 = new Float32Array(numDets);
    const y2 = new Float32Array(numDets);

    const scores = new Float32Array(numDets);
    const areas = new Float32Array(numDets);

    for (let i = 0; i < numDets; i++) {
      const idx = i * 5;
      x1[i] = dets[idx]!;
      y1[i] = dets[idx + 1]!;

      x2[i] = dets[idx + 2]!;
      y2[i] = dets[idx + 3]!;

      scores[i] = dets[idx + 4]!;
      areas[i] = (x2[i]! - x1[i]!) * (y2[i]! - y1[i]!);
    }

    const order = Array.from({ length: numDets }, (_, i) => i);
    order.sort((a, b) => scores[b]! - scores[a]!);

    const keep: number[] = [];
    const suppressed = new Set<number>();

    for (const i of order) {
      if (suppressed.has(i)) continue;

      keep.push(i);

      for (const j of order) {
        if (i === j || suppressed.has(j)) continue;

        const xx1 = Math.max(x1[i]!, x1[j]!);
        const yy1 = Math.max(y1[i]!, y1[j]!);
        const xx2 = Math.min(x2[i]!, x2[j]!);
        const yy2 = Math.min(y2[i]!, y2[j]!);

        const w = Math.max(0.0, xx2 - xx1);
        const h = Math.max(0.0, yy2 - yy1);
        const inter = w * h;

        const iou = inter / (areas[i]! + areas[j]! - inter);

        if (iou > threshold) {
          suppressed.add(j);
        }
      }
    }

    return keep;
  }
}
