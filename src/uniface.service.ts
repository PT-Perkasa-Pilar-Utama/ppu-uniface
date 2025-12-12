import { Canvas, ImageProcessor } from "ppu-ocv";
import { alignAndCropFace } from "./alignment.face.js";
import {
  SpoofingDetection,
  type SpoofingOptions,
  type SpoofingResult,
} from "./analysis/spoofing.ana.js";
import type {
  BaseDetection,
  DetectOptions,
  DetectionModelOptions,
  DetectionResult,
} from "./detection/base.interface.js";
import { RetinaNetDetection } from "./detection/retinanet.det.js";
import { logger } from "./logger.js";
import type {
  BaseRecognition,
  RecognitionModelOptions,
  RecognitionResult,
} from "./recognition/base.interface.js";
import { FaceNet512Recognition } from "./recognition/facenet512.rec.js";
import type {
  UnifaceCompactResult,
  UnifaceFullResult,
  UnifaceVerifyOptions,
} from "./uniface.interface.js";
import type {
  BaseVerification,
  VerificationModelOptions,
  VerificationResult,
} from "./verification/base.interface.js";
import { CosineVerification } from "./verification/cosine.ver.js";

/**
 * Configuration options for Uniface service
 */
export interface UnifaceOptions {
  /** Options for face detection model */
  detection?: Partial<DetectionModelOptions>;
  /** Options for face recognition model */
  recognition?: Partial<RecognitionModelOptions>;
  /** Options for face verification */
  verification?: Partial<VerificationModelOptions>;
  /** Options for anti spoofing face detection */
  spoofing?: Partial<SpoofingOptions>;
}

/**
 * Processed image result containing detection, recognition, and spoofing data
 */
interface ProcessedImage {
  detection: DetectionResult | null;
  recognition: RecognitionResult;
  spoofing: SpoofingResult | null;
}

/**
 * Input that can be either raw image or pre-detected result
 */
type ImageInput = ArrayBuffer | Canvas;
type DetectedInput = { image: ImageInput; detection: DetectionResult };

/**
 * Main service class for face detection, recognition, and verification
 */
export class Uniface {
  /** Face detection model instance */
  protected detection: BaseDetection;
  /** Face recognition model instance */
  protected recognition: BaseRecognition;
  /** Face verification method instance */
  protected verification: BaseVerification;
  /** Face spoofing detection method instance */
  protected spoofing: SpoofingDetection;

  /** Initializes Uniface service with default models */
  constructor(protected options: UnifaceOptions = {}) {
    this.log("constructor", "Initializing Uniface service...");

    this.detection = new RetinaNetDetection(options.detection);
    this.recognition = new FaceNet512Recognition(options.recognition);
    this.verification = new CosineVerification(options.verification);
    this.spoofing = new SpoofingDetection(options.spoofing);

    this.log(
      "constructor",
      "All Uniface's service model initialized successfully"
    );
  }

  /** Initializes all models (detection, recognition) */
  async initialize(): Promise<void> {
    this.log("initialize", "Starting initialization...");

    await ImageProcessor.initRuntime();
    await this.detection.initialize();
    await this.recognition.initialize();
    await this.spoofing.initialize();

    this.log("initialize", "All models initialized successfully");
  }

  /**
   * Detects face in an image
   * @param image - Input image as ArrayBuffer or Canvas
   * @param options - Optional detection options
   * @returns Detection result or null if no face found
   */
  async detect(
    image: ImageInput,
    options?: DetectOptions
  ): Promise<DetectionResult | null> {
    const result = await this.detection.detect(image, options);
    return result;
  }

  /**
   * Generates face embedding from image
   * @param image - Input image as ArrayBuffer or Canvas
   * @returns Recognition result with embedding vector
   */
  async recognize(image: ImageInput): Promise<RecognitionResult> {
    const result = await this.recognition.recognize(image);
    return result;
  }

  /**
   * Verifies if two images contain the same person
   * @param image1 - First image
   * @param image2 - Second image
   * @param options - Verification options with compact: true
   * @returns Compact verification result
   */
  async verify(
    image1: ImageInput,
    image2: ImageInput,
    options: UnifaceVerifyOptions & { compact: true }
  ): Promise<UnifaceCompactResult>;

  /**
   * Verifies if two images contain the same person
   * @param image1 - First image
   * @param image2 - Second image
   * @param options - Verification options with compact: false
   * @returns Full verification result
   */
  async verify(
    image1: ImageInput,
    image2: ImageInput,
    options: UnifaceVerifyOptions & { compact: false }
  ): Promise<UnifaceFullResult>;

  /**
   * Verifies if two images contain the same person (defaults to compact)
   */
  async verify(
    image1: ImageInput,
    image2: ImageInput,
    options?: UnifaceVerifyOptions
  ): Promise<UnifaceCompactResult>;

  async verify(
    image1: ImageInput,
    image2: ImageInput,
    options: UnifaceVerifyOptions = { compact: true }
  ): Promise<UnifaceFullResult | UnifaceCompactResult> {
    const [result1, result2] = await Promise.all([
      this.processImage(image1, options.detection),
      this.processImage(image2, options.detection),
    ]);

    return this.buildVerificationResult(result1, result2, options);
  }

  /**
   * Verifies with pre-computed detections (optimized - skips detection)
   * @param input1 - First image with detection or raw image
   * @param input2 - Second image with detection or raw image
   * @param options - Verification options with compact: true
   * @returns Compact verification result
   */
  async verifyWithDetections(
    input1: DetectedInput | ImageInput,
    input2: DetectedInput | ImageInput,
    options: UnifaceVerifyOptions & { compact: true }
  ): Promise<UnifaceCompactResult>;

  /**
   * Verifies with pre-computed detections (optimized - skips detection)
   * @param input1 - First image with detection or raw image
   * @param input2 - Second image with detection or raw image
   * @param options - Verification options with compact: false
   * @returns Full verification result
   */
  async verifyWithDetections(
    input1: DetectedInput | ImageInput,
    input2: DetectedInput | ImageInput,
    options: UnifaceVerifyOptions & { compact: false }
  ): Promise<UnifaceFullResult>;

  /**
   * Verifies with pre-computed detections (optimized - skips detection, defaults to compact)
   */
  async verifyWithDetections(
    input1: DetectedInput | ImageInput,
    input2: DetectedInput | ImageInput,
    options?: UnifaceVerifyOptions
  ): Promise<UnifaceCompactResult>;

  async verifyWithDetections(
    input1: DetectedInput | ImageInput,
    input2: DetectedInput | ImageInput,
    options: UnifaceVerifyOptions = { compact: true }
  ): Promise<UnifaceFullResult | UnifaceCompactResult> {
    const [result1, result2] = await Promise.all([
      this.processImageInput(input1, options.detection),
      this.processImageInput(input2, options.detection),
    ]);

    return this.buildVerificationResult(result1, result2, options);
  }

  /**
   * Processes input that can be either raw image or detected input
   */
  private async processImageInput(
    input: DetectedInput | ImageInput,
    detectOptions?: DetectOptions
  ): Promise<ProcessedImage> {
    if (this.isDetectedInput(input)) {
      return this.processImageWithDetection(input.image, input.detection);
    }
    return this.processImage(input, detectOptions);
  }

  /**
   * Type guard for DetectedInput
   */
  private isDetectedInput(
    input: DetectedInput | ImageInput
  ): input is DetectedInput {
    return (
      typeof input === "object" &&
      input !== null &&
      "image" in input &&
      "detection" in input
    );
  }

  /**
   * Processes an image through full pipeline (detection + recognition + spoofing)
   */
  private async processImage(
    imageBuffer: ImageInput,
    detectOptions?: DetectOptions
  ): Promise<ProcessedImage> {
    const detection = await this.detect(imageBuffer, detectOptions);
    return this.processImageWithDetection(imageBuffer, detection);
  }

  /**
   * Processes an image with pre-computed detection (skips detection step)
   */
  private async processImageWithDetection(
    imageBuffer: ImageInput,
    detection: DetectionResult | null
  ): Promise<ProcessedImage> {
    let recognition: RecognitionResult = { embedding: new Float32Array(0) };
    let spoofing: SpoofingResult | null = null;

    if (detection != null) {
      const alignedCanvas = await alignAndCropFace(imageBuffer, detection);
      recognition = await this.recognize(alignedCanvas);

      if (this.options.spoofing?.enable) {
        spoofing = await this.spoofing.analyze(imageBuffer, {
          x: 0,
          y: 0,
          width: alignedCanvas.width,
          height: alignedCanvas.height,
        });
      }
    }

    return { detection, recognition, spoofing };
  }

  /**
   * Builds verification result from processed images
   */
  private async buildVerificationResult(
    result1: ProcessedImage,
    result2: ProcessedImage,
    options: UnifaceVerifyOptions
  ): Promise<UnifaceFullResult | UnifaceCompactResult> {
    const verification = await this.verifyEmbedding(
      result1.recognition.embedding,
      result2.recognition.embedding,
      options.threshold
    );

    if (options.compact) {
      return {
        multipleFaces: {
          face1: result1.detection?.multipleFaces ?? null,
          face2: result2.detection?.multipleFaces ?? null,
        },
        spoofing: {
          face1: result1.spoofing ? !result1.spoofing.real : null,
          face2: result2.spoofing ? !result2.spoofing.real : null,
        },
        verified: verification.verified,
        similarity: verification.similarity,
      };
    }

    return {
      detection: {
        face1: result1.detection,
        face2: result2.detection,
      },
      recognition: {
        face1: result1.recognition,
        face2: result2.recognition,
      },
      spoofing: {
        face1: result1.spoofing,
        face2: result2.spoofing,
      },
      verification,
    };
  }

  /**
   * Compares two face embeddings directly
   * @param embedding1 - First face embedding
   * @param embedding2 - Second face embedding
   * @param threshold - Optional threshold to override model-level default
   * @returns Verification result with similarity score
   */
  async verifyEmbedding(
    embedding1: Float32Array,
    embedding2: Float32Array,
    threshold?: number
  ): Promise<VerificationResult> {
    const result = this.verification.compare(embedding1, embedding2, threshold);
    return result;
  }

  /**
   * Analyzes if a cropped facial image is spoofed
   * @param image - Cropped facial area image as ArrayBuffer or Canvas
   * @returns Spoofing analysis result
   */
  async spoofingAnalysis(image: ImageInput): Promise<SpoofingResult> {
    const result = await this.spoofing.analyze(image);
    return result;
  }

  /**
   * Analyzes if an image contains a spoofed face (with automatic detection and cropping)
   * @param image - Full image as ArrayBuffer or Canvas
   * @param options - Optional detection options
   * @returns Spoofing analysis result or null if no face detected
   */
  async spoofingAnalysisWithDetection(
    image: ImageInput,
    options?: DetectOptions
  ): Promise<SpoofingResult | null> {
    const detection = await this.detect(image, options);

    if (detection == null) {
      return null;
    }

    const result = await this.spoofing.analyze(image, detection.box);
    return result;
  }

  /**
   * Logs a message using the logger utility
   * @param methodName - Name of the method
   * @param message - Log message
   */
  private log(methodName: string, message: string): void {
    logger("Uniface", methodName, message);
  }

  /** Releases all model resources */
  async destroy(): Promise<void> {
    this.log("destroy", "Starting cleanup...");
    await this.detection.destroy();
    await this.recognition.destroy();
    this.log("destroy", "Cleanup completed");
  }
}
