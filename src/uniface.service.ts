import { Canvas, ImageProcessor } from "ppu-ocv";
import { alignAndCropFace } from "./alignment.face";
import {
  SpoofingDetection,
  type SpoofingOptions,
  type SpoofingResult,
} from "./analysis/spoofing.ana";
import type {
  BaseDetection,
  DetectionModelOptions,
  DetectionResult,
} from "./detection/base.interface";
import { RetinaNetDetection } from "./detection/retinanet.det";
import { logger } from "./logger";
import type {
  BaseRecognition,
  RecognitionModelOptions,
  RecognitionResult,
} from "./recognition/base.interface";
import { FaceNet512Recognition } from "./recognition/facenet512.rec";
import type {
  UnifaceCompactResult,
  UnifaceFullResult,
  UnifaceVerifyOptions,
} from "./uniface.interface";
import type {
  BaseVerification,
  VerificationModelOptions,
  VerificationResult,
} from "./verification/base.interface";
import { CosineVerification } from "./verification/cosine.ver";

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
   * @returns Detection result or null if no face found
   */
  async detect(image: ArrayBuffer | Canvas): Promise<DetectionResult | null> {
    const result = await this.detection.detect(image);
    return result;
  }

  /**
   * Generates face embedding from image
   * @param image - Input image as ArrayBuffer or Canvas
   * @returns Recognition result with embedding vector
   */
  async recognize(image: ArrayBuffer | Canvas): Promise<RecognitionResult> {
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
    image1: ArrayBuffer | Canvas,
    image2: ArrayBuffer | Canvas,
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
    image1: ArrayBuffer | Canvas,
    image2: ArrayBuffer | Canvas,
    options: UnifaceVerifyOptions & { compact: false }
  ): Promise<UnifaceFullResult>;

  /**
   * Verifies if two images contain the same person (defaults to compact)
   */
  async verify(
    image1: ArrayBuffer | Canvas,
    image2: ArrayBuffer | Canvas,
    options?: UnifaceVerifyOptions
  ): Promise<UnifaceCompactResult>;

  // Implementation
  async verify(
    image1: ArrayBuffer | Canvas,
    image2: ArrayBuffer | Canvas,
    options: UnifaceVerifyOptions = { compact: true }
  ): Promise<UnifaceFullResult | UnifaceCompactResult> {
    const [result1, result2] = await Promise.all([
      this.processImage(image1),
      this.processImage(image2),
    ]);

    const verification = await this.verifyEmbedding(
      result1.recognition.embedding,
      result2.recognition.embedding
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
   * Processes an image through detection and recognition pipeline
   * @param imageBuffer - Input image
   * @returns Detection and recognition results
   */
  private async processImage(imageBuffer: ArrayBuffer | Canvas): Promise<{
    detection: DetectionResult | null;
    recognition: RecognitionResult;
    spoofing: SpoofingResult | null;
  }> {
    const detection = await this.detect(imageBuffer);
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
   * Compares two face embeddings directly
   * @param embedding1 - First face embedding
   * @param embedding2 - Second face embedding
   * @returns Verification result with similarity score
   */
  async verifyEmbedding(
    embedding1: Float32Array,
    embedding2: Float32Array
  ): Promise<VerificationResult> {
    const result = this.verification.compare(embedding1, embedding2);
    return result;
  }

  /**
   * Analyzes if a cropped facial image is spoofed
   * @param image - Cropped facial area image as ArrayBuffer or Canvas
   * @returns Spoofing analysis result
   */
  async spoofingAnalysis(image: ArrayBuffer | Canvas): Promise<SpoofingResult> {
    const result = await this.spoofing.analyze(image);
    return result;
  }

  /**
   * Analyzes if an image contains a spoofed face (with automatic detection and cropping)
   * @param image - Full image as ArrayBuffer or Canvas
   * @returns Spoofing analysis result or null if no face detected
   */
  async spoofingAnalysisWithDetection(
    image: ArrayBuffer | Canvas
  ): Promise<SpoofingResult | null> {
    const detection = await this.detect(image);

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
