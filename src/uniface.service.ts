import { Canvas, ImageProcessor } from "ppu-ocv";
import { alignAndCropFace } from "./alignment.face";
import type {
  BaseDetection,
  DetectionResult,
} from "./detection/base.interface";
import { RetinaNetDetection } from "./detection/retinanet.det";
import { logger } from "./logger";
import type {
  BaseRecognition,
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
  VerificationResult,
} from "./verification/base.interface";
import { CosineVerification } from "./verification/cosine.ver";

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

  /** Initializes Uniface service with default models */
  constructor() {
    this.log("constructor", "Initializing Uniface service...");

    this.detection = new RetinaNetDetection();
    this.recognition = new FaceNet512Recognition();
    this.verification = new CosineVerification();

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

    this.log("initialize", "All models initialized successfully");
  }

  /**
   * Detects face in an image
   * @param image - Input image as ArrayBuffer or Canvas
   * @returns Detection result or null if no face found
   */
  async detect(image: ArrayBuffer | Canvas): Promise<DetectionResult | null> {
    this.log("detect", "Starting face detection...");
    const result = await this.detection.detect(image);

    this.log(
      "detect",
      `Detection completed: ${result ? "Face found" : "No face detected"}`
    );
    return result;
  }

  /**
   * Generates face embedding from image
   * @param image - Input image as ArrayBuffer or Canvas
   * @returns Recognition result with embedding vector
   */
  async recognize(image: ArrayBuffer | Canvas): Promise<RecognitionResult> {
    this.log("recognize", "Starting face recognition...");
    const result = await this.recognition.recognize(image);

    this.log(
      "recognize",
      `Recognition completed: embedding size ${result.embedding.length}`
    );
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
    this.log("verify", "Starting verification process...");
    const [result1, result2] = await Promise.all([
      this.processImage(image1),
      this.processImage(image2),
    ]);

    const verification = await this.verifyEmbedding(
      result1.recognition.embedding,
      result2.recognition.embedding
    );
    this.log(
      "verify",
      `Verification result: ${verification.verified ? "VERIFIED" : "NOT VERIFIED"}`
    );

    if (options.compact) {
      return {
        multipleFaces: {
          face1: result1.detection?.multipleFaces ?? null,
          face2: result2.detection?.multipleFaces ?? null,
        },
        spoofing: {
          face1: result1.detection?.spoofing ?? null,
          face2: result2.detection?.spoofing ?? null,
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
  }> {
    this.log("processImage", "Starting image processing...");
    const detection = await this.detect(imageBuffer);
    let recognition: RecognitionResult = { embedding: new Float32Array(0) };

    if (detection != null) {
      const alignedCanvas = await alignAndCropFace(imageBuffer, detection);
      recognition = await this.recognize(alignedCanvas);
    }

    this.log("processImage", "Image processing completed");
    return { detection, recognition };
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
    this.log(
      "verifyEmbedding",
      `Comparing embeddings (size: ${embedding1.length} vs ${embedding2.length})`
    );
    const result = this.verification.compare(embedding1, embedding2);
    this.log(
      "verifyEmbedding",
      `Comparison completed: similarity=${result.similarity.toFixed(4)}`
    );
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
