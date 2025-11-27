import * as ort from "onnxruntime-node";
import { Canvas, ImageProcessor } from "ppu-ocv";
import { GITHUB_BASE_URL } from "../constant";
import { BaseAnalysis } from "./base.interface";

/**
 * Configuration options for face spoofing detection models
 */
export interface SpoofingOptions {
  /** Threshold score values for spoofing detection */
  threshold: number;
  /** Wether to activate the anti-spoofing analysis */
  enable: boolean;
}

/**
 * Spoofing detection result
 */
export interface SpoofingResult {
  /** Whether the image is real or fake */
  real: boolean;
  /** The given score from analysis */
  score: number;
}

export class SpoofingDetection extends BaseAnalysis {
  protected override className: string = "SpoofingDetection";

  protected firstModelPath: string = "spoofing/MiniFASNetV2.onnx";
  protected secondModelPath: string = "spoofing/MiniFASNetV1SE.onnx";

  protected firstSession: ort.InferenceSession | null = null;
  protected secondSession: ort.InferenceSession | null = null;

  protected options: SpoofingOptions = {
    threshold: 0.5,
    enable: true,
  };

  constructor(options: Partial<SpoofingOptions> = {}) {
    super();
    this.options = {
      ...this.options,
      ...options,
    };
  }

  async initialize(): Promise<void> {
    this.log("initialize", "Starting MiniFASNet initialization...");
    await ImageProcessor.initRuntime();

    const firstBuffer = await this.loadResource(
      undefined,
      `${GITHUB_BASE_URL}${this.firstModelPath}`
    );

    this.firstSession = await ort.InferenceSession.create(
      new Uint8Array(firstBuffer)
    );

    const secondBuffer = await this.loadResource(
      undefined,
      `${GITHUB_BASE_URL}${this.secondModelPath}`
    );

    this.secondSession = await ort.InferenceSession.create(
      new Uint8Array(secondBuffer)
    );
    this.log("initialize", "MiniFASNet initialized");
  }

  /**
   * Checks if the model has been initialized
   * @returns True if initialized
   */
  protected isInitialized(): boolean {
    return this.firstSession !== null && this.secondSession !== null;
  }

  /**
   * Analyze whether a face is spoofed like a photo of a photo or a display from a phone etc.
   * @param image The facial area of the image
   */
  async analyze(image: ArrayBuffer | Canvas): Promise<SpoofingResult> {
    if (!this.isInitialized)
      throw Error(`${this.className} session was not initialized`);

    const result: SpoofingResult = {
      real: false,
      score: 0,
    };

    return result;
  }

  /**
   * Detect face in an image with additional spoofing detection
   * @param image
   */
  async analyzeWithDetection(
    image: ArrayBuffer | Canvas
  ): Promise<SpoofingResult> {
    if (!this.isInitialized)
      throw Error(`${this.className} session was not initialized`);

    const result: SpoofingResult = {
      real: false,
      score: 0,
    };

    return result;
  }

  async destroy(): Promise<void> {
    await this.firstSession?.release();
    await this.secondSession?.release();

    this.firstSession = null;
    this.secondSession = null;
  }
}
