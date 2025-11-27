import * as ort from "onnxruntime-node";
import { Canvas, createCanvas, ImageProcessor } from "ppu-ocv";
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

    const canvas =
      image instanceof ArrayBuffer
        ? await ImageProcessor.prepareCanvas(image)
        : image;

    // Crop and preprocess for first model (scale 2.7)
    const firstTensor = this.preprocess(canvas, 2.7, 80, 80);

    // Crop and preprocess for second model (scale 4.0)
    const secondTensor = this.preprocess(canvas, 4.0, 80, 80);

    // Run inference on both models
    const firstOutput = await this.inference(
      this.firstSession!,
      firstTensor,
      [1, 3, 80, 80]
    );
    const secondOutput = await this.inference(
      this.secondSession!,
      secondTensor,
      [1, 3, 80, 80]
    );

    // Postprocess and combine results
    const result = this.postprocess(firstOutput, secondOutput);

    this.log(
      "analyze",
      `Spoofing analysis: real=${result.real}, score=${result.score.toFixed(3)}`
    );

    return result;
  }

  /**
   * Preprocess image by cropping with scale and resizing
   * @param canvas Input canvas
   * @param scale Scale factor for cropping
   * @param outW Output width
   * @param outH Output height
   * @returns Preprocessed tensor
   */
  protected preprocess(
    canvas: Canvas,
    scale: number,
    outW: number,
    outH: number
  ): Float32Array {
    const { width: srcW, height: srcH } = canvas;

    // Get the entire image as bounding box
    const bbox = { x: 0, y: 0, width: srcW, height: srcH };

    // Calculate new box with scale
    const newBox = this.getNewBox(srcW, srcH, bbox, scale);

    // Crop the image
    const ctx = canvas.getContext("2d");
    const croppedImageData = ctx.getImageData(
      newBox.x,
      newBox.y,
      newBox.width,
      newBox.height
    );

    // Create a temporary canvas for the cropped image
    const tempCanvas = createCanvas(newBox.width, newBox.height);
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.putImageData(croppedImageData, 0, 0);

    // Resize to target size
    let resizedCanvas = tempCanvas;
    if (newBox.width !== outW || newBox.height !== outH) {
      const processor = new ImageProcessor(tempCanvas);
      resizedCanvas = processor.resize({ width: outW, height: outH }).toCanvas();
      processor.destroy();
    }

    // Convert to tensor in CHW format (channels, height, width)
    const channels = 3;
    const tensor = new Float32Array(channels * outH * outW);

    const resizedCtx = resizedCanvas.getContext("2d");
    const imageData = resizedCtx.getImageData(0, 0, outW, outH).data;

    const totalPixels = outH * outW;
    const stride = outH * outW;

    let ptr = 0;
    for (let i = 0; i < totalPixels; i++) {
      const r = imageData[ptr++];
      const g = imageData[ptr++];
      const b = imageData[ptr++];
      ptr++; // skip alpha

      // Store in CHW format (RGB channels)
      tensor[i] = r; // R channel
      tensor[stride + i] = g; // G channel
      tensor[2 * stride + i] = b; // B channel
    }

    return tensor;
  }

  /**
   * Calculate new bounding box with scale
   * @param srcW Source width
   * @param srcH Source height
   * @param bbox Original bounding box
   * @param scale Scale factor
   * @returns New bounding box
   */
  protected getNewBox(
    srcW: number,
    srcH: number,
    bbox: { x: number; y: number; width: number; height: number },
    scale: number
  ): { x: number; y: number; width: number; height: number } {
    const { x, y, width: boxW, height: boxH } = bbox;

    // Adjust scale to fit within image bounds
    const adjustedScale = Math.min(
      (srcH - 1) / boxH,
      Math.min((srcW - 1) / boxW, scale)
    );

    const newWidth = boxW * adjustedScale;
    const newHeight = boxH * adjustedScale;
    const centerX = boxW / 2 + x;
    const centerY = boxH / 2 + y;

    let leftTopX = centerX - newWidth / 2;
    let leftTopY = centerY - newHeight / 2;
    let rightBottomX = centerX + newWidth / 2;
    let rightBottomY = centerY + newHeight / 2;

    // Adjust bounds to stay within image
    if (leftTopX < 0) {
      rightBottomX -= leftTopX;
      leftTopX = 0;
    }
    if (leftTopY < 0) {
      rightBottomY -= leftTopY;
      leftTopY = 0;
    }
    if (rightBottomX > srcW - 1) {
      leftTopX -= rightBottomX - srcW + 1;
      rightBottomX = srcW - 1;
    }
    if (rightBottomY > srcH - 1) {
      leftTopY -= rightBottomY - srcH + 1;
      rightBottomY = srcH - 1;
    }

    return {
      x: Math.round(leftTopX),
      y: Math.round(leftTopY),
      width: Math.round(rightBottomX - leftTopX),
      height: Math.round(rightBottomY - leftTopY),
    };
  }

  /**
   * Run inference on a model
   * @param session ONNX session
   * @param tensor Input tensor
   * @param shape Input shape
   * @returns Model output
   */
  protected async inference(
    session: ort.InferenceSession,
    tensor: Float32Array,
    shape: number[]
  ): Promise<ort.InferenceSession.OnnxValueMapType> {
    const feeds: Record<string, ort.Tensor> = {};
    const inputName = session.inputNames[0]!;

    feeds[inputName] = new ort.Tensor("float32", tensor, shape);

    const result = await session.run(feeds);
    return result;
  }

  /**
   * Apply softmax to logits
   * @param logits Input logits
   * @returns Softmax probabilities
   */
  protected softmax(logits: Float32Array): Float32Array {
    const max = Math.max(...Array.from(logits));
    const exps = logits.map((x) => Math.exp(x - max));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return new Float32Array(exps.map((x) => x / sumExps));
  }

  /**
   * Postprocess model outputs
   * @param firstOutput First model output
   * @param secondOutput Second model output
   * @returns Spoofing result
   */
  protected postprocess(
    firstOutput: ort.InferenceSession.OnnxValueMapType,
    secondOutput: ort.InferenceSession.OnnxValueMapType
  ): SpoofingResult {
    // Get output tensors
    const firstOutputName = this.firstSession!.outputNames[0]!;
    const secondOutputName = this.secondSession!.outputNames[0]!;

    const firstLogits = firstOutput[firstOutputName]!.data as Float32Array;
    const secondLogits = secondOutput[secondOutputName]!.data as Float32Array;

    // Apply softmax to both outputs
    const firstProbs = this.softmax(firstLogits);
    const secondProbs = this.softmax(secondLogits);

    // Combine predictions (sum of probabilities)
    const prediction = new Float32Array(3);
    for (let i = 0; i < 3; i++) {
      prediction[i] = firstProbs[i] + secondProbs[i];
    }

    // Get the label with highest probability
    let maxIdx = 0;
    let maxVal = prediction[0];
    for (let i = 1; i < 3; i++) {
      if (prediction[i] > maxVal) {
        maxVal = prediction[i];
        maxIdx = i;
      }
    }

    // Label 1 means real, others mean fake
    const isReal = maxIdx === 1;
    const score = prediction[maxIdx] / 2; // Average of two models

    return {
      real: isReal,
      score: score,
    };
  }

  async destroy(): Promise<void> {
    await this.firstSession?.release();
    await this.secondSession?.release();

    this.firstSession = null;
    this.secondSession = null;
  }
}
