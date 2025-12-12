import * as ort from "onnxruntime-node";
import { Canvas, createCanvas, ImageProcessor } from "ppu-ocv";
import { GITHUB_BASE_URL } from "../constant.js";
import { BaseAnalysis } from "./base.interface.js";

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
  /** The given score from analysis (confidence of the predicted label) */
  score: number;
  /** Probability of the image being real (0-1) */
  realness: number;
  /** Probability of the image being fake (0-1) */
  fakeness: number;
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
   * Analyze whether a face is spoofed
   * @param image The image (full image recommended)
   * @param facialArea Optional facial bounding box {x, y, width, height}.
   *                   If not provided, uses the entire image as the facial area.
   */
  async analyze(
    image: ArrayBuffer | Canvas,
    facialArea?: { x: number; y: number; width: number; height: number }
  ): Promise<SpoofingResult> {
    if (!this.isInitialized)
      throw Error(`${this.className} session was not initialized`);

    const canvas =
      image instanceof ArrayBuffer
        ? await ImageProcessor.prepareCanvas(image)
        : image;

    const bbox = facialArea || {
      x: Math.floor(canvas.width / 2 - canvas.width / 8),
      y: Math.floor(canvas.height / 2 - canvas.height / 8),
      width: Math.floor(canvas.width / 4),
      height: Math.floor(canvas.height / 4),
    };

    const [firstOutput, secondOutput] = await Promise.all([
      this.preprocessPromise(canvas, bbox, 2.7, 80, 80, this.firstSession!),
      this.preprocessPromise(canvas, bbox, 4.0, 80, 80, this.secondSession!),
    ]);

    const result = this.postprocess(firstOutput, secondOutput);
    this.log(
      "analyze",
      `Spoofing analysis: real=${result.real}, score=${result.score.toFixed(
        3
      )}, realness=${result.realness.toFixed(3)}`
    );

    return result;
  }

  /**
   * Preprocess image by cropping from original image with scale and resizing
   * @param canvas Original full image canvas
   * @param bbox Facial bounding box
   * @param scale Scale factor for cropping
   * @param outW Output width
   * @param outH Output height
   * @returns Preprocessed tensor
   */
  protected async preprocessPromise(
    canvas: Canvas,
    bbox: { x: number; y: number; width: number; height: number },
    scale: number,
    outW: number,
    outH: number,
    session: ort.InferenceSession
  ): Promise<ort.InferenceSession.OnnxValueMapType> {
    const tensor = this.preprocessWithCrop(canvas, bbox, scale, outW, outH);
    const output = await this.inference(session!, tensor, [1, 3, 80, 80]);
    return output;
  }

  /**
   * Preprocess image by cropping from original image with scale and resizing
   * @param canvas Original full image canvas
   * @param bbox Facial bounding box
   * @param scale Scale factor for cropping
   * @param outW Output width
   * @param outH Output height
   * @returns Preprocessed tensor
   */
  protected preprocessWithCrop(
    canvas: Canvas,
    bbox: { x: number; y: number; width: number; height: number },
    scale: number,
    outW: number,
    outH: number
  ): Float32Array {
    const { width: srcW, height: srcH } = canvas;
    const newBox = this.getNewBox(srcW, srcH, bbox, scale);

    const ctx = canvas.getContext("2d");
    const croppedImageData = ctx.getImageData(
      newBox.x,
      newBox.y,
      newBox.width,
      newBox.height
    );

    const tempCanvas = createCanvas(newBox.width, newBox.height);
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.putImageData(croppedImageData, 0, 0);

    // Resize to target size
    let resizedCanvas = tempCanvas;
    if (newBox.width !== outW || newBox.height !== outH) {
      const targetCanvas = createCanvas(outW, outH);
      const targetCtx = targetCanvas.getContext("2d");
      targetCtx.imageSmoothingEnabled = true;
      targetCtx.imageSmoothingQuality = "high";
      targetCtx.drawImage(
        tempCanvas,
        0,
        0,
        newBox.width,
        newBox.height,
        0,
        0,
        outW,
        outH
      );
      resizedCanvas = targetCanvas;
    }

    return this.canvasToTensor(resizedCanvas, outW, outH);
  }

  /**
   * Simple preprocess by resizing only (for pre-cropped faces)
   * @param canvas Input canvas
   * @param outW Output width
   * @param outH Output height
   * @returns Preprocessed tensor
   */
  protected preprocess(
    canvas: Canvas,
    outW: number,
    outH: number
  ): Float32Array {
    const { width, height } = canvas;

    let resizedCanvas = canvas;
    if (width !== outW || height !== outH) {
      const targetCanvas = createCanvas(outW, outH);
      const targetCtx = targetCanvas.getContext("2d");
      targetCtx.imageSmoothingEnabled = true;
      targetCtx.imageSmoothingQuality = "high";
      targetCtx.drawImage(canvas, 0, 0, width, height, 0, 0, outW, outH);
      resizedCanvas = targetCanvas;
    }

    return this.canvasToTensor(resizedCanvas, outW, outH);
  }

  /**
   * Convert canvas to tensor in CHW format
   */
  protected canvasToTensor(
    canvas: Canvas,
    width: number,
    height: number
  ): Float32Array {
    const channels = 3;
    const tensor = new Float32Array(channels * height * width);

    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, width, height).data;

    const totalPixels = height * width;
    const stride = height * width;

    let ptr = 0;
    for (let i = 0; i < totalPixels; i++) {
      const r = imageData[ptr++];
      const g = imageData[ptr++];
      const b = imageData[ptr++];
      ptr++;

      tensor[i] = b;
      tensor[stride + i] = g;
      tensor[2 * stride + i] = r;
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
      x: Math.floor(leftTopX),
      y: Math.floor(leftTopY),
      width: Math.floor(rightBottomX) - Math.floor(leftTopX) + 1,
      height: Math.floor(rightBottomY) - Math.floor(leftTopY) + 1,
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
    const firstOutputName = this.firstSession!.outputNames[0]!;
    const secondOutputName = this.secondSession!.outputNames[0]!;

    const firstLogits = firstOutput[firstOutputName]!.data as Float32Array;
    const secondLogits = secondOutput[secondOutputName]!.data as Float32Array;

    const firstProbs = this.softmax(firstLogits);
    const secondProbs = this.softmax(secondLogits);

    const prediction = new Float32Array(3);
    for (let i = 0; i < 3; i++) {
      prediction[i] = firstProbs[i] + secondProbs[i];
    }

    const probabilities = new Float32Array(3);
    for (let i = 0; i < 3; i++) {
      probabilities[i] = prediction[i] / 2;
    }

    let maxIdx = 0;
    let maxVal = probabilities[0];
    for (let i = 1; i < 3; i++) {
      if (probabilities[i] > maxVal) {
        maxVal = probabilities[i];
        maxIdx = i;
      }
    }

    const isReal = maxIdx === 1;
    const realness = probabilities[1];

    const fakeness = probabilities[0] + probabilities[2];
    const score = isReal ? realness : fakeness;

    return {
      real: isReal,
      score: score,
      realness: realness,
      fakeness: fakeness,
    };
  }

  async destroy(): Promise<void> {
    await this.firstSession?.release();
    await this.secondSession?.release();

    this.firstSession = null;
    this.secondSession = null;
  }
}
