import * as ort from "onnxruntime-node";
import { Canvas, ImageProcessor } from "ppu-ocv";
import { GITHUB_BASE_URL } from "../constant";
import {
  BaseDetection,
  type DetectionModelOptions,
  type DetectionResult,
} from "./base.interface";

export class RetinaNetDetection extends BaseDetection {
  protected override className: string = "RetinaNetDetection";
  protected override modelPath: string = "detection/retinaface_mv2.onnx";
  protected override session: ort.InferenceSession | null = null;

  protected anchorsCache: Float32Array | null = null;
  protected anchorsCacheShape: string = "";

  protected override detectionOptions: DetectionModelOptions = {
    threshold: {
      confidence: 0.7,
      nonMaximumSuppression: 0.4,
    },
    topK: {
      preNonMaximumSuppression: 5000,
      postNonMaxiumSuppression: 750,
    },
    size: {
      input: [320, 320],
    },
  };

  constructor(options: Partial<DetectionModelOptions> = {}) {
    super();
    this.detectionOptions = {
      ...this.detectionOptions,
      ...options,
      threshold: {
        ...this.detectionOptions.threshold,
        ...(options.threshold || {}),
      },
      topK: {
        ...this.detectionOptions.topK,
        ...(options.topK || {}),
      },
      size: {
        ...this.detectionOptions.size,
        ...(options.size || {}),
      },
    };
  }

  async initialize(): Promise<void> {
    this.log("initialize", "Starting RetinaNet initialization...");
    await ImageProcessor.initRuntime();

    const buffer = await this.loadResource(
      undefined,
      `${GITHUB_BASE_URL}${this.modelPath}`
    );

    this.session = await ort.InferenceSession.create(new Uint8Array(buffer));

    const [inputH, inputW] = this.detectionOptions.size.input;
    this.anchorsCache = this.generateAnchors([inputH, inputW]);
    this.anchorsCacheShape = `${inputH}x${inputW}`;

    this.log(
      "initialize",
      `RetinaNet initialized: ${this.anchorsCache.length / 4} anchors for ${inputW}x${inputH}`
    );
  }

  /**
   * Detect face in an image, prioritize face that is largest
   * @param image
   */
  async detect(image: ArrayBuffer | Canvas): Promise<DetectionResult | null> {
    if (!this.isInitialized)
      throw Error(`${this.className} session was not initialized`);

    const canvas =
      image instanceof ArrayBuffer
        ? await ImageProcessor.prepareCanvas(image)
        : image;
    const { height, width } = canvas;

    const [targetW, targetH] = this.detectionOptions.size.input;
    let inputCanvas = canvas;

    if (width !== targetW || height !== targetH) {
      const processor = new ImageProcessor(canvas);
      inputCanvas = processor
        .resize({ width: targetW, height: targetH })
        .toCanvas();
      processor.destroy();
    }

    const tensor = this.preprocess(inputCanvas, targetH, targetW);
    const outputs = await this.inference(tensor, [targetH, targetW]);
    const result = this.postprocess(outputs);
    const numDetections = result.boxes.length / 4;

    if (numDetections === 0) {
      return null;
    }

    const multipleFaces = numDetections > 1;
    let largestIdx = 0;
    let largestArea = 0;

    for (let i = 0; i < numDetections; i++) {
      const idx = i * 4;
      const x1 = result.boxes[idx]!;
      const y1 = result.boxes[idx + 1]!;

      const x2 = result.boxes[idx + 2]!;
      const y2 = result.boxes[idx + 3]!;

      const area = (x2 - x1) * (y2 - y1);

      if (area > largestArea) {
        largestArea = area;
        largestIdx = i;
      }
    }
    const boxIdx = largestIdx * 4;
    const landmarkIdx = largestIdx * 10;

    const x1 = result.boxes[boxIdx]!;
    const y1 = result.boxes[boxIdx + 1]!;
    const x2 = result.boxes[boxIdx + 2]!;
    const y2 = result.boxes[boxIdx + 3]!;

    const box = {
      x: Math.round(x1 * width),
      y: Math.round(y1 * height),
      width: Math.round((x2 - x1) * width),
      height: Math.round((y2 - y1) * height),
    };

    const landmarks: number[][] = [];
    for (let i = 0; i < 5; i++) {
      const idx = landmarkIdx + i * 2;
      landmarks.push([
        Math.round(result.landmarks[idx]! * width),
        Math.round(result.landmarks[idx + 1]! * height),
      ]);
    }

    const confidence = result.scores[largestIdx]!;
    this.log(
      "detect",
      `Detected face: confidence=${(confidence * 100).toFixed(1)}%, multiple=${multipleFaces}`
    );

    return {
      box,
      confidence,
      landmarks,
      multipleFaces,
    };
  }

  preprocess(canvas: Canvas, height: number, width: number): Float32Array {
    const channels = 3;
    const tensor = new Float32Array(1 * channels * height * width);

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

      tensor[i] = r - 104;
      tensor[stride + i] = g - 117;
      tensor[2 * stride + i] = b - 123;
    }

    return tensor;
  }

  async inference(
    tensor: Float32Array,
    shape: [number, number]
  ): Promise<ort.InferenceSession.OnnxValueMapType> {
    if (!this.isInitialized)
      throw Error(`${this.className} session was not initialized`);

    const feeds: Record<string, ort.Tensor> = {};
    const inputName = this.session!.inputNames[0]!;
    const inputShape = [1, 3, ...shape];

    feeds[inputName] = new ort.Tensor("float32", tensor, inputShape);

    const result = await this.session!.run(feeds);
    return result;
  }

  postprocess(outputs: ort.InferenceSession.OnnxValueMapType): {
    boxes: Float32Array;
    scores: Float32Array;
    landmarks: Float32Array;
  } {
    const { loc, conf, landmarks } = outputs;

    const locData = loc!.data as Float32Array;
    const confData = conf!.data as Float32Array;
    const landmarksData = landmarks!.data as Float32Array;

    const numPriors = conf!.dims[1]!;

    const threshold = this.detectionOptions.threshold.confidence;
    const passingIndices: number[] = [];

    for (let i = 0; i < numPriors; i++) {
      if (confData[i * 2 + 1]! > threshold) {
        passingIndices.push(i);
      }
    }

    const numPassing = passingIndices.length;

    if (numPassing === 0) {
      return {
        boxes: new Float32Array(0),
        scores: new Float32Array(0),
        landmarks: new Float32Array(0),
      };
    }

    if (!this.anchorsCache) {
      throw new Error("Anchors not initialized. Call initialize() first.");
    }
    const priors = this.anchorsCache;

    const locSubset = new Float32Array(numPassing * 4);
    const priorsSubset = new Float32Array(numPassing * 4);
    const landmarksSubset = new Float32Array(numPassing * 10);
    const scoresSubset = new Float32Array(numPassing);

    for (let i = 0; i < numPassing; i++) {
      const idx = passingIndices[i]!;
      scoresSubset[i] = confData[idx * 2 + 1]!;

      const locIdx = idx * 4;
      locSubset[i * 4] = locData[locIdx]!;
      locSubset[i * 4 + 1] = locData[locIdx + 1]!;
      locSubset[i * 4 + 2] = locData[locIdx + 2]!;
      locSubset[i * 4 + 3] = locData[locIdx + 3]!;

      priorsSubset[i * 4] = priors[locIdx]!;
      priorsSubset[i * 4 + 1] = priors[locIdx + 1]!;
      priorsSubset[i * 4 + 2] = priors[locIdx + 2]!;
      priorsSubset[i * 4 + 3] = priors[locIdx + 3]!;

      const lmIdx = idx * 10;
      for (let j = 0; j < 10; j++) {
        landmarksSubset[i * 10 + j] = landmarksData[lmIdx + j]!;
      }
    }

    const boxesDecoded = this.decodeBoxes(locSubset, priorsSubset);
    const landmarksDecoded = this.decodeLandmarks(
      landmarksSubset,
      priorsSubset
    );

    const result = this.applyNMSAndTopK(
      boxesDecoded,
      scoresSubset,
      landmarksDecoded
    );

    return result;
  }

  protected applyNMSAndTopK(
    filteredBoxesArray: Float32Array,
    filteredScoresArray: Float32Array,
    filteredLandmarksArray: Float32Array
  ): { boxes: Float32Array; scores: Float32Array; landmarks: Float32Array } {
    const numDetections = filteredScoresArray.length;

    if (numDetections === 0) {
      return {
        boxes: new Float32Array(0),
        scores: new Float32Array(0),
        landmarks: new Float32Array(0),
      };
    }

    const detections = new Float32Array(numDetections * 5);
    for (let i = 0; i < numDetections; i++) {
      const boxIdx = i * 4;
      const detIdx = i * 5;

      detections[detIdx] = filteredBoxesArray[boxIdx]!;
      detections[detIdx + 1] = filteredBoxesArray[boxIdx + 1]!;
      detections[detIdx + 2] = filteredBoxesArray[boxIdx + 2]!;
      detections[detIdx + 3] = filteredBoxesArray[boxIdx + 3]!;
      detections[detIdx + 4] = filteredScoresArray[i]!;
    }

    const keep = this.nonMaximumSuppression(
      detections,
      numDetections,
      this.detectionOptions.threshold.nonMaximumSuppression
    );

    const topK = Math.min(
      keep.length,
      this.detectionOptions.topK.postNonMaxiumSuppression
    );
    const keepTopK = keep.slice(0, topK);

    const finalBoxes = new Float32Array(keepTopK.length * 4);
    const finalScores = new Float32Array(keepTopK.length);
    const finalLandmarks = new Float32Array(keepTopK.length * 10);

    for (let i = 0; i < keepTopK.length; i++) {
      const idx = keepTopK[i]!;

      const boxSrcIdx = idx * 4;
      const boxDstIdx = i * 4;
      for (let j = 0; j < 4; j++) {
        finalBoxes[boxDstIdx + j] = filteredBoxesArray[boxSrcIdx + j]!;
      }

      finalScores[i] = filteredScoresArray[idx]!;

      const landmarkSrcIdx = idx * 10;
      const landmarkDstIdx = i * 10;
      for (let j = 0; j < 10; j++) {
        finalLandmarks[landmarkDstIdx + j] =
          filteredLandmarksArray[landmarkSrcIdx + j]!;
      }
    }

    return {
      boxes: finalBoxes,
      scores: finalScores,
      landmarks: finalLandmarks,
    };
  }

  override async destroy(): Promise<void> {
    await this.session?.release();
    this.session = null;
  }
}
