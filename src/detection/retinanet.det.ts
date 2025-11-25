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

  protected override detectionOptions: DetectionModelOptions = {
    threshold: {
      confidence: 0.5,
      nonMaximumSuppression: 0.4,
    },
    topK: {
      preNonMaximumSuppression: 5000,
      postNonMaxiumSuppression: 750,
    },
    size: {
      input: [640, 640],
    },
  };

  constructor() {
    super();
  }

  async initialize(): Promise<void> {
    this.log("initialize", "Starting RetinaNet initialization...");
    await ImageProcessor.initRuntime();

    this.log(
      "initialize",
      `Loading model from: ${GITHUB_BASE_URL}${this.modelPath}`
    );
    const buffer = await this.loadResource(
      undefined,
      `${GITHUB_BASE_URL}${this.modelPath}`
    );
    this.log("initialize", `Model buffer loaded: ${buffer.byteLength} bytes`);

    this.log("initialize", "Creating ONNX inference session...");
    this.session = await ort.InferenceSession.create(new Uint8Array(buffer));

    this.log(
      "initialize",
      `Model loaded successfully\n\tinput: ${this.session.inputNames}\n\toutput: ${this.session.outputNames}`
    );
    this.log("initialize", "RetinaNet initialization completed");
  }

  /**
   * Detect face in an image, prioritize face that is largest
   * @param image
   */
  async detect(image: ArrayBuffer | Canvas): Promise<DetectionResult | null> {
    this.log("detect", "Starting face detection...");
    if (!this.isInitialized)
      throw Error(`${this.className} session was not initialized`);

    this.log("detect", "Preparing canvas...");
    const canvas =
      image instanceof ArrayBuffer
        ? await ImageProcessor.prepareCanvas(image)
        : image;
    const { height, width } = canvas;
    this.log("detect", `Canvas size: ${width}x${height}`);

    this.log("detect", "Preprocessing image...");
    const tensor = this.preprocess(canvas, height, width);
    this.log("detect", `Preprocessed tensor size: ${tensor.length}`);

    this.log("detect", "Running inference...");
    const outputs = await this.inference(tensor, [height, width]);
    this.log("detect", "Inference completed");

    this.log("detect", "Postprocessing results...");
    const result = this.postprocess(outputs, [width, height]);
    const numDetections = result.boxes.length / 4;
    this.log("detect", `Found ${numDetections} face(s) after postprocessing`);

    if (numDetections === 0) {
      this.log("detect", "No faces detected");
      return null;
    }

    const multipleFaces = numDetections > 1;
    this.log("detect", `Multiple faces: ${multipleFaces}`);

    this.log("detect", "Finding largest face...");
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
    this.log(
      "detect",
      `Largest face index: ${largestIdx}, area: ${largestArea.toFixed(2)}`
    );

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
    this.log(
      "detect",
      `Bounding box: x=${box.x}, y=${box.y}, w=${box.width}, h=${box.height}`
    );

    const landmarks: number[][] = [];
    for (let i = 0; i < 5; i++) {
      const idx = landmarkIdx + i * 2;
      landmarks.push([
        Math.round(result.landmarks[idx]! * width),
        Math.round(result.landmarks[idx + 1]! * height),
      ]);
    }
    this.log("detect", `Extracted ${landmarks.length} landmarks`);

    const confidence = result.scores[largestIdx]!;
    this.log("detect", `Confidence: ${(confidence * 100).toFixed(2)}%`);
    this.log("detect", "Face detection completed successfully");

    return {
      box,
      confidence,
      landmarks,
      multipleFaces,
      spoofing: false,
    };
  }

  preprocess(canvas: Canvas, height: number, width: number): Float32Array {
    this.log("preprocess", `Starting preprocessing: ${width}x${height}`);
    const channels = 3;
    const mean = [104, 117, 123];
    const tensor = new Float32Array(1 * channels * height * width);
    this.log(
      "preprocess",
      `Tensor size: ${tensor.length}, Mean: [${mean.join(", ")}]`
    );

    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, width, height).data;
    this.log("preprocess", `Image data size: ${imageData.length}`);

    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        for (let c = 0; c < channels; c++) {
          const pixelIndex = h * width + w;
          const rgbaIndex = pixelIndex * 4;
          const chwIndex = c * height * width + h * width + w;

          tensor[chwIndex] = imageData[rgbaIndex + c]! - mean[c]!;
        }
      }
    }

    this.log("preprocess", "Preprocessing completed");
    return tensor;
  }

  async inference(
    tensor: Float32Array,
    shape: [number, number]
  ): Promise<ort.InferenceSession.OnnxValueMapType> {
    this.log(
      "inference",
      `Starting inference with shape: [${shape.join(", ")}]`
    );
    if (!this.isInitialized)
      throw Error(`${this.className} session was not initialized`);

    const feeds: Record<string, ort.Tensor> = {};
    const inputName = this.session!.inputNames[0]!;
    const inputShape = [1, 3, ...shape];
    this.log(
      "inference",
      `Input name: ${inputName}, shape: [${inputShape.join(", ")}]`
    );

    feeds[inputName] = new ort.Tensor("float32", tensor, inputShape);

    this.log("inference", "Running ONNX session...");
    const result = await this.session!.run(feeds);
    this.log("inference", "Inference completed");
    return result;
  }

  postprocess(
    outputs: ort.InferenceSession.OnnxValueMapType,
    shape: [number, number]
  ): { boxes: Float32Array; scores: Float32Array; landmarks: Float32Array } {
    this.log(
      "postprocess",
      `Starting postprocessing with shape: [${shape.join(", ")}]`
    );
    const { loc, conf, landmarks } = outputs;

    const locData = loc!.data as Float32Array;
    const confData = conf!.data as Float32Array;
    const landmarksData = landmarks!.data as Float32Array;
    this.log(
      "postprocess",
      `Output sizes - loc: ${locData.length}, conf: ${confData.length}, landmarks: ${landmarksData.length}`
    );

    const numPriors = conf!.dims[1]!;
    this.log("postprocess", `Number of priors: ${numPriors}`);

    this.log("postprocess", "Generating anchors...");
    const priors = this.generateAnchors([shape[1], shape[0]]);
    this.log("postprocess", `Generated ${priors.length / 4} anchors`);

    this.log("postprocess", "Decoding boxes...");
    const boxesDecoded = this.decodeBoxes(locData, priors);
    this.log("postprocess", "Decoding landmarks...");
    const landmarksDecoded = this.decodeLandmarks(landmarksData, priors);
    this.log("postprocess", "Decoding completed");

    const scores = new Float32Array(numPriors);

    for (let i = 0; i < numPriors; i++) {
      scores[i] = confData[i * 2 + 1]!;
    }

    this.log(
      "postprocess",
      `Filtering by confidence threshold: ${this.detectionOptions.threshold.confidence}`
    );
    const filteredBoxes: number[] = [];
    const filteredLandmarks: number[] = [];
    const filteredScores: number[] = [];

    for (let i = 0; i < numPriors; i++) {
      if (scores[i]! <= this.detectionOptions.threshold.confidence) continue;

      const boxIdx = i * 4;
      filteredBoxes.push(
        boxesDecoded[boxIdx]!,
        boxesDecoded[boxIdx + 1]!,
        boxesDecoded[boxIdx + 2]!,
        boxesDecoded[boxIdx + 3]!
      );

      const landmarkIdx = i * 10;
      for (let j = 0; j < 10; j++) {
        filteredLandmarks.push(landmarksDecoded[landmarkIdx + j]!);
      }

      filteredScores.push(scores[i]!);
    }

    this.log("postprocess", `Filtered to ${filteredScores.length} detections`);
    let filteredBoxesArray = new Float32Array(filteredBoxes);
    let filteredLandmarksArray = new Float32Array(filteredLandmarks);
    let filteredScoresArray = new Float32Array(filteredScores);

    const numFiltered = filteredScoresArray.length;
    if (numFiltered > this.detectionOptions.topK.preNonMaximumSuppression) {
      this.log(
        "postprocess",
        `Applying pre-NMS top-K: ${this.detectionOptions.topK.preNonMaximumSuppression}`
      );
      const indices = Array.from({ length: numFiltered }, (_, i) => i);
      indices.sort((a, b) => filteredScoresArray[b]! - filteredScoresArray[a]!);

      const topK = this.detectionOptions.topK.preNonMaximumSuppression;
      const topKIndices = indices.slice(0, topK);

      const topKBoxes = new Float32Array(topK * 4);
      const topKLandmarks = new Float32Array(topK * 10);
      const topKScores = new Float32Array(topK);

      for (let i = 0; i < topK; i++) {
        const idx = topKIndices[i]!;

        for (let j = 0; j < 4; j++) {
          topKBoxes[i * 4 + j] = filteredBoxesArray[idx * 4 + j]!;
        }

        for (let j = 0; j < 10; j++) {
          topKLandmarks[i * 10 + j] = filteredLandmarksArray[idx * 10 + j]!;
        }

        topKScores[i] = filteredScoresArray[idx]!;
      }

      filteredBoxesArray = topKBoxes;
      filteredLandmarksArray = topKLandmarks;
      filteredScoresArray = topKScores;
      this.log("postprocess", `Reduced to top ${topK} detections`);
    }

    this.log("postprocess", "Applying NMS and final top-K...");
    const result = this.applyNMSAndTopK(
      filteredBoxesArray,
      filteredScoresArray,
      filteredLandmarksArray
    );
    this.log(
      "postprocess",
      `Postprocessing completed with ${
        result.boxes.length / 4
      } final detections`
    );
    return result;
  }

  protected applyNMSAndTopK(
    filteredBoxesArray: Float32Array,
    filteredScoresArray: Float32Array,
    filteredLandmarksArray: Float32Array
  ): { boxes: Float32Array; scores: Float32Array; landmarks: Float32Array } {
    this.log(
      "applyNMSAndTopK",
      `Starting NMS with ${filteredScoresArray.length} detections`
    );
    const numDetections = filteredScoresArray.length;

    if (numDetections === 0) {
      this.log("applyNMSAndTopK", "No detections to process");
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

    this.log(
      "applyNMSAndTopK",
      `Applying NMS with threshold: ${this.detectionOptions.threshold.nonMaximumSuppression}`
    );
    const keep = this.nonMaximumSuppression(
      detections,
      numDetections,
      this.detectionOptions.threshold.nonMaximumSuppression
    );
    this.log("applyNMSAndTopK", `NMS kept ${keep.length} detections`);

    const topK = Math.min(
      keep.length,
      this.detectionOptions.topK.postNonMaxiumSuppression
    );
    const keepTopK = keep.slice(0, topK);
    this.log("applyNMSAndTopK", `Keeping top ${topK} detections`);

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

    this.log("applyNMSAndTopK", "NMS and top-K filtering completed");
    return {
      boxes: finalBoxes,
      scores: finalScores,
      landmarks: finalLandmarks,
    };
  }

  override async destroy(): Promise<void> {
    this.log("destroy", "Releasing session...");
    await this.session?.release();
    this.session = null;
  }
}
