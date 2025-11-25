import * as ort from "onnxruntime-node";
import { ImageProcessor, type Canvas } from "ppu-ocv";
import { GITHUB_BASE_URL } from "../constant";
import {
  BaseRecognition,
  type RecognitionModelOptions,
  type RecognitionResult,
} from "./base.interface";

export class FaceNet512Recognition extends BaseRecognition {
  protected override className: string = "FaceNet512Recognition";
  protected override modelPath: string = "recognition/facenet512.onnx";
  protected override session: ort.InferenceSession | null = null;

  protected override recognitionOptions: RecognitionModelOptions = {
    size: {
      input: [1, 160, 160, 3],
      output: [1, 512],
    },
  };

  constructor() {
    super();
  }

  async initialize(): Promise<void> {
    this.log("initialize", "Starting FaceNet512 initialization...");
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
    this.log("initialize", "FaceNet512 initialization completed");
  }

  async recognize(image: ArrayBuffer | Canvas): Promise<RecognitionResult> {
    this.log("recognize", "Starting face recognition...");
    if (!this.isInitialized)
      throw Error(`${this.className} session was not initialized`);

    this.log("recognize", "Preparing canvas...");
    const canvas =
      image instanceof ArrayBuffer
        ? await ImageProcessor.prepareCanvas(image)
        : image;
    this.log("recognize", `Canvas size: ${canvas.width}x${canvas.height}`);

    this.log("recognize", "Preprocessing image...");
    const tensor = this.preprocess(canvas);
    this.log("recognize", `Preprocessed tensor size: ${tensor.length}`);

    this.log("recognize", "Running inference...");
    const outputs = await this.inference(tensor);
    this.log("recognize", "Inference completed");

    this.log("recognize", "Postprocessing results...");
    const result = this.postprocess(outputs);
    this.log("recognize", `Generated embedding size: ${result.length}`);
    this.log("recognize", "Face recognition completed successfully");

    return {
      embedding: result,
    };
  }

  preprocess(canvas: Canvas): Float32Array {
    this.log("preprocess", "Starting preprocessing...");
    const { width, height } = canvas;
    this.log("preprocess", `Input canvas size: ${width}x${height}`);

    const expectedHeight = this.recognitionOptions.size.input[1];
    const expectedWidth = this.recognitionOptions.size.input[2];
    this.log(
      "preprocess",
      `Expected model input size: ${expectedWidth}x${expectedHeight}`
    );

    let resizedCanvas = canvas;
    if (width !== expectedWidth || height !== expectedHeight) {
      this.log(
        "preprocess",
        `Resizing canvas from ${width}x${height} to ${expectedWidth}x${expectedHeight}...`
      );
      const processor = new ImageProcessor(canvas);
      resizedCanvas = processor
        .resize({
          width: expectedWidth,
          height: expectedHeight,
        })
        .toCanvas();
      processor.destroy();
      this.log("preprocess", "Canvas resized successfully");
    }

    const tensorSize =
      this.recognitionOptions.size.input[3] * expectedHeight * expectedWidth;
    const tensor = new Float32Array(tensorSize);
    this.log("preprocess", `Tensor size: ${tensor.length}`);

    const ctx = resizedCanvas.getContext("2d");
    const imageData = ctx.getImageData(
      0,
      0,
      expectedWidth,
      expectedHeight
    ).data;
    this.log("preprocess", `Image data size: ${imageData.length}`);

    this.log(
      "preprocess",
      "Normalizing pixels (BGR order, mean=127.5, std=128.0)..."
    );
    for (let h = 0; h < expectedHeight; h++) {
      for (let w = 0; w < expectedWidth; w++) {
        const pixelIndex = h * expectedWidth + w;
        const rgbaIndex = pixelIndex * 4;
        const tensorIndex = pixelIndex * 3;

        tensor[tensorIndex] = (imageData[rgbaIndex + 2]! - 127.5) / 128.0;
        tensor[tensorIndex + 1] = (imageData[rgbaIndex + 1]! - 127.5) / 128.0;
        tensor[tensorIndex + 2] = (imageData[rgbaIndex]! - 127.5) / 128.0;
      }
    }

    this.log("preprocess", "Preprocessing completed");
    return tensor;
  }

  async inference(
    tensor: Float32Array
  ): Promise<ort.InferenceSession.OnnxValueMapType> {
    this.log("inference", "Starting inference...");
    if (!this.isInitialized)
      throw Error(`${this.className} session was not initialized`);

    const feeds: Record<string, ort.Tensor> = {};
    const inputName = this.session!.inputNames[0]!;
    this.log(
      "inference",
      `Input name: ${inputName}, shape: [${this.recognitionOptions.size.input.join(
        ", "
      )}]`
    );

    feeds[inputName] = new ort.Tensor(
      "float32",
      tensor,
      this.recognitionOptions.size.input
    );

    this.log("inference", "Running ONNX session...");
    const result = await this.session!.run(feeds);
    this.log("inference", "Inference completed");
    return result;
  }

  postprocess(outputs: ort.InferenceSession.OnnxValueMapType): Float32Array {
    this.log("postprocess", "Starting postprocessing...");
    const outputName = this.session!.outputNames[0]!;
    this.log("postprocess", `Output name: ${outputName}`);

    const outputTensor = outputs[outputName]!;
    this.log(
      "postprocess",
      `Output tensor shape: [${outputTensor.dims.join(", ")}]`
    );

    const embedding = outputTensor.data as Float32Array;
    this.log("postprocess", `Embedding size: ${embedding.length}`);
    this.log("postprocess", "Postprocessing completed");

    return embedding;
  }

  override async destroy(): Promise<void> {
    this.log("destroy", "Releasing session...");
    await this.session?.release();
    this.session = null;
  }
}
