import type { InferenceSession } from "onnxruntime-common";
import { GITHUB_BASE_URL } from "../constant.js";
import type { CoreCanvas, PlatformProvider } from "../core/platform.js";
import {
  BaseRecognition,
  type RecognitionModelOptions,
  type RecognitionResult,
} from "./base.interface.js";

export class FaceNet512Recognition extends BaseRecognition {
  protected override className: string = "FaceNet512Recognition";
  protected override modelPath: string = "recognition/facenet512.onnx";
  protected override session: InferenceSession | null = null;

  protected override recognitionOptions: RecognitionModelOptions = {
    size: {
      input: [1, 160, 160, 3],
      output: [1, 512],
    },
  };

  constructor(
    options: Partial<RecognitionModelOptions> = {},
    platform?: PlatformProvider,
  ) {
    super(platform);
    this.recognitionOptions = {
      ...this.recognitionOptions,
      ...options,
      size: {
        ...this.recognitionOptions.size,
        ...(options.size || {}),
      },
    };
  }

  async initialize(): Promise<void> {
    this.log("initialize", "Starting FaceNet512 initialization...");
    await this.platform.initRuntime();

    const buffer = await this.loadResource(
      undefined,
      `${GITHUB_BASE_URL}${this.modelPath}`,
    );

    this.session = await this.platform.ort.InferenceSession.create(
      new Uint8Array(buffer),
    );

    this.log("initialize", "FaceNet512 initialized");
  }

  async recognize(image: ArrayBuffer | CoreCanvas): Promise<RecognitionResult> {
    if (!this.isInitialized)
      throw Error(`${this.className} session was not initialized`);

    const canvas =
      image instanceof ArrayBuffer
        ? await this.platform.prepareCanvas(image)
        : image;

    const tensor = this.preprocess(canvas);
    const outputs = await this.inference(tensor);
    const result = this.postprocess(outputs);

    return {
      embedding: result,
    };
  }

  preprocess(canvas: CoreCanvas): Float32Array {
    const { width, height } = canvas;

    const expectedHeight = this.recognitionOptions.size.input[1];
    const expectedWidth = this.recognitionOptions.size.input[2];

    let resizedCanvas = canvas;
    if (width !== expectedWidth || height !== expectedHeight) {
      resizedCanvas = this.platform.createCanvas(expectedWidth, expectedHeight);
      const ctx = resizedCanvas.getContext("2d");
      ctx.drawImage(
        canvas,
        0,
        0,
        width,
        height,
        0,
        0,
        expectedWidth,
        expectedHeight,
      );
    }

    const tensorSize =
      this.recognitionOptions.size.input[3] * expectedHeight * expectedWidth;
    const tensor = new Float32Array(tensorSize);

    const ctx = resizedCanvas.getContext("2d");
    const imageData = ctx.getImageData(
      0,
      0,
      expectedWidth,
      expectedHeight,
    ).data;
    const totalPixels = expectedHeight * expectedWidth;
    let ptr = 0;

    for (let i = 0; i < totalPixels; i++) {
      const r = imageData[ptr++];
      const g = imageData[ptr++];
      const b = imageData[ptr++];
      ptr++;

      const idx = i * 3;
      tensor[idx] = (b - 127.5) / 128.0;
      tensor[idx + 1] = (g - 127.5) / 128.0;
      tensor[idx + 2] = (r - 127.5) / 128.0;
    }

    return tensor;
  }

  async inference(
    tensor: Float32Array,
  ): Promise<InferenceSession.OnnxValueMapType> {
    if (!this.isInitialized)
      throw Error(`${this.className} session was not initialized`);

    const feeds: Record<string, any> = {};
    const inputName = this.session!.inputNames[0]!;

    feeds[inputName] = new this.platform.ort.Tensor(
      "float32",
      tensor,
      this.recognitionOptions.size.input,
    );

    const result = await this.session!.run(feeds);
    return result;
  }

  postprocess(outputs: InferenceSession.OnnxValueMapType): Float32Array {
    const outputName = this.session!.outputNames[0]!;
    const outputTensor = outputs[outputName]!;
    const embedding = outputTensor.data as Float32Array;

    return embedding;
  }

  override async destroy(): Promise<void> {
    await this.session?.release();
    this.session = null;
  }
}
