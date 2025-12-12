import * as ort from "onnxruntime-node";
import type { Canvas } from "ppu-ocv";
import { Base } from "../global.interface.js";

/**
 * Configuration options for face recognition models
 */
export interface RecognitionModelOptions {
  /** Model input and output sizes */
  size: {
    /** Input tensor shape [batch, height, width, channels] */
    input: [number, number, number, number];
    /** Output tensor shape [batch, embedding_size] */
    output: [number, number];
  };
}

/**
 * Face recognition result containing embedding vector
 */
export interface RecognitionResult {
  /** Face embedding vector */
  embedding: Float32Array;
}

/**
 * Base class for face recognition models
 */
export abstract class BaseRecognition extends Base {
  /** Class name for logging */
  protected abstract override className: string;
  /** Path to the model file */
  protected abstract modelPath: string;
  /** Recognition configuration options */
  protected abstract recognitionOptions: RecognitionModelOptions;
  /** ONNX inference session */
  protected abstract session: ort.InferenceSession | null;

  /** Initializes the recognition model */
  abstract initialize(): Promise<void>;

  /**
   * Generates face embedding from image
   * @param image - Input image as ArrayBuffer or Canvas
   * @returns Recognition result with embedding
   */
  abstract recognize(image: ArrayBuffer | Canvas): Promise<RecognitionResult>;

  /**
   * Preprocesses image for model inference
   * @param canvas - Input canvas
   * @returns Preprocessed tensor
   */
  abstract preprocess(canvas: Canvas): Float32Array;

  /**
   * Runs model inference
   * @param tensor - Preprocessed input tensor
   * @returns Model outputs
   */
  abstract inference(
    tensor: Float32Array
  ): Promise<ort.InferenceSession.OnnxValueMapType>;

  /**
   * Extracts embedding from model outputs
   * @param outputs - Raw model outputs
   * @returns Face embedding vector
   */
  abstract postprocess(
    outputs: ort.InferenceSession.OnnxValueMapType
  ): Float32Array;

  /** Releases model resources */
  abstract destroy(): Promise<void>;

  /**
   * Checks if the model has been initialized
   * @returns True if initialized
   */
  protected isInitialized(): boolean {
    return this.session !== null;
  }
}
