import type { InferenceSession, Tensor } from "onnxruntime-common";

/**
 * Union type for canvas objects across platforms.
 * Node.js uses ppu-ocv Canvas, browsers use HTMLCanvasElement or OffscreenCanvas.
 */
// biome-ignore lint: platform-agnostic canvas type
export type CoreCanvas = any;

/**
 * Platform abstraction interface for cross-platform ONNX inference.
 * Implementations provide Node.js or browser-specific logic while
 * core services remain platform-agnostic.
 */
export interface PlatformProvider {
  /** ONNX Runtime module — InferenceSession and Tensor constructors */
  readonly ort: {
    InferenceSession: typeof InferenceSession;
    Tensor: typeof Tensor;
  };

  /**
   * Creates a blank canvas with the specified dimensions
   * @param width - Canvas width in pixels
   * @param height - Canvas height in pixels
   */
  createCanvas(width: number, height: number): CoreCanvas;

  /**
   * Converts raw image data (ArrayBuffer) into a canvas for processing
   * @param image - Raw image data
   */
  prepareCanvas(image: ArrayBuffer): Promise<CoreCanvas>;

  /**
   * Loads a resource from various sources (ArrayBuffer, file path, URL, or default URL)
   * @param source - ArrayBuffer, file path, URL, or undefined to use defaultUrl
   * @param defaultUrl - Fallback URL if source is undefined
   */
  loadResource(
    source: string | ArrayBuffer | undefined,
    defaultUrl: string,
  ): Promise<ArrayBuffer>;

  /** Initializes the image processing runtime (e.g. OpenCV) */
  initRuntime(): Promise<void>;
}
