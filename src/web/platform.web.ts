import * as ort from "onnxruntime-web";
import type { CoreCanvas, PlatformProvider } from "../core/platform.js";
import { logger } from "../logger.js";

// Set WASM paths for onnxruntime-web to load from CDN
if (typeof globalThis !== "undefined" && !ort.env.wasm.wasmPaths) {
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
}

/**
 * Web (browser) implementation of PlatformProvider.
 * Uses onnxruntime-web and Canvas API for browser-based inference.
 */
export class WebPlatformProvider implements PlatformProvider {
  readonly ort: PlatformProvider["ort"] = {
    InferenceSession: ort.InferenceSession,
    Tensor: ort.Tensor,
  };

  createCanvas(width: number, height: number): CoreCanvas {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }

  async prepareCanvas(image: ArrayBuffer): Promise<CoreCanvas> {
    const blob = new Blob([image]);
    const bitmap = await createImageBitmap(blob);

    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;

    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();

    return canvas;
  }

  async loadResource(
    source: string | ArrayBuffer | undefined,
    defaultUrl: string,
  ): Promise<ArrayBuffer> {
    if (source instanceof ArrayBuffer) {
      logger(
        "WebPlatform",
        "loadResource",
        "Loading resource from ArrayBuffer",
      );
      return source;
    }

    const url = typeof source === "string" ? source : defaultUrl;
    logger("WebPlatform", "loadResource", `Fetching resource from: ${url}`);

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch resource from ${url}`);
    }

    return response.arrayBuffer();
  }

  private _runtimeInitialized = false;

  async initRuntime(): Promise<void> {
    if (this._runtimeInitialized) return;
    this._runtimeInitialized = true;

    // Import ppu-ocv/web for OpenCV initialization if available
    try {
      const { ImageProcessor } = await import("ppu-ocv/web");
      await ImageProcessor.initRuntime();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("already started") || msg.includes("already init")) {
        // Already initialized — safe to ignore
        return;
      }
      // ppu-ocv/web may not be available — that's fine for basic usage
      logger(
        "WebPlatform",
        "initRuntime",
        "ppu-ocv/web not available, skipping OpenCV init",
      );
    }
  }
}

/** Singleton default Web platform provider */
export const defaultWebPlatform: WebPlatformProvider =
  new WebPlatformProvider();
