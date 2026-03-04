import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import * as ort from "onnxruntime-node";
import * as os from "os";
import * as path from "path";
import { createCanvas, ImageProcessor } from "ppu-ocv";
import { logger } from "../logger.js";
import type { CoreCanvas, PlatformProvider } from "./platform.js";

/** Local cache directory for downloaded models (Node.js only) */
const CACHE_DIR: string = path.join(os.homedir(), ".cache", "ppu-uniface");

/**
 * Node.js implementation of PlatformProvider.
 * Uses onnxruntime-node, ppu-ocv, and filesystem for resource management.
 */
export class NodePlatformProvider implements PlatformProvider {
  readonly ort: PlatformProvider["ort"] = {
    InferenceSession: ort.InferenceSession,
    Tensor: ort.Tensor,
  };

  createCanvas(width: number, height: number): CoreCanvas {
    return createCanvas(width, height);
  }

  async prepareCanvas(image: ArrayBuffer): Promise<CoreCanvas> {
    return ImageProcessor.prepareCanvas(image);
  }

  async loadResource(
    source: string | ArrayBuffer | undefined,
    defaultUrl: string,
  ): Promise<ArrayBuffer> {
    if (source instanceof ArrayBuffer) {
      logger(
        "NodePlatform",
        "loadResource",
        "Loading resource from ArrayBuffer",
      );
      return source;
    }

    if (typeof source === "string") {
      if (source.startsWith("http")) {
        return this.fetchAndCache(source);
      } else {
        const resolvedPath = path.resolve(process.cwd(), source);
        logger(
          "NodePlatform",
          "loadResource",
          `Loading resource from path: ${resolvedPath}`,
        );
        const buf = readFileSync(resolvedPath);
        return buf.buffer.slice(
          buf.byteOffset,
          buf.byteOffset + buf.byteLength,
        );
      }
    }

    return this.fetchAndCache(defaultUrl);
  }

  private _runtimeInitialized = false;

  async initRuntime(): Promise<void> {
    if (this._runtimeInitialized) return;
    this._runtimeInitialized = true;
    await ImageProcessor.initRuntime();
  }

  /**
   * Fetches a resource from a URL and caches it locally
   * @param url - URL to fetch the resource from
   * @returns ArrayBuffer containing the resource data
   */
  private async fetchAndCache(url: string): Promise<ArrayBuffer> {
    const fileName = path.basename(new URL(url).pathname);
    const cachePath = path.join(CACHE_DIR, fileName);

    if (existsSync(cachePath)) {
      logger(
        "NodePlatform",
        "fetchAndCache",
        `Loading cached resource from: ${cachePath}`,
      );
      const buf = readFileSync(cachePath);
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }

    logger(
      "NodePlatform",
      "fetchAndCache",
      `Downloading resource: ${fileName}\n` +
        `                 Cached at: ${CACHE_DIR}`,
    );
    logger(
      "NodePlatform",
      "fetchAndCache",
      `Fetching resource from URL: ${url}`,
    );

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch resource from ${url}`);
    }
    if (!response.body) {
      throw new Error("Response body is null or undefined");
    }

    const contentLength = response.headers.get("Content-Length");
    const totalLength = contentLength ? parseInt(contentLength, 10) : 0;
    let receivedLength = 0;
    const chunks: Uint8Array[] = [];

    const reader = response.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      chunks.push(value);
      receivedLength += value.length;

      if (totalLength > 0) {
        const percentage = ((receivedLength / totalLength) * 100).toFixed(2);
        process.stdout.write(`\rDownloading... ${percentage}%`);
      }
    }
    process.stdout.write("\n");

    const buffer = new Uint8Array(receivedLength);
    let position = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, position);
      position += chunk.length;
    }

    logger(
      "NodePlatform",
      "fetchAndCache",
      `Caching resource to: ${cachePath}`,
    );
    if (!existsSync(CACHE_DIR)) {
      mkdirSync(CACHE_DIR, { recursive: true });
    }
    writeFileSync(cachePath, Buffer.from(buffer));

    return buffer.buffer;
  }
}

/** Singleton default Node.js platform provider */
export const defaultNodePlatform: NodePlatformProvider =
  new NodePlatformProvider();
