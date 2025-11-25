import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";

import * as path from "path";
import { CACHE_DIR } from "./constant";
import { logger } from "./logger";

/**
 * Base class providing resource loading and caching functionality
 */
export class Base {
  /** Name of the class for logging purposes */
  protected className: string = "Base";

  /**
   * Fetches a resource from a URL and caches it locally
   * @param url - URL to fetch the resource from
   * @returns ArrayBuffer containing the resource data
   */
  async fetchAndCache(url: string): Promise<ArrayBuffer> {
    const fileName = path.basename(new URL(url).pathname);
    const cachePath = path.join(CACHE_DIR, fileName);

    if (existsSync(cachePath)) {
      this.log("fetchAndCache", `Loading cached resource from: ${cachePath}`);
      const buf = readFileSync(cachePath);
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }

    this.log(
      "fetchAndCache",
      `Downloading resource: ${fileName}\n` +
        `                 Cached at: ${CACHE_DIR}`
    );
    this.log("fetchAndCache", `Fetching resource from URL: ${url}`);

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

    this.log("fetchAndCache", `Caching resource to: ${cachePath}`);
    if (!existsSync(CACHE_DIR)) {
      mkdirSync(CACHE_DIR, { recursive: true });
    }
    writeFileSync(cachePath, Buffer.from(buffer));

    return buffer.buffer;
  }

  /**
   * Loads a resource from various sources
   * @param source - ArrayBuffer, file path, URL, or undefined to use defaultUrl
   * @param defaultUrl - Default URL to use if source is undefined
   * @returns ArrayBuffer containing the resource data
   */
  async loadResource(
    source: string | ArrayBuffer | undefined,
    defaultUrl: string
  ): Promise<ArrayBuffer> {
    if (source instanceof ArrayBuffer) {
      this.log("loadResource", "Loading resource from ArrayBuffer");
      return source;
    }

    if (typeof source === "string") {
      if (source.startsWith("http")) {
        return this.fetchAndCache(source);
      } else {
        const resolvedPath = path.resolve(process.cwd(), source);
        this.log("loadResource", `Loading resource from path: ${resolvedPath}`);
        const buf = readFileSync(resolvedPath);
        return buf.buffer.slice(
          buf.byteOffset,
          buf.byteOffset + buf.byteLength
        );
      }
    }

    return this.fetchAndCache(defaultUrl);
  }

  /**
   * Logs a message using the logger utility
   * @param methodName - Name of the method
   * @param message - Log message
   */
  protected log(methodName: string, message: string): void {
    logger(this.className, methodName, message);
  }
}
