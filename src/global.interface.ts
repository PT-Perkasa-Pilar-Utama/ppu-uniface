import type { PlatformProvider } from "./core/platform.js";
import { logger } from "./logger.js";

/**
 * Base class providing resource loading and platform-agnostic functionality.
 * Platform-specific behavior is delegated to the injected PlatformProvider.
 */
export class Base {
  /** Name of the class for logging purposes */
  protected className: string = "Base";

  /** Platform provider for cross-platform operations */
  protected platform: PlatformProvider;

  constructor(platform?: PlatformProvider) {
    if (platform) {
      this.platform = platform;
    } else {
      // Lazy-load NodePlatformProvider to avoid importing Node-specific modules in web builds
      this.platform = undefined!;
      this._loadNodePlatform();
    }
  }

  /** @internal Lazily loads the default Node platform provider */
  private _loadNodePlatform(): void {
    try {
      // Dynamic require to avoid bundling in web contexts
      const { defaultNodePlatform } = require("./core/platform.node.js");
      this.platform = defaultNodePlatform;
    } catch {
      // Will be set by web subclass before use
    }
  }

  /**
   * Loads a resource from various sources
   * @param source - ArrayBuffer, file path, URL, or undefined to use defaultUrl
   * @param defaultUrl - Default URL to use if source is undefined
   * @returns ArrayBuffer containing the resource data
   */
  async loadResource(
    source: string | ArrayBuffer | undefined,
    defaultUrl: string,
  ): Promise<ArrayBuffer> {
    return this.platform.loadResource(source, defaultUrl);
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
