import type { PlatformProvider } from "../core/platform.js";
import { Base } from "../global.interface.js";

/**
 * Base class for analysis models (e.g., spoofing detection)
 */
export abstract class BaseAnalysis extends Base {
  /** Class name for logging */
  protected abstract override className: string;

  constructor(platform?: PlatformProvider) {
    super(platform);
  }

  /** Initializes the analysis model */
  abstract initialize(): Promise<void>;

  /**
   * Checks if the model has been initialized
   * @returns True if initialized
   */
  protected abstract isInitialized(): boolean;

  /** Releases model resources */
  abstract destroy(): Promise<void>;
}
