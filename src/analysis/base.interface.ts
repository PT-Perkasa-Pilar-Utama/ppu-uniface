import { Base } from "../global.interface";

export abstract class BaseAnalysis extends Base {
  /** Class name for logging */
  protected abstract override className: string;

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
