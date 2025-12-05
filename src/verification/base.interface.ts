import { Base } from "../global.interface";

/**
 * Configuration options for face verification
 */
export interface VerificationModelOptions {
  /** Similarity threshold for verification */
  threshold: number;
}

/**
 * Face verification result
 */
export interface VerificationResult {
  /** Similarity score between two faces (0-1) */
  similarity: number;
  /** Whether faces are verified as the same person */
  verified: boolean;
  /** Threshold used for verification to determine if passing then the face is verified */
  threshold: number;
}

/**
 * Base class for face verification methods
 */
export abstract class BaseVerification extends Base {
  /** Verification configuration options */
  protected abstract verificationOptions: VerificationModelOptions;

  /**
   * Compares two face embeddings
   * @param embedding1 - First face embedding
   * @param embedding2 - Second face embedding
   * @param threshold - Optional threshold to override model-level default
   * @returns Verification result with similarity and verification status
   */
  abstract compare(
    embedding1: Float32Array,
    embedding2: Float32Array,
    threshold?: number
  ): VerificationResult;
}
