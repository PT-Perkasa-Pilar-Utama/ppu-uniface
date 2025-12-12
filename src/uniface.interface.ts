import type { SpoofingResult } from "./analysis/spoofing.ana.js";
import type {
  DetectOptions,
  DetectionResult,
} from "./detection/base.interface.js";
import type { RecognitionResult } from "./recognition/base.interface.js";
import type { VerificationResult } from "./verification/base.interface.js";

/**
 * Full verification result containing all detection, recognition, and verification data
 */
export interface UnifaceFullResult {
  /** Detection results for both faces */
  detection: {
    /** Detection result for first face */
    face1: DetectionResult | null;
    /** Detection result for second face */
    face2: DetectionResult | null;
  };
  /** Recognition results for both faces */
  recognition: {
    /** Recognition result for first face */
    face1: RecognitionResult;
    /** Recognition result for second face */
    face2: RecognitionResult;
  };
  /** Spoofing analysis results for both faces */
  spoofing: {
    /** Spoofing result for first face */
    face1: SpoofingResult | null;
    /** Spoofing result for second face */
    face2: SpoofingResult | null;
  };
  /** Verification result comparing the two faces */
  verification: VerificationResult;
}

/**
 * Compact verification result containing only essential verification data
 */
export interface UnifaceCompactResult {
  /** Multiple faces detection flags for both images */
  multipleFaces: {
    /** Multiple faces flag for first image */
    face1: DetectionResult["multipleFaces"] | null;
    /** Multiple faces flag for second image */
    face2: DetectionResult["multipleFaces"] | null;
  };
  /** Spoofing detection flags for both images */
  spoofing: {
    /** Spoofing flag for first image */
    face1: boolean | null;
    /** Spoofing flag for second image */
    face2: boolean | null;
  };
  /** Whether the two faces are verified as the same person */
  verified: VerificationResult["verified"];
  /** Similarity score between two faces (0-1) */
  similarity: number;
}

/**
 * Options for face verification
 */
export interface UnifaceVerifyOptions {
  /** Return compact result instead of full result
   * @default true
   * */
  compact: boolean;
  /** Optional detection options to override model-level defaults */
  detection?: DetectOptions;
  /** Optional verification threshold to override model-level default.
   * Falls back to model-level verification confidence threshold if not provided.
   */
  threshold?: number;
}
