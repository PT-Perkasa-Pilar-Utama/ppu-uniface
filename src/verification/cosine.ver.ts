import {
  BaseVerification,
  type VerificationModelOptions,
  type VerificationResult,
} from "./base.interface.js";

export class CosineVerification extends BaseVerification {
  protected verificationOptions: VerificationModelOptions = {
    threshold: 0.7,
  };

  constructor(options: Partial<VerificationModelOptions> = {}) {
    super();
    this.verificationOptions = {
      ...this.verificationOptions,
      ...options,
    };
  }

  compare(
    embedding1: Float32Array,
    embedding2: Float32Array,
    threshold?: number
  ): VerificationResult {
    // Use provided threshold or fall back to model-level threshold
    const effectiveThreshold = threshold ?? this.verificationOptions.threshold;

    this.log("compare", `Starting cosine similarity calculation...`);
    this.log(
      "compare",
      `Embedding sizes: ${embedding1.length} vs ${embedding2.length}`
    );

    this.log("compare", "Calculating dot product and norms...");
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < embedding1.length; i++) {
      dotProduct += embedding1[i]! * embedding2[i]!;
      normA += embedding1[i]! * embedding1[i]!;
      normB += embedding2[i]! * embedding2[i]!;
    }

    this.log("compare", `Dot product calculated: ${dotProduct.toFixed(6)}`);
    this.log(
      "compare",
      `Norm A squared: ${normA.toFixed(6)}, Norm B squared: ${normB.toFixed(6)}`
    );

    if (normA === 0 || normB === 0) {
      this.log("compare", "Zero norm detected, returning zero similarity");
      return {
        similarity: 0.0,
        threshold: effectiveThreshold,
        verified: false,
      };
    }

    const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));

    this.log(
      "compare",
      `  Dot product: ${dotProduct.toFixed(6)}, NormA: ${Math.sqrt(
        normA
      ).toFixed(6)}, NormB: ${Math.sqrt(normB).toFixed(6)}`
    );
    this.log(
      "compare",
      `  Cosine similarity: ${similarity.toFixed(6)} (${(
        similarity * 100
      ).toFixed(2)}%)`
    );
    this.log(
      "compare",
      `  Threshold: ${effectiveThreshold}, Verified: ${similarity >= effectiveThreshold}`
    );

    return {
      similarity: similarity,
      threshold: effectiveThreshold,
      verified: similarity >= effectiveThreshold,
    };
  }
}
