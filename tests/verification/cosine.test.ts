import { describe, expect, test } from "bun:test";
import { CosineVerification } from "../../src/verification/cosine.ver.js";

describe("CosineVerification", () => {
  test("should calculate cosine similarity for identical embeddings", () => {
    const verifier = new CosineVerification();
    const embedding1 = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const embedding2 = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0]);

    const result = verifier.compare(embedding1, embedding2);

    expect(result.similarity).toBeCloseTo(1.0, 5);
    expect(result.verified).toBe(true);
    expect(result.threshold).toBe(0.7);
  });

  test("should calculate cosine similarity for orthogonal embeddings", () => {
    const verifier = new CosineVerification();
    const embedding1 = new Float32Array([1.0, 0.0, 0.0]);
    const embedding2 = new Float32Array([0.0, 1.0, 0.0]);

    const result = verifier.compare(embedding1, embedding2);

    expect(result.similarity).toBeCloseTo(0.0, 5);
    expect(result.verified).toBe(false);
  });

  test("should calculate cosine similarity for opposite embeddings", () => {
    const verifier = new CosineVerification();
    const embedding1 = new Float32Array([1.0, 2.0, 3.0]);
    const embedding2 = new Float32Array([-1.0, -2.0, -3.0]);

    const result = verifier.compare(embedding1, embedding2);

    expect(result.similarity).toBeCloseTo(-1.0, 5);
    expect(result.verified).toBe(false);
  });

  test("should handle zero norm embeddings", () => {
    const verifier = new CosineVerification();
    const embedding1 = new Float32Array([0.0, 0.0, 0.0]);
    const embedding2 = new Float32Array([1.0, 2.0, 3.0]);

    const result = verifier.compare(embedding1, embedding2);

    expect(result.similarity).toBe(0.0);
    expect(result.verified).toBe(false);
  });

  test("should calculate similarity for similar but not identical embeddings", () => {
    const verifier = new CosineVerification();
    const embedding1 = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const embedding2 = new Float32Array([1.1, 2.1, 2.9, 4.0]);

    const result = verifier.compare(embedding1, embedding2);

    expect(result.similarity).toBeGreaterThan(0.95);
    expect(result.similarity).toBeLessThan(1.0);
    expect(result.verified).toBe(true);
  });

  test("should verify based on threshold", () => {
    const verifier = new CosineVerification();
    const embedding1 = new Float32Array([1.0, 2.0, 3.0]);
    const embedding2 = new Float32Array([1.0, 2.0, 2.5]);

    const result = verifier.compare(embedding1, embedding2);

    if (result.similarity >= 0.7) {
      expect(result.verified).toBe(true);
    } else {
      expect(result.verified).toBe(false);
    }
  });

  test("should handle large embeddings", () => {
    const verifier = new CosineVerification();
    const size = 512;
    const embedding1 = new Float32Array(size);
    const embedding2 = new Float32Array(size);

    for (let i = 0; i < size; i++) {
      embedding1[i] = Math.random();
      embedding2[i] = embedding1[i]! + Math.random() * 0.1;
    }

    const result = verifier.compare(embedding1, embedding2);

    expect(result.similarity).toBeGreaterThan(0);
    expect(result.similarity).toBeLessThanOrEqual(1);
    expect(typeof result.verified).toBe("boolean");
  });

  test("should handle normalized embeddings", () => {
    const verifier = new CosineVerification();
    const embedding1 = new Float32Array([0.6, 0.8]);
    const embedding2 = new Float32Array([0.8, 0.6]);

    const result = verifier.compare(embedding1, embedding2);

    expect(result.similarity).toBeGreaterThan(0);
    expect(result.similarity).toBeLessThan(1);
  });
});
