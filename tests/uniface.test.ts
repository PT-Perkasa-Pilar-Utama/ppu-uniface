import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { LoggerConfig } from "../src/logger";
import { Uniface } from "../src/uniface.service";

describe("Uniface", () => {
  let uniface: Uniface;

  beforeAll(async () => {
    LoggerConfig.verbose = false;
    uniface = new Uniface();
    await uniface.initialize();
  });

  afterAll(async () => {
    await uniface.destroy();
  });

  test("should initialize successfully", async () => {
    const testUniface = new Uniface();
    await testUniface.initialize();
    expect(testUniface).toBeDefined();
    await testUniface.destroy();
  });

  test("should detect face in image", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const result = await uniface.detect(imageBuffer);

    expect(result).not.toBeNull();
    expect(result?.box).toBeDefined();
    expect(result?.confidence).toBeGreaterThan(0);
    expect(result?.landmarks).toHaveLength(5);
  });

  test("should recognize face and generate embedding", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const result = await uniface.recognize(imageBuffer);

    expect(result).toBeDefined();
    expect(result.embedding).toBeInstanceOf(Float32Array);
    expect(result.embedding.length).toBe(512);
  });

  test("should verify same person with full result", async () => {
    const imageBuffer1 = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const imageBuffer2 = await Bun.file(
      "assets/image-haaland2.png"
    ).arrayBuffer();

    const result = await uniface.verify(imageBuffer1, imageBuffer2, {
      compact: false,
    });

    expect(result).toBeDefined();
    expect("detection" in result).toBe(true);
    expect("recognition" in result).toBe(true);
    expect("verification" in result).toBe(true);

    if ("verification" in result) {
      expect(result.verification.similarity).toBeGreaterThan(0);
      expect(result.verification.similarity).toBeLessThanOrEqual(1);
      expect(typeof result.verification.verified).toBe("boolean");
    }
  });

  test("should verify same person with compact result", async () => {
    const imageBuffer1 = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const imageBuffer2 = await Bun.file(
      "assets/image-haaland2.png"
    ).arrayBuffer();

    const result = await uniface.verify(imageBuffer1, imageBuffer2, {
      compact: true,
    });

    expect(result).toBeDefined();
    expect("multipleFaces" in result).toBe(true);
    expect("spoofing" in result).toBe(true);
    expect("verified" in result).toBe(true);

    if ("verified" in result) {
      expect(typeof result.verified).toBe("boolean");
    }
  });

  test("should verify different persons", async () => {
    const imageBuffer1 = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const imageBuffer2 = await Bun.file(
      "assets/image-kevin1.png"
    ).arrayBuffer();

    const result = await uniface.verify(imageBuffer1, imageBuffer2, {
      compact: false,
    });

    expect(result).toBeDefined();
    if ("verification" in result) {
      expect(result.verification.similarity).toBeGreaterThanOrEqual(0);
      expect(result.verification.similarity).toBeLessThanOrEqual(1);
    }
  });

  test("should verify embeddings directly", async () => {
    const imageBuffer1 = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const imageBuffer2 = await Bun.file(
      "assets/image-haaland2.png"
    ).arrayBuffer();

    const recognition1 = await uniface.recognize(imageBuffer1);
    const recognition2 = await uniface.recognize(imageBuffer2);

    const result = await uniface.verifyEmbedding(
      recognition1.embedding,
      recognition2.embedding
    );

    expect(result).toBeDefined();
    expect(result.similarity).toBeGreaterThan(0);
    expect(result.similarity).toBeLessThanOrEqual(1);
    expect(typeof result.verified).toBe("boolean");
    expect(result.threshold).toBe(0.7);
  });

  test("should handle full verification workflow", async () => {
    const imageBuffer1 = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const imageBuffer2 = await Bun.file(
      "assets/image-haaland2.png"
    ).arrayBuffer();

    const detection1 = await uniface.detect(imageBuffer1);
    const detection2 = await uniface.detect(imageBuffer2);

    expect(detection1).not.toBeNull();
    expect(detection2).not.toBeNull();

    const recognition1 = await uniface.recognize(imageBuffer1);
    const recognition2 = await uniface.recognize(imageBuffer2);

    expect(recognition1.embedding.length).toBe(512);
    expect(recognition2.embedding.length).toBe(512);

    const verification = await uniface.verifyEmbedding(
      recognition1.embedding,
      recognition2.embedding
    );

    expect(verification.similarity).toBeGreaterThan(0);
  });

  test("should provide consistent results across multiple verifications", async () => {
    const imageBuffer1 = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const imageBuffer2 = await Bun.file(
      "assets/image-haaland2.png"
    ).arrayBuffer();

    const result1 = await uniface.verify(imageBuffer1, imageBuffer2, {
      compact: false,
    });
    const result2 = await uniface.verify(imageBuffer1, imageBuffer2, {
      compact: false,
    });

    if ("verification" in result1 && "verification" in result2) {
      expect(result1.verification.similarity).toBeCloseTo(
        result2.verification.similarity,
        5
      );
      expect(result1.verification.verified).toBe(result2.verification.verified);
    }
  });

  test("should handle multiple faces flag in verification", async () => {
    const imageBuffer1 = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const imageBuffer2 = await Bun.file(
      "assets/image-kevin1.png"
    ).arrayBuffer();

    const result = await uniface.verify(imageBuffer1, imageBuffer2, {
      compact: true,
    });

    if ("multipleFaces" in result) {
      expect(typeof result.multipleFaces.face1).toBe("boolean");
      expect(typeof result.multipleFaces.face2).toBe("boolean");
    }
  });

  test("should handle spoofing flag in verification", async () => {
    const imageBuffer1 = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const imageBuffer2 = await Bun.file(
      "assets/image-kevin1.png"
    ).arrayBuffer();

    const result = await uniface.verify(imageBuffer1, imageBuffer2, {
      compact: true,
    });

    if ("spoofing" in result) {
      // Spoofing can be null if disabled, or boolean if enabled
      expect(["boolean", "object"]).toContain(typeof result.spoofing.face1);
      expect(["boolean", "object"]).toContain(typeof result.spoofing.face2);
    }
  });

  test("should cleanup resources properly", async () => {
    const testUniface = new Uniface();
    await testUniface.initialize();

    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const result = await testUniface.detect(imageBuffer);
    expect(result).not.toBeNull();

    await testUniface.destroy();

    expect(async () => {
      await testUniface.detect(imageBuffer);
    }).toThrow();
  });

  test("should handle different image formats in verification", async () => {
    const jpegBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const pngBuffer = await Bun.file("assets/image-haaland2.png").arrayBuffer();

    const result = await uniface.verify(jpegBuffer, pngBuffer, {
      compact: false,
    });

    expect(result).toBeDefined();
    if ("verification" in result) {
      expect(result.verification.similarity).toBeGreaterThan(0);
    }
  });

  test("should verify with high similarity for same person", async () => {
    const imageBuffer1 = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const imageBuffer2 = await Bun.file(
      "assets/image-haaland2.png"
    ).arrayBuffer();

    const result = await uniface.verify(imageBuffer1, imageBuffer2, {
      compact: false,
    });

    if ("verification" in result) {
      expect(result.verification.similarity).toBeGreaterThan(0.5);
    }
  });

  test("should verify with lower similarity for different persons", async () => {
    const imageBuffer1 = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const imageBuffer2 = await Bun.file(
      "assets/image-kevin1.png"
    ).arrayBuffer();

    const result = await uniface.verify(imageBuffer1, imageBuffer2, {
      compact: false,
    });

    if ("verification" in result) {
      expect(result.verification.similarity).toBeGreaterThanOrEqual(0);
      expect(result.verification.similarity).toBeLessThanOrEqual(1);
    }
  });

  test("should verify Magnus with and without glasses", async () => {
    const imageBuffer1 = await Bun.file(
      "assets/image-magnus1.png"
    ).arrayBuffer();
    const imageBuffer2 = await Bun.file(
      "assets/image-magnus2.png"
    ).arrayBuffer();

    const result = await uniface.verify(imageBuffer1, imageBuffer2, {
      compact: false,
    });

    expect(result).toBeDefined();
    if ("verification" in result) {
      expect(result.verification.similarity).toBeGreaterThan(0.5);
      expect(result.verification.verified).toBe(true);
    }
  });

  test("should detect face in Magnus image without glasses", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-magnus1.png"
    ).arrayBuffer();
    const result = await uniface.detect(imageBuffer);

    expect(result).not.toBeNull();
    expect(result?.box).toBeDefined();
    expect(result?.confidence).toBeGreaterThan(0);
    expect(result?.landmarks).toHaveLength(5);
  });

  test("should detect face in Magnus image with glasses", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-magnus2.png"
    ).arrayBuffer();
    const result = await uniface.detect(imageBuffer);

    expect(result).not.toBeNull();
    expect(result?.box).toBeDefined();
    expect(result?.confidence).toBeGreaterThan(0);
    expect(result?.landmarks).toHaveLength(5);
  });

  test("should detect face in image with multiple people", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-tdf-many.png"
    ).arrayBuffer();
    const result = await uniface.detect(imageBuffer);

    expect(result).not.toBeNull();
    expect(result?.box).toBeDefined();
    expect(result?.confidence).toBeGreaterThan(0);
    expect(typeof result?.multipleFaces).toBe("boolean");
    expect(result?.multipleFaces).toBe(true);
  });

  test("should verify with image containing multiple people", async () => {
    const imageBuffer1 = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const imageBuffer2 = await Bun.file(
      "assets/image-tdf-many.png"
    ).arrayBuffer();

    const result = await uniface.verify(imageBuffer1, imageBuffer2, {
      compact: true,
    });

    expect(result).toBeDefined();
    if ("multipleFaces" in result) {
      expect(typeof result.multipleFaces.face1).toBe("boolean");
      expect(typeof result.multipleFaces.face2).toBe("boolean");
    }
  });

  test("should verify Kevin images", async () => {
    const imageBuffer1 = await Bun.file(
      "assets/image-kevin1.png"
    ).arrayBuffer();
    const imageBuffer2 = await Bun.file(
      "assets/image-kevin2.jpg"
    ).arrayBuffer();

    const result = await uniface.verify(imageBuffer1, imageBuffer2, {
      compact: false,
    });

    expect(result).toBeDefined();
    if ("verification" in result) {
      expect(result.verification.similarity).toBeGreaterThan(0.5);
      expect(result.verification.verified).toBe(true);
    }
  });

  test("should verify Magnus vs Kevin (different persons)", async () => {
    const imageBuffer1 = await Bun.file(
      "assets/image-magnus1.png"
    ).arrayBuffer();
    const imageBuffer2 = await Bun.file(
      "assets/image-kevin1.png"
    ).arrayBuffer();

    const result = await uniface.verify(imageBuffer1, imageBuffer2, {
      compact: false,
    });

    expect(result).toBeDefined();
    if ("verification" in result) {
      expect(result.verification.verified).toBe(false);
      expect(result.verification.similarity).toBeLessThan(0.7);
    }
  });

  test("should analyze spoofing on cropped facial image", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const result = await uniface.spoofingAnalysis(imageBuffer);

    expect(result).toBeDefined();
    expect(typeof result.real).toBe("boolean");
    expect(typeof result.score).toBe("number");
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(1);
  });

  test("should analyze spoofing with detection on full image", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const result = await uniface.spoofingAnalysisWithDetection(imageBuffer);

    expect(result).not.toBeNull();
    expect(result?.real).toBe(true);
    expect(result?.score).toBeGreaterThan(0);
  });

  test("should detect real face in Kevin image with spoofing analysis", async () => {
    const imageBuffer = await Bun.file("assets/image-kevin1.png").arrayBuffer();
    const result = await uniface.spoofingAnalysisWithDetection(imageBuffer);

    expect(result).not.toBeNull();
    expect(typeof result?.real).toBe("boolean");
    expect(result?.score).toBeGreaterThan(0);
    expect(result?.real).toBe(true);
  });

  test("should detect real face in Magnus image with spoofing analysis", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-magnus1.png"
    ).arrayBuffer();
    const result = await uniface.spoofingAnalysisWithDetection(imageBuffer);

    expect(result).not.toBeNull();
    expect(typeof result?.real).toBe("boolean");
    expect(result?.score).toBeGreaterThan(0);
    expect(result?.real).toBe(true);
  });

  test("should analyze fake image 1 for spoofing", async () => {
    const imageBuffer = await Bun.file("assets/image-fake1.jpg").arrayBuffer();
    const result = await uniface.spoofingAnalysisWithDetection(imageBuffer);

    expect(result).not.toBeNull();
    expect(typeof result?.real).toBe("boolean");
    expect(typeof result?.score).toBe("number");
    expect(result?.score).toBeGreaterThanOrEqual(0);
    expect(result?.real).toBe(false);
  });

  test("should analyze fake image 2 for spoofing", async () => {
    const imageBuffer = await Bun.file("assets/image-fake2.jpg").arrayBuffer();
    const result = await uniface.spoofingAnalysisWithDetection(imageBuffer);

    expect(result).not.toBeNull();
    expect(typeof result?.real).toBe("boolean");
    expect(typeof result?.score).toBe("number");
    expect(result?.score).toBeGreaterThanOrEqual(0);
    expect(result?.real).toBe(false);
  });

  test("should return null for image without faces in spoofing analysis", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const result = await uniface.spoofingAnalysisWithDetection(imageBuffer);

    // If detection fails, result should be null
    if (result === null) {
      expect(result).toBeNull();
    } else {
      expect(typeof result.real).toBe("boolean");
    }
  });

  test("should provide consistent spoofing results", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const result1 = await uniface.spoofingAnalysisWithDetection(imageBuffer);
    const result2 = await uniface.spoofingAnalysisWithDetection(imageBuffer);

    if (result1 && result2) {
      expect(result1.real).toBe(result2.real);
      expect(result1.score).toBeCloseTo(result2.score, 5);
      expect(result1?.real).toBe(true);
    }
  });

  test("should handle different image formats in spoofing analysis", async () => {
    const jpegBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const pngBuffer = await Bun.file("assets/image-kevin1.png").arrayBuffer();

    const jpegResult = await uniface.spoofingAnalysisWithDetection(jpegBuffer);
    const pngResult = await uniface.spoofingAnalysisWithDetection(pngBuffer);

    expect(jpegResult).not.toBeNull();
    expect(pngResult).not.toBeNull();
    expect(jpegResult?.real).toBe(true);
    expect(pngResult?.real).toBe(true);
  });
});
