import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { alignAndCropFace, RetinaNetDetection } from "../../src";
import { SpoofingDetection } from "../../src/analysis/spoofing.ana";
import { LoggerConfig } from "../../src/logger";

describe("SpoofingDetection", () => {
  let spoofing: SpoofingDetection;
  let detection: RetinaNetDetection;

  beforeAll(async () => {
    LoggerConfig.verbose = false;

    spoofing = new SpoofingDetection();
    detection = new RetinaNetDetection();

    await spoofing.initialize();
    await detection.initialize();
  });

  afterAll(async () => {
    await spoofing.destroy();
  });

  test("should initialize successfully", async () => {
    const testSpoofing = new SpoofingDetection();
    await testSpoofing.initialize();

    expect(testSpoofing).toBeDefined();
    await testSpoofing.destroy();
  });

  test("should detect real face in image", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const result = await spoofing.analyze(imageBuffer);

    expect(result).toBeDefined();
    expect(typeof result.real).toBe("boolean");
    expect(typeof result.score).toBe("number");

    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(1);
    expect(result.real).toBe(true);
  });

  test("should detect real face in image via facial area", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const face = await detection.detect(imageBuffer);
    const crop = await alignAndCropFace(imageBuffer, face!);

    const result = await spoofing.analyze(imageBuffer, {
      x: 0,
      y: 0,
      width: crop.width,
      height: crop.height,
    });
    expect(result).toBeDefined();
    expect(typeof result.real).toBe("boolean");
    expect(typeof result.score).toBe("number");

    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(1);
    expect(result.real).toBe(true);
  });

  test("should detect real face in Kevin image", async () => {
    const imageBuffer = await Bun.file("assets/image-kevin1.png").arrayBuffer();
    const result = await spoofing.analyze(imageBuffer);

    expect(result).toBeDefined();
    expect(result.real).toBe(true);
    expect(result.score).toBeGreaterThan(0);
  });

  test("should detect real face in Magnus image", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-magnus1.png"
    ).arrayBuffer();
    const result = await spoofing.analyze(imageBuffer);

    expect(result).toBeDefined();
    expect(result.real).toBe(true);
    expect(result.score).toBeGreaterThan(0);
  });

  test("should analyze fake image 1 for spoofing via facial area", async () => {
    const imageBuffer = await Bun.file("assets/image-fake1.jpg").arrayBuffer();
    const face = await detection.detect(imageBuffer);
    const crop = await alignAndCropFace(imageBuffer, face!);

    const result = await spoofing.analyze(imageBuffer, {
      x: 0,
      y: 0,
      width: crop.width,
      height: crop.height,
    });

    expect(result).toBeDefined();
    expect(typeof result.real).toBe("boolean");
    expect(typeof result.score).toBe("number");
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(1);
    expect(result.real).toBe(false);
  });

  test("should analyze fake image 2 for spoofing via facial area", async () => {
    const imageBuffer = await Bun.file("assets/image-fake2.jpg").arrayBuffer();
    const face = await detection.detect(imageBuffer);
    const crop = await alignAndCropFace(imageBuffer, face!);

    const result = await spoofing.analyze(imageBuffer, {
      x: 0,
      y: 0,
      width: crop.width,
      height: crop.height,
    });

    expect(result).toBeDefined();
    expect(typeof result.real).toBe("boolean");
    expect(typeof result.score).toBe("number");
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(1);
    expect(result.real).toBe(false);
  });

  test("should analyze fake image 1 for spoofing", async () => {
    const imageBuffer = await Bun.file("assets/image-fake1.jpg").arrayBuffer();
    const result = await spoofing.analyze(imageBuffer);

    expect(result).toBeDefined();
    expect(typeof result.real).toBe("boolean");
    expect(typeof result.score).toBe("number");
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(1);
    expect(result.real).toBe(false);
  });

  test("should analyze fake image 2 for spoofing", async () => {
    const imageBuffer = await Bun.file("assets/image-fake2.jpg").arrayBuffer();
    const result = await spoofing.analyze(imageBuffer);

    expect(result).toBeDefined();
    expect(typeof result.real).toBe("boolean");
    expect(typeof result.score).toBe("number");
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(1);
    expect(result.real).toBe(false);
  });

  test("should provide consistent results for same image", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const result1 = await spoofing.analyze(imageBuffer);
    const result2 = await spoofing.analyze(imageBuffer);

    expect(result1.real).toBe(result2.real);
    expect(result1.score).toBeCloseTo(result2.score, 5);
    expect(result1.real).toBe(true);
  });

  test("should handle different image formats", async () => {
    const jpegBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const pngBuffer = await Bun.file("assets/image-kevin1.png").arrayBuffer();

    const jpegResult = await spoofing.analyze(jpegBuffer);
    const pngResult = await spoofing.analyze(pngBuffer);

    expect(jpegResult).toBeDefined();
    expect(pngResult).toBeDefined();

    expect(jpegResult.real).toBe(true);
    expect(pngResult.real).toBe(true);
  });

  test("should throw error if not initialized", async () => {
    const uninitializedSpoofing = new SpoofingDetection();
    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();

    expect(async () => {
      await uninitializedSpoofing.analyze(imageBuffer);
    }).toThrow();
  });

  test("should cleanup resources on destroy", async () => {
    const testSpoofing = new SpoofingDetection();
    await testSpoofing.initialize();
    await testSpoofing.destroy();

    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    expect(async () => {
      await testSpoofing.analyze(imageBuffer);
    }).toThrow();
  });

  test("should handle custom threshold option", async () => {
    const customSpoofing = new SpoofingDetection({ threshold: 0.6 });
    await customSpoofing.initialize();

    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const result = await customSpoofing.analyze(imageBuffer);

    expect(result).toBeDefined();
    expect(typeof result.real).toBe("boolean");

    await customSpoofing.destroy();
  });

  test("should handle disabled spoofing option", async () => {
    const disabledSpoofing = new SpoofingDetection({ enable: false });
    await disabledSpoofing.initialize();

    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();
    const result = await disabledSpoofing.analyze(imageBuffer);

    expect(result).toBeDefined();
    await disabledSpoofing.destroy();
  });
});
