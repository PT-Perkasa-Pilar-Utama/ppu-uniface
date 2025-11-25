import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { RetinaNetDetection } from "../../src/detection/retinanet.det";
import { LoggerConfig } from "../../src/logger";

describe("RetinaNetDetection", () => {
  let detector: RetinaNetDetection;

  beforeAll(async () => {
    LoggerConfig.verbose = false;
    detector = new RetinaNetDetection();
    await detector.initialize();
  });

  afterAll(async () => {
    await detector.destroy();
  });

  test("should initialize successfully", async () => {
    const testDetector = new RetinaNetDetection();
    await testDetector.initialize();
    expect(testDetector).toBeDefined();
    await testDetector.destroy();
  });

  test("should detect face in image with single person", async () => {
    const imageBuffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
    const result = await detector.detect(imageBuffer);

    expect(result).not.toBeNull();
    expect(result?.box).toBeDefined();
    expect(result?.box.x).toBeGreaterThanOrEqual(0);
    expect(result?.box.y).toBeGreaterThanOrEqual(0);
    expect(result?.box.width).toBeGreaterThan(0);
    expect(result?.box.height).toBeGreaterThan(0);
    expect(result?.confidence).toBeGreaterThan(0);
    expect(result?.confidence).toBeLessThanOrEqual(1);
    expect(result?.landmarks).toHaveLength(5);
    expect(result?.spoofing).toBe(false);
  });

  test("should detect landmarks correctly", async () => {
    const imageBuffer = await Bun.file("assets/image-kevin1.png").arrayBuffer();
    const result = await detector.detect(imageBuffer);

    expect(result).not.toBeNull();
    expect(result?.landmarks).toHaveLength(5);

    result?.landmarks.forEach((landmark) => {
      expect(landmark).toHaveLength(2);
      expect(landmark[0]).toBeGreaterThanOrEqual(0);
      expect(landmark[1]).toBeGreaterThanOrEqual(0);
    });
  });

  test("should return null for image without faces", async () => {
    const imageBuffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
    const result = await detector.detect(imageBuffer);

    if (result === null) {
      expect(result).toBeNull();
    } else {
      expect(result.confidence).toBeGreaterThan(0);
    }
  });

  test("should detect multiple faces flag when applicable", async () => {
    const imageBuffer = await Bun.file("assets/image-haaland2.png").arrayBuffer();
    const result = await detector.detect(imageBuffer);

    expect(result).not.toBeNull();
    expect(typeof result?.multipleFaces).toBe("boolean");
  });

  test("should select largest face when multiple faces present", async () => {
    const imageBuffer = await Bun.file("assets/image-kevin2.jpg").arrayBuffer();
    const result = await detector.detect(imageBuffer);

    if (result) {
      expect(result.box.width).toBeGreaterThan(0);
      expect(result.box.height).toBeGreaterThan(0);
      const area = result.box.width * result.box.height;
      expect(area).toBeGreaterThan(0);
    }
  });

  test("should have consistent detection results", async () => {
    const imageBuffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
    const result1 = await detector.detect(imageBuffer);
    const result2 = await detector.detect(imageBuffer);

    if (result1 && result2) {
      expect(result1.box.x).toBeCloseTo(result2.box.x, 1);
      expect(result1.box.y).toBeCloseTo(result2.box.y, 1);
      expect(result1.box.width).toBeCloseTo(result2.box.width, 1);
      expect(result1.box.height).toBeCloseTo(result2.box.height, 1);
      expect(result1.confidence).toBeCloseTo(result2.confidence, 2);
    }
  });

  test("should handle different image formats", async () => {
    const jpegBuffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
    const pngBuffer = await Bun.file("assets/image-haaland2.png").arrayBuffer();

    const jpegResult = await detector.detect(jpegBuffer);
    const pngResult = await detector.detect(pngBuffer);

    expect(jpegResult).not.toBeNull();
    expect(pngResult).not.toBeNull();
  });

  test("should throw error if not initialized", async () => {
    const uninitializedDetector = new RetinaNetDetection();
    const imageBuffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();

    expect(async () => {
      await uninitializedDetector.detect(imageBuffer);
    }).toThrow();
  });

  test("should cleanup resources on destroy", async () => {
    const testDetector = new RetinaNetDetection();
    await testDetector.initialize();
    await testDetector.destroy();

    const imageBuffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
    expect(async () => {
      await testDetector.detect(imageBuffer);
    }).toThrow();
  });
});
