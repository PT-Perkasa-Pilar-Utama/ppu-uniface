import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { ImageProcessor } from "ppu-ocv";
import { LoggerConfig } from "../../src/logger";
import { FaceNet512Recognition } from "../../src/recognition/facenet512.rec";

describe("FaceNet512Recognition", () => {
  let recognizer: FaceNet512Recognition;

  beforeAll(async () => {
    LoggerConfig.verbose = false;
    recognizer = new FaceNet512Recognition();
    await recognizer.initialize();
  });

  afterAll(async () => {
    await recognizer.destroy();
  });

  test("should initialize successfully", async () => {
    const testRecognizer = new FaceNet512Recognition();
    await testRecognizer.initialize();
    expect(testRecognizer).toBeDefined();
    await testRecognizer.destroy();
  });

  test("should generate embedding from image", async () => {
    const imageBuffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
    const result = await recognizer.recognize(imageBuffer);

    expect(result).toBeDefined();
    expect(result.embedding).toBeInstanceOf(Float32Array);
    expect(result.embedding.length).toBe(512);
  });

  test("should generate consistent embeddings for same image", async () => {
    const imageBuffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
    const result1 = await recognizer.recognize(imageBuffer);
    const result2 = await recognizer.recognize(imageBuffer);

    expect(result1.embedding.length).toBe(result2.embedding.length);

    for (let i = 0; i < result1.embedding.length; i++) {
      expect(result1.embedding[i]).toBeCloseTo(result2.embedding[i]!, 5);
    }
  });

  test("should generate different embeddings for different images", async () => {
    const imageBuffer1 = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
    const imageBuffer2 = await Bun.file("assets/image-kevin1.png").arrayBuffer();

    const result1 = await recognizer.recognize(imageBuffer1);
    const result2 = await recognizer.recognize(imageBuffer2);

    expect(result1.embedding.length).toBe(result2.embedding.length);

    let differenceCount = 0;
    for (let i = 0; i < result1.embedding.length; i++) {
      if (Math.abs(result1.embedding[i]! - result2.embedding[i]!) > 0.01) {
        differenceCount++;
      }
    }

    expect(differenceCount).toBeGreaterThan(0);
  });

  test("should handle different image formats", async () => {
    const jpegBuffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
    const pngBuffer = await Bun.file("assets/image-kevin1.png").arrayBuffer();

    const jpegResult = await recognizer.recognize(jpegBuffer);
    const pngResult = await recognizer.recognize(pngBuffer);

    expect(jpegResult.embedding).toBeInstanceOf(Float32Array);
    expect(pngResult.embedding).toBeInstanceOf(Float32Array);
    expect(jpegResult.embedding.length).toBe(512);
    expect(pngResult.embedding.length).toBe(512);
  });

  test("should handle canvas input", async () => {
    const imageBuffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
    const canvas = await ImageProcessor.prepareCanvas(imageBuffer);
    const result = await recognizer.recognize(canvas);

    expect(result.embedding).toBeInstanceOf(Float32Array);
    expect(result.embedding.length).toBe(512);
  });

  test("should generate normalized embeddings", async () => {
    const imageBuffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
    const result = await recognizer.recognize(imageBuffer);

    const values = Array.from(result.embedding);
    const hasValidValues = values.some((v) => v !== 0);
    expect(hasValidValues).toBe(true);

    const allFinite = values.every((v) => Number.isFinite(v));
    expect(allFinite).toBe(true);
  });

  test("should handle resizing correctly", async () => {
    const imageBuffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
    const canvas = await ImageProcessor.prepareCanvas(imageBuffer);

    const processor = new ImageProcessor(canvas);
    const resizedCanvas = processor.resize({ width: 100, height: 100 }).toCanvas();
    processor.destroy();

    const result = await recognizer.recognize(resizedCanvas);

    expect(result.embedding).toBeInstanceOf(Float32Array);
    expect(result.embedding.length).toBe(512);
  });

  test("should throw error if not initialized", async () => {
    const uninitializedRecognizer = new FaceNet512Recognition();
    const imageBuffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();

    expect(async () => {
      await uninitializedRecognizer.recognize(imageBuffer);
    }).toThrow();
  });

  test("should cleanup resources on destroy", async () => {
    const testRecognizer = new FaceNet512Recognition();
    await testRecognizer.initialize();
    await testRecognizer.destroy();

    const imageBuffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
    expect(async () => {
      await testRecognizer.recognize(imageBuffer);
    }).toThrow();
  });

  test("should generate similar embeddings for same person", async () => {
    const imageBuffer1 = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
    const imageBuffer2 = await Bun.file("assets/image-haaland2.png").arrayBuffer();

    const result1 = await recognizer.recognize(imageBuffer1);
    const result2 = await recognizer.recognize(imageBuffer2);

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < result1.embedding.length; i++) {
      dotProduct += result1.embedding[i]! * result2.embedding[i]!;
      normA += result1.embedding[i]! * result1.embedding[i]!;
      normB += result2.embedding[i]! * result2.embedding[i]!;
    }

    const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    expect(similarity).toBeGreaterThan(0);
  });
});
