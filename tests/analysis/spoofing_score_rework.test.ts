import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { SpoofingDetection } from "../../src/analysis/spoofing.ana.js";
import { RetinaNetDetection } from "../../src/detection/retinanet.det.js";
import { LoggerConfig } from "../../src/logger.js";

describe("SpoofingDetection Score Rework", () => {
  let spoofing: SpoofingDetection;
  let detection: RetinaNetDetection;

  beforeAll(async () => {
    LoggerConfig.verbose = false;
    spoofing = new SpoofingDetection();
    detection = new RetinaNetDetection();
    await Promise.all([spoofing.initialize(), detection.initialize()]);
  });

  afterAll(async () => {
    await spoofing.destroy();
    await detection.destroy();
  });

  test("should have realness and fakeness properties for real image", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();

    const face = await detection.detect(imageBuffer);
    expect(face).not.toBeNull();

    const result = await spoofing.analyze(imageBuffer, face!.box);

    expect(result).toBeDefined();
    expect(result.real).toBe(true);

    // Check new properties
    expect(typeof result.realness).toBe("number");
    expect(typeof result.fakeness).toBe("number");

    // For real image:
    // realness should be high (close to 1)
    // fakeness should be low (close to 0)
    expect(result.realness).toBeGreaterThan(0.9);
    expect(result.fakeness).toBeLessThan(0.1);

    // Score should match realness for real images (confidence of being real)
    expect(result.score).toBeCloseTo(result.realness, 5);
  });

  test("should have realness and fakeness properties for fake image", async () => {
    const imageBuffer = await Bun.file("assets/image-fake2.jpg").arrayBuffer();

    // Fake images might not have detectable faces if they are just crops or poor quality,
    // but typically we run detection first. Let's try detection.
    const face = await detection.detect(imageBuffer);

    // If detection fails, we might fall back to full image or center crop.
    // For this test, let's assume detection works or use undefined if null.
    const bbox = face ? face.box : undefined;

    const result = await spoofing.analyze(imageBuffer, bbox);

    expect(result).toBeDefined();
    expect(result.real).toBe(false);

    // Check new properties
    expect(typeof result.realness).toBe("number");
    expect(typeof result.fakeness).toBe("number");

    // For fake image:
    // realness should be low (close to 0)
    // fakeness should be high (close to 1)
    expect(result.realness).toBeLessThan(0.1);
    expect(result.fakeness).toBeGreaterThan(0.9);

    // Score should match fakeness for fake images (confidence of being fake)
    // Note: score is confidence of predicted label. Since predicted is fake, score is confidence of fake.
    expect(result.score).toBeCloseTo(result.fakeness, 5);
  });

  test("should sum to approximately 1", async () => {
    const imageBuffer = await Bun.file(
      "assets/image-haaland1.jpeg"
    ).arrayBuffer();

    const face = await detection.detect(imageBuffer);
    const result = await spoofing.analyze(imageBuffer, face?.box);

    // realness + fakeness should be approx 1
    expect(result.realness + result.fakeness).toBeCloseTo(1, 5);
  });
});
