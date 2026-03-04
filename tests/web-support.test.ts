import { describe, expect, test } from "bun:test";
import { readFileSync, readdirSync } from "fs";
import { join, resolve } from "path";

/**
 * Web module export tests — validates that ppu-uniface/web exports
 * the correct types and classes for browser usage without Node-specific deps.
 */

const WEB_SRC_DIR = resolve(import.meta.dir, "../src/web");
const SRC_DIR = resolve(import.meta.dir, "../src");

describe("Web module exports", () => {
  test("should export Uniface class from web index", async () => {
    const webModule = await import("../src/web/index.js");
    expect(webModule.Uniface).toBeDefined();
    expect(typeof webModule.Uniface).toBe("function");
  });

  test("should export WebPlatformProvider", async () => {
    const webModule = await import("../src/web/index.js");
    expect(webModule.WebPlatformProvider).toBeDefined();
    expect(webModule.defaultWebPlatform).toBeDefined();
  });

  test("should export shared constants", async () => {
    const webModule = await import("../src/web/index.js");
    expect(webModule.GITHUB_BASE_URL).toBeDefined();
    expect(typeof webModule.GITHUB_BASE_URL).toBe("string");
    expect(webModule.GITHUB_BASE_URL).toContain("github");
  });

  test("should export logger utilities", async () => {
    const webModule = await import("../src/web/index.js");
    expect(webModule.logger).toBeDefined();
    expect(webModule.LoggerConfig).toBeDefined();
  });

  test("should export alignment functions", async () => {
    const webModule = await import("../src/web/index.js");
    expect(webModule.alignAndCropFace).toBeDefined();
    expect(webModule.alignFace).toBeDefined();
    expect(webModule.cropFace).toBeDefined();
  });

  test("should export individual service classes", async () => {
    const webModule = await import("../src/web/index.js");
    expect(webModule.RetinaNetDetection).toBeDefined();
    expect(webModule.FaceNet512Recognition).toBeDefined();
    expect(webModule.SpoofingDetection).toBeDefined();
    expect(webModule.CosineVerification).toBeDefined();
  });
});

describe("No Node.js imports in web files", () => {
  const nodeModules = [
    'from "fs"',
    'from "path"',
    'from "os"',
    'from "node:fs"',
    'from "node:path"',
    'from "node:os"',
    'from "onnxruntime-node"',
    "require(",
  ];

  const webFiles = readdirSync(WEB_SRC_DIR).filter((f) => f.endsWith(".ts"));

  for (const file of webFiles) {
    test(`${file} should not import Node-specific modules`, () => {
      const content = readFileSync(join(WEB_SRC_DIR, file), "utf-8");

      for (const nodeModule of nodeModules) {
        expect(content).not.toContain(nodeModule);
      }
    });
  }
});

describe("Shared modules reused by web", () => {
  test("web index re-exports from parent module", () => {
    const content = readFileSync(join(WEB_SRC_DIR, "index.ts"), "utf-8");
    expect(content).toContain("../uniface.service.js");
    expect(content).toContain("../constant.js");
  });

  test("web platform imports from core platform", () => {
    const content = readFileSync(
      join(WEB_SRC_DIR, "platform.web.ts"),
      "utf-8"
    );
    expect(content).toContain("../core/platform.js");
  });

  test("core platform file should not import Node-specific modules", () => {
    const content = readFileSync(
      join(SRC_DIR, "core/platform.ts"),
      "utf-8"
    );

    const nodeModules = [
      'from "fs"',
      'from "path"',
      'from "os"',
      'from "onnxruntime-node"',
    ];

    for (const nodeModule of nodeModules) {
      expect(content).not.toContain(nodeModule);
    }
  });

  test("constant.ts should not import Node-specific modules", () => {
    const content = readFileSync(join(SRC_DIR, "constant.ts"), "utf-8");
    const nodeModules = ['from "os"', 'from "path"', 'from "node:os"', 'from "node:path"'];

    for (const nodeModule of nodeModules) {
      expect(content).not.toContain(nodeModule);
    }
  });
});
