/**
 * @module ppu-uniface/web
 *
 * Browser/web entrypoint for ppu-uniface. Uses onnxruntime-web for
 * ONNX inference in the browser. No Node.js, fs, path, or os dependencies.
 *
 * @example
 * ```html
 * <script type="importmap">
 * {
 *   "imports": {
 *     "onnxruntime-web": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/ort.all.bundle.min.mjs",
 *     "ppu-ocv/web": "https://cdn.jsdelivr.net/npm/ppu-ocv@2/index.web.js"
 *   }
 * }
 * </script>
 * <script type="module">
 *   import { Uniface } from "https://cdn.jsdelivr.net/npm/ppu-uniface@3/web/index.js";
 *
 *   const uniface = new Uniface();
 *   await uniface.initialize();
 *
 *   // Use with an image ArrayBuffer from fetch, FileReader, etc.
 *   const result = await uniface.detect(imageArrayBuffer);
 * </script>
 * ```
 */

import { SpoofingDetection } from "../analysis/spoofing.ana.js";
import { RetinaNetDetection } from "../detection/retinanet.det.js";
import { FaceNet512Recognition } from "../recognition/facenet512.rec.js";
import {
  Uniface as BaseUniface,
  type UnifaceOptions,
} from "../uniface.service.js";
import { CosineVerification } from "../verification/cosine.ver.js";
import { WebPlatformProvider, defaultWebPlatform } from "./platform.web.js";

// Re-export shared types and interfaces
export type {
  UnifaceCompactResult,
  UnifaceFullResult,
  UnifaceVerifyOptions,
} from "../uniface.interface.js";

export type {
  DetectOptions,
  DetectionModelOptions,
  DetectionResult,
} from "../detection/base.interface.js";

export type {
  RecognitionModelOptions,
  RecognitionResult,
} from "../recognition/base.interface.js";

export type {
  VerificationModelOptions,
  VerificationResult,
} from "../verification/base.interface.js";

export type {
  SpoofingOptions,
  SpoofingResult,
} from "../analysis/spoofing.ana.js";

export type { CoreCanvas, PlatformProvider } from "../core/platform.js";
export type { UnifaceOptions } from "../uniface.service.js";

export { alignAndCropFace, alignFace, cropFace } from "../alignment.face.js";
export { GITHUB_BASE_URL } from "../constant.js";
export { LoggerConfig, logger } from "../logger.js";
export { WebPlatformProvider, defaultWebPlatform };

/**
 * Web-specific Uniface service that automatically uses WebPlatformProvider.
 * Drop-in replacement for the Node.js Uniface class in browser environments.
 */
export class Uniface extends BaseUniface {
  constructor(options: UnifaceOptions = {}) {
    super(options, defaultWebPlatform);
  }
}

// Export individual web service constructors for advanced usage
export {
  CosineVerification,
  FaceNet512Recognition,
  RetinaNetDetection,
  SpoofingDetection,
};
