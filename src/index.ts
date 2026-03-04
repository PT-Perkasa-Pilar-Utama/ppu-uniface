export type {
  UnifaceCompactResult,
  UnifaceFullResult,
  UnifaceVerifyOptions,
} from "./uniface.interface.js";
export { Uniface } from "./uniface.service.js";

export { Base } from "./global.interface.js";
export { LoggerConfig, logger } from "./logger.js";

export { alignAndCropFace, alignFace, cropFace } from "./alignment.face.js";
export { GITHUB_BASE_URL } from "./constant.js";

export type { CoreCanvas, PlatformProvider } from "./core/platform.js";

export {
  BaseDetection,
  type DetectOptions,
  type DetectionModelOptions,
  type DetectionResult,
} from "./detection/base.interface.js";
export { RetinaNetDetection } from "./detection/retinanet.det.js";

export {
  BaseRecognition,
  type RecognitionModelOptions,
  type RecognitionResult,
} from "./recognition/base.interface.js";
export { FaceNet512Recognition } from "./recognition/facenet512.rec.js";

export {
  BaseVerification,
  type VerificationModelOptions,
  type VerificationResult,
} from "./verification/base.interface.js";
export { CosineVerification } from "./verification/cosine.ver.js";

export { BaseAnalysis } from "./analysis/base.interface.js";
export {
  SpoofingDetection,
  type SpoofingOptions,
  type SpoofingResult,
} from "./analysis/spoofing.ana.js";
