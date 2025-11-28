export type {
  UnifaceCompactResult,
  UnifaceFullResult,
  UnifaceVerifyOptions,
} from "./uniface.interface";
export { Uniface } from "./uniface.service";

export { Base } from "./global.interface";
export { logger, LoggerConfig } from "./logger";

export { alignAndCropFace, alignFace, cropFace } from "./alignment.face";
export { CACHE_DIR, GITHUB_BASE_URL } from "./constant";

export {
  BaseDetection,
  type DetectionModelOptions,
  type DetectionResult,
} from "./detection/base.interface";
export { RetinaNetDetection } from "./detection/retinanet.det";

export {
  BaseRecognition,
  type RecognitionModelOptions,
  type RecognitionResult,
} from "./recognition/base.interface";
export { FaceNet512Recognition } from "./recognition/facenet512.rec";

export {
  BaseVerification,
  type VerificationModelOptions,
  type VerificationResult,
} from "./verification/base.interface";
export { CosineVerification } from "./verification/cosine.ver";

export { BaseAnalysis } from "./analysis/base.interface";
export {
  SpoofingDetection,
  type SpoofingOptions,
  type SpoofingResult,
} from "./analysis/spoofing.ana";
