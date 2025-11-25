import * as ort from "onnxruntime-node";

type BufferType =
  | "uint8"
  | "uint16"
  | "uint32"
  | "int8"
  | "int16"
  | "int32"
  | "float32"
  | "float64";

const BUFFER_CONSTRUCTORS = {
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
  int8: Int8Array,
  int16: Int16Array,
  int32: Int32Array,
  float32: Float32Array,
  float64: Float64Array,
} as const;

async function loadOnnxModel(
  modelPath: string,
  bufferType: BufferType = "uint8"
) {
  const model = Bun.file(modelPath);
  const buffer = await model.arrayBuffer();

  const session =
    bufferType === "uint8"
      ? await ort.InferenceSession.create(new Uint8Array(buffer))
      : await ort.InferenceSession.create(buffer);

  console.log({
    inputNames: session.inputNames,
    outputNames: session.outputNames,
    inputMetadata: JSON.stringify(session.inputMetadata),
    outputMetadata: JSON.stringify(session.outputMetadata),
  });

  await session.release();
}

// Usage:
loadOnnxModel("models/attribute/landmark/2d106det.onnx");

// facenet512.onnx
// {
//   inputNames: [ "input" ],
//   outputNames: [ "Bottleneck_BatchNorm" ],
//   inputMetadata: "[{\"name\":\"input\",\"isTensor\":true,\"type\":\"float32\",\"shape\":[1,160,160,3]}]",
//   outputMetadata: "[{\"name\":\"Bottleneck_BatchNorm\",\"isTensor\":true,\"type\":\"float32\",\"shape\":[1,512]}]",
// }

// retinaface_mv2.onnx
// {
//   inputNames: [ "input" ],
//   outputNames: [ "loc", "conf", "landmarks" ],
//   inputMetadata: "[{\"name\":\"input\",\"isTensor\":true,\"type\":\"float32\",\"shape\":[\"batch_size\",3,\"height\",\"width\"]}]",
//   outputMetadata: "[{\"name\":\"loc\",\"isTensor\":true,\"type\":\"float32\",\"shape\":[\"batch_size\",\"Concatloc_dim_1\",4]},{\"name\":\"conf\",\"isTensor\":true,\"type\":\"float32\",\"shape\":[\"batch_size\",\"Softmaxconf_dim_1\",2]},{\"name\":\"landmarks\",\"isTensor\":true,\"type\":\"float32\",\"shape\":[\"batch_size\",\"Concatlandmarks_dim_1\",10]}]",
// }

// 2d106det.onnx
// {
//   inputNames: [ "data" ],
//   outputNames: [ "fc1" ],
//   inputMetadata: "[{\"name\":\"data\",\"isTensor\":true,\"type\":\"float32\",\"shape\":[\"None\",3,192,192]}]",
//   outputMetadata: "[{\"name\":\"fc1\",\"isTensor\":true,\"type\":\"float32\",\"shape\":[1,212]}]",
// }
