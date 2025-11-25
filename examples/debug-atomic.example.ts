import { CanvasToolkit, ImageProcessor } from "ppu-ocv";
import { CosineVerification, FaceNet512Recognition } from "../src";
import { alignAndCropFace } from "../src/alignment.face";
import { RetinaNetDetection } from "../src/detection/retinanet.det";
import { LoggerConfig } from "../src/logger";

LoggerConfig.verbose = true;

const buffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
const buffer2 = await Bun.file("assets/image-haaland2.png").arrayBuffer();

const retina = new RetinaNetDetection();
await retina.initialize();

const facenet = new FaceNet512Recognition();
await facenet.initialize();

const toolkit = CanvasToolkit.getInstance();
const cosine = new CosineVerification();
toolkit.clearOutput();

async function analyze(buffer: ArrayBuffer) {
  await retina.detect(buffer);

  const startTime = Date.now();
  const result = await retina.detect(buffer);
  console.log(`Completed in: ${Date.now() - startTime}ms`);

  console.log(result);

  const originalCanvas = await ImageProcessor.prepareCanvas(buffer);
  const canvas = await ImageProcessor.prepareCanvas(buffer);
  const ctx = canvas.getContext("2d");

  toolkit.drawLine({
    ctx,
    x: result?.box.x!,
    y: result?.box.y!,
    width: result?.box.width!,
    height: result?.box.height!,
  });

  for (const landmark of result?.landmarks!) {
    toolkit.drawLine({
      ctx,
      x: landmark[0]!,
      y: landmark[1]!,
      width: 1,
      height: 1,
      color: "red",
    });
  }

  await toolkit.saveImage({
    canvas,
    filename: "detection" + Date.now(),
    path: "out",
  });

  const aligned = await alignAndCropFace(originalCanvas, result!);
  await toolkit.saveImage({
    canvas: aligned,
    filename: "crop_aligned" + Date.now(),
    path: "out",
  });

  const embedding = await facenet.recognize(aligned);
  return embedding;
}

const face1 = await analyze(buffer);
const face2 = await analyze(buffer2);

const result = cosine.compare(face1.embedding, face2.embedding);
console.log("Result: ", result);
