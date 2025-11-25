import { bench, group, run } from "mitata";
import { CosineVerification, Uniface } from "../src";

console.log("Initializing FaceService for benchmarks...");
const service = new Uniface();
await service.initialize();

const imgBuf = await Bun.file("../assets/image-kevin1.png").arrayBuffer();

console.log("Warming up models...");
await service.verify(imgBuf, imgBuf);

const emb1 = new Float32Array(512).fill(0.1);
const emb2 = new Float32Array(512).fill(0.1);
const verifier = new CosineVerification();

group("Face Verification Pipeline", () => {
  bench("Full Verify (Same Image)", async () => {
    await service.verify(imgBuf, imgBuf);
  });
});

group("Micro Benchmarks", () => {
  bench("Cosine Distance Calculation", async () => {
    verifier.compare(emb1, emb2);
  });
});

await run();

await service.destroy();
