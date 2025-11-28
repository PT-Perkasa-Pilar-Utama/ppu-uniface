import { readFileSync } from "fs";
import { bench, group, run } from "mitata";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { CosineVerification, Uniface } from "../src";
console.log("========== Mitata Speed Benchmark ===========");
console.log("Initializing Uniface for benchmarks...");
const service = new Uniface();
await service.initialize();

const __dirname = dirname(fileURLToPath(import.meta.url));

const fileKevin1 = readFileSync(join(__dirname, "../assets/image-kevin1.png"));
const imgBuf = fileKevin1.buffer.slice(
  fileKevin1.byteOffset,
  fileKevin1.byteOffset + fileKevin1.byteLength,
);

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
console.log("============================================");
