import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { Uniface } from "../src";

const __dirname = dirname(fileURLToPath(import.meta.url));

const fileKevin1 = readFileSync(join(__dirname, "../assets/image-kevin1.png"));
const kevin1 = fileKevin1.buffer.slice(
  fileKevin1.byteOffset,
  fileKevin1.byteOffset + fileKevin1.byteLength
);

const fileKevin2 = readFileSync(join(__dirname, "../assets/image-kevin2.jpg"));
const kevin2 = fileKevin2.buffer.slice(
  fileKevin2.byteOffset,
  fileKevin2.byteOffset + fileKevin2.byteLength
);

const faceService = new Uniface();

console.log("Warming up...");
await faceService.initialize();
await faceService.verify(kevin1, kevin2);

console.log("Benchmarking...");
let totalDuration = 0;
const iterations = 7;

for (let i = 0; i < iterations; i++) {
  const startTime = Date.now();
  await faceService.verify(kevin1, kevin2);
  const duration = Date.now() - startTime;

  totalDuration += duration;
  console.log(`Iteration ${i + 1}: ${duration}ms`);
}

const averageDuration = totalDuration / iterations;
console.log(
  `Average duration over ${iterations} runs: ${averageDuration.toFixed(2)}ms`
);
