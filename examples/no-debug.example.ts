import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { Uniface } from "../src";

const __dirname = dirname(fileURLToPath(import.meta.url));

const file1 = readFileSync(join(__dirname, "../assets/image-kevin1.png"));
const buffer = file1.buffer.slice(
  file1.byteOffset,
  file1.byteOffset + file1.byteLength
);

const file2 = readFileSync(join(__dirname, "../assets/image-kevin2.jpg"));
const buffer2 = file2.buffer.slice(
  file2.byteOffset,
  file2.byteOffset + file2.byteLength
);

console.log("Warming up...");
const uniface = new Uniface();
await uniface.initialize();
// for fair measurement
await uniface.verify(buffer, buffer2);

console.log("Processing...");
const startTime = Date.now();
const result = await uniface.verify(buffer, buffer2);
console.log(`Completed in ${Date.now() - startTime}ms`);

console.log(result);
await uniface.destroy();
