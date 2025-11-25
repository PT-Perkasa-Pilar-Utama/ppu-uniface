import { Uniface } from "../src";

const buffer = await Bun.file("assets/image-haaland1.jpeg").arrayBuffer();
const buffer2 = await Bun.file("assets/image-haaland2.png").arrayBuffer();

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
