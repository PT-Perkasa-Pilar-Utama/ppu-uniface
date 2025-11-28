import { Uniface } from "../src";

import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const GROUND_TRUTH: { [key: string]: boolean } = {
  "image-kevin1.png-image-kevin2.jpg": true,
  "image-kevin1.png-image-haaland1.jpeg": false,
  "image-kevin1.png-image-haaland2.png": false,
  "image-kevin2.jpg-image-haaland1.jpeg": false,
  "image-kevin2.jpg-image-haaland2.png": false,
  "image-haaland1.jpeg-image-haaland2.png": true,
};

const uniFace = new Uniface();
await uniFace.initialize();

const images = [
  { name: "image-kevin1.png", path: "../assets/image-kevin1.png" },
  { name: "image-kevin2.jpg", path: "../assets/image-kevin2.jpg" },
  { name: "image-haaland1.jpeg", path: "../assets/image-haaland1.jpeg" },
  { name: "image-haaland2.png", path: "../assets/image-haaland2.png" },
];

console.log("============= Accuracy Benchmark ============");

const loadedImages = await Promise.all(
  images.map(async (img) => {
    const file = readFileSync(join(__dirname, img.path));
    const buffer = file.buffer.slice(
      file.byteOffset,
      file.byteOffset + file.byteLength,
    );

    return {
      name: img.name,
      buffer,
    };
  }),
);

const rows: {
  image1: string;
  image2: string;
  similarity: number | string;
  verified: boolean | string;
  groundTruth: boolean;
}[] = [];

for (let i = 0; i < loadedImages.length; i++) {
  for (let j = i + 1; j < loadedImages.length; j++) {
    const img1 = loadedImages[i]!;
    const img2 = loadedImages[j]!;

    try {
      const result = await uniFace.verify(img1.buffer, img2.buffer);

      rows.push({
        image1: img1.name,
        image2: img2.name,
        similarity: result.similarity,
        verified: result.verified,
        groundTruth: GROUND_TRUTH[img1.name + "-" + img2.name],
      });
    } catch (error) {
      rows.push({
        image1: img1.name,
        image2: img2.name,
        similarity: "ERROR",
        verified: "ERROR",
        groundTruth: GROUND_TRUTH[img1.name + "-" + img2.name],
      });
    }
  }
}

console.table(rows);
console.log("============================================");
