import { Uniface } from "../src";

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
  { name: "image-kevin1.png", path: "assets/image-kevin1.png" },
  { name: "image-kevin2.jpg", path: "assets/image-kevin2.jpg" },
  { name: "image-haaland1.jpeg", path: "assets/image-haaland1.jpeg" },
  { name: "image-haaland2.png", path: "assets/image-haaland2.png" },
];

console.log("Image 1, Image 2, Similarity, Verified, Ground Truth");

const loadedImages = await Promise.all(
  images.map(async (img) => ({
    name: img.name,
    buffer: await Bun.file(img.path).arrayBuffer(),
  }))
);

for (let i = 0; i < loadedImages.length; i++) {
  for (let j = i + 1; j < loadedImages.length; j++) {
    const img1 = loadedImages[i]!;
    const img2 = loadedImages[j]!;

    try {
      const result = await uniFace.verify(img1.buffer, img2.buffer);
      console.log(
        `${img1.name}, ${img2.name}, ${result.similarity}, ${result.verified}, ${GROUND_TRUTH[img1.name + "-" + img2.name]}`
      );
    } catch (error) {
      console.log(`$${img1.name}, ${img2.name}, ERROR, ${error}`);
    }
  }
}
