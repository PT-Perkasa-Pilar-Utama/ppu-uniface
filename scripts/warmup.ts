import { Uniface } from "../src";

export async function warmup(): Promise<void> {
  const uniface = new Uniface();
  await uniface.initialize();
  console.log("warm up model complete");
}
warmup();
