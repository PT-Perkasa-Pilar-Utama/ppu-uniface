import * as os from "os";
import * as path from "path";

/** Base URL for downloading model files from GitHub (using LFS media endpoint) */
export const GITHUB_BASE_URL =
  "https://media.githubusercontent.com/media/PT-Perkasa-Pilar-Utama/ppu-uniface/main/models/";

/** Local cache directory for downloaded models */
export const CACHE_DIR: string = path.join(
  os.homedir(),
  ".cache",
  "ppu-uniface"
);
