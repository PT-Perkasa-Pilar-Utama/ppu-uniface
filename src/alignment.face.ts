import { CanvasToolkit, cv, ImageProcessor, type Canvas } from "ppu-ocv";
import type { DetectionResult } from "./detection/base.interface";

/**
 * Aligns and crops a face from an image based on detection results
 * @param image - Input image as ArrayBuffer or Canvas
 * @param detection - Face detection result containing landmarks and bounding box
 * @returns Cropped and aligned face canvas
 */
export async function alignAndCropFace(
  image: ArrayBuffer | Canvas,
  detection: DetectionResult
): Promise<Canvas> {
  const canvas =
    image instanceof ArrayBuffer
      ? await ImageProcessor.prepareCanvas(image)
      : image;
  
  const { canvas: alignedCanvas, detection: alignedDetection } = alignFace(
    canvas,
    detection
  );
  
  const croppedCanvas = cropFace(alignedCanvas, alignedDetection);
  
  return croppedCanvas;
}

/**
 * Rotates face to align eyes horizontally
 * @param canvas - Input canvas containing the face
 * @param detection - Face detection result with landmarks
 * @returns Aligned canvas and updated detection result
 */
export function alignFace(
  canvas: Canvas,
  detection: DetectionResult
): { canvas: Canvas; detection: DetectionResult } {
  const landmarks = detection.landmarks;
  if (!landmarks || landmarks.length < 2) {
    return { canvas, detection };
  }

  const leftEye = landmarks[0]!;
  const rightEye = landmarks[1]!;

  const dx = rightEye[0]! - leftEye[0]!;
  const dy = rightEye[1]! - leftEye[1]!;
  const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
  
  if (Math.abs(angle) < 2) {
    return { canvas, detection };
  }

  const center = { x: canvas.width / 2, y: canvas.height / 2 };

  const processor = new ImageProcessor(canvas);
  const rotatedCanvas = processor
    .rotate({
      angle: angle,
      center: new cv.Point(center.x, center.y),
    })
    .toCanvas();
  processor.destroy();

  const box = detection.box;
  const corners = [
    { x: box.x, y: box.y },
    { x: box.x + box.width, y: box.y },
    { x: box.x, y: box.y + box.height },
    { x: box.x + box.width, y: box.y + box.height },
  ];

  const rotatedCorners = corners.map((p) => rotatePoint(p, center, angle));

  const xs = rotatedCorners.map((p) => p.x);
  const ys = rotatedCorners.map((p) => p.y);

  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const maxX = Math.max(...xs);
  const maxY = Math.max(...ys);

  const newBox = {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  };

  const newLandmarks = landmarks.map((l) => {
    const p = rotatePoint({ x: l[0]!, y: l[1]! }, center, angle);
    return [p.x, p.y];
  });

  return {
    canvas: rotatedCanvas,
    detection: {
      ...detection,
      box: newBox,
      landmarks: newLandmarks,
    },
  };
}

/**
 * Rotates a point around a center by a given angle
 * @param point - Point to rotate
 * @param center - Center of rotation
 * @param angle - Rotation angle in degrees
 * @returns Rotated point coordinates
 */
function rotatePoint(
  point: { x: number; y: number },
  center: { x: number; y: number },
  angle: number
) {
  const radians = (-angle * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);

  const dx = point.x - center.x;
  const dy = point.y - center.y;

  const rotated = {
    x: center.x + (dx * cos - dy * sin),
    y: center.y + (dx * sin + dy * cos),
  };
  
  return rotated;
}

/**
 * Crops face region from canvas based on bounding box
 * @param canvas - Input canvas
 * @param detection - Detection result with bounding box
 * @returns Cropped face canvas
 */
export function cropFace(canvas: Canvas, detection: DetectionResult): Canvas {
  const toolkit = CanvasToolkit.getInstance();
  const box = detection.box;

  const x0 = Math.max(0, box.x);
  const y0 = Math.max(0, box.y);
  const x1 = Math.min(canvas.width, box.x + box.width);
  const y1 = Math.min(canvas.height, box.y + box.height);

  const croppedCanvas = toolkit.crop({
    canvas,
    bbox: { x0, y0, x1, y1 },
  });
  
  return croppedCanvas;
}
