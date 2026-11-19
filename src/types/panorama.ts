export type ViewMode = 'plane' | 'cylinder' | 'tunnel';

export interface LogEntry {
  filename: string;
  dx: string | number;
  dy: string | number;
}

export interface ImageItem {
  id: number;
  url: string;
  position: [number, number, number];
  filename: string;
}

export interface DetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  label: string;
}

export interface DetectionsJson {
  generatedAt: string;
  totalImages: number;
  detectedImagesCount: number;
  totalBoxesCount: number;
  detections: Record<string, DetectionBox[]>;
}