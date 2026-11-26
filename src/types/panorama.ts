export type ViewMode = 'plane' | 'cylinder' | 'tunnel';

export interface LogEntry {
  filename: string;
  dx: string | number;
  dy: string | number;
}

export interface ImageItem {
  id: number;
  url: string;
  originalUrl?: string;
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
  rawLabel?: string;
}

export interface DetectionsJson {
  status?: 'running' | 'completed';
  generatedAt: string;
  totalImages: number;
  processedImages: number;
  detectedImagesCount: number;
  totalBoxesCount: number;
  logs: string[];
  detections: Record<string, DetectionBox[]>;
}