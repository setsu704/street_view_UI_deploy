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

// 途中経過を出力・表示するための進捗データ型
export interface YoloProgress {
  status: 'idle' | 'loading_model' | 'detecting' | 'completed' | 'error';
  message: string;
  current: number;
  total: number;
  currentFilename: string;
  detectedImagesCount: number;
  totalBoxesCount: number;
  logs: string[]; // 途中経過のログ履歴
}