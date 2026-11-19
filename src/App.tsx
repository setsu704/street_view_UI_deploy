import { useState, useEffect, useMemo } from 'react';
import Papa, { type ParseResult } from 'papaparse';
import type { DetectionBox, DetectionsJson, ImageItem, LogEntry } from './types/panorama';
import { PanoramaView, IMAGE_WIDTH } from './components/PanoramaView';

const processImagesForPanorama = (logs: LogEntry[], width: number): ImageItem[] => {
  const result: ImageItem[] = [];
  let accumulatedX = 0;
  let accumulatedY = 0;

  for (let i = 0; i < logs.length; i++) {
    const log = logs[i];
    if (!log.filename) continue;

    const dxNum = typeof log.dx === 'string' ? parseFloat(log.dx) : log.dx || 0;
    const dyNum = typeof log.dy === 'string' ? parseFloat(log.dy) : log.dy || 0;

    if (i > 0) {
      const stepX = dxNum ? dxNum / 100 : width;
      const stepY = dyNum ? dyNum / 100 : 0;
      accumulatedX -= stepX;
      accumulatedY += stepY;
    }

    result.push({
      id: i,
      url: `/images/left/${log.filename.trim()}`,
      position: [accumulatedX, accumulatedY, 0],
      filename: log.filename.trim(),
    });
  }
  return result;
};

export default function App() {
  const [rawLogs, setRawLogs] = useState<LogEntry[]>([]);
  const [detectionsMap, setDetectionsMap] = useState<Record<string, DetectionBox[]>>({});
  const [detectionStatus, setDetectionStatus] = useState<string>('検出データ読込中...');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 1. CSVの読み込み
    const loadCsv = new Promise<LogEntry[]>((resolve) => {
      Papa.parse<LogEntry>('/images_a_0_0.csv', {
        download: true,
        header: true,
        skipEmptyLines: true,
        complete: (results: ParseResult<LogEntry>) => resolve(results.data),
        error: () => resolve([]),
      });
    });

    // 2. 事前検出済みJSONの読み込み
    const loadDetections = fetch('/detections.json')
      .then((res) => {
        if (!res.ok) throw new Error('detections.json が見つかりません');
        return res.json() as Promise<DetectionsJson>;
      })
      .then((json) => {
        setDetectionsMap(json.detections || {});
        setDetectionStatus(
          `事前検出データ読込済 (${json.detectedImagesCount}/${json.totalImages}枚で反応)`
        );
      })
      .catch((err) => {
        console.warn('検出データの読み込みに失敗:', err);
        setDetectionStatus('事前検出データなし (npm run detect を実行してください)');
      });

    Promise.all([loadCsv, loadDetections]).then(([csvData]) => {
      setRawLogs(csvData);
      setLoading(false);
    });
  }, []);

  const images = useMemo(() => processImagesForPanorama(rawLogs, IMAGE_WIDTH), [rawLogs]);

  if (loading) {
    return (
      <div style={{ color: '#fff', background: '#000', height: '100vh', padding: '20px' }}>
        データを読み込み中...
      </div>
    );
  }

  return (
    <PanoramaView
      images={images}
      detectionsMap={detectionsMap}
      statusText={detectionStatus}
    />
  );
}