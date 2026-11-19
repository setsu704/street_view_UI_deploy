import { useState, useEffect, useMemo } from 'react';
import Papa, { type ParseResult } from 'papaparse';
import type { ImageItem, LogEntry } from './types/panorama';
import { useYoloDetector } from './hooks/useYoloDetector';
import { PanoramaView, IMAGE_WIDTH } from './components/PanoramaView';

// dx, dyから3D空間上の2D累積位置を計算
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
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Papa.parse<LogEntry>('/images_a_0_0.csv', {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (results: ParseResult<LogEntry>) => {
        setRawLogs(results.data);
        setLoading(false);
      },
      error: (error: Error) => {
        console.error('CSV読み込みエラー:', error);
        setLoading(false);
      },
    });
  }, []);

  // 1. 座標データの生成
  const images = useMemo(() => processImagesForPanorama(rawLogs, IMAGE_WIDTH), [rawLogs]);

  // 2. YOLO検出プログラムの実行（途中経過と結果をリアルタイム取得）
  const { detectionsMap, progress } = useYoloDetector(images);

  if (loading) {
    return <div style={{ color: '#fff', background: '#000', height: '100vh', padding: '20px' }}>CSVデータを読み込み中...</div>;
  }

  // 3. 3D空間配置プログラムへデータを渡して描画
  return (
    <PanoramaView
      images={images}
      detectionsMap={detectionsMap}
      yoloProgress={progress}
    />
  );
}