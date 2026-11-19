import { useState, useEffect, useMemo } from 'react';
import Papa, { type ParseResult } from 'papaparse';
import type { DetectionsJson, ImageItem, LogEntry } from './types/panorama';
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
  const [detectionData, setDetectionData] = useState<DetectionsJson | null>(null);
  const [errorText, setErrorText] = useState<string>('');
  const [loading, setLoading] = useState(true);

  // 1. CSVの読み込み
  useEffect(() => {
    Papa.parse<LogEntry>('/images_a_0_0.csv', {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (results: ParseResult<LogEntry>) => {
        setRawLogs(results.data);
        setLoading(false);
      },
      error: () => {
        setLoading(false);
      },
    });
  }, []);

  // 2. detections.json の取得＆実行中のリアルタイム監視（1秒間隔）
  useEffect(() => {
    let isMounted = true;
    let timerId: number | undefined;

    const fetchDetections = async () => {
      try {
        // キャッシュを回避して最新の途中経過を取得
        const res = await fetch(`/detections.json?t=${Date.now()}`);
        if (!res.ok) throw new Error('detections.json が見つかりません');
        const json = (await res.json()) as DetectionsJson;

        if (!isMounted) return;
        setDetectionData(json);
        setErrorText('');

        // まだ実行中（running）なら1秒後に再取得する
        if (json.status !== 'completed') {
          timerId = window.setTimeout(fetchDetections, 1000);
        }
      } catch {
        if (!isMounted) return;
        setErrorText('事前検出データなし (npm run detect を実行してください)');
        // ファイルが作られるのを待つため2秒後に再試行
        timerId = window.setTimeout(fetchDetections, 2000);
      }
    };

    fetchDetections();

    return () => {
      isMounted = false;
      if (timerId) clearTimeout(timerId);
    };
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
      detectionData={detectionData}
      errorText={errorText}
    />
  );
}