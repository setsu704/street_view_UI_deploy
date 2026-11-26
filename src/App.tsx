import { useState, useEffect, useMemo } from 'react';
import Papa, { type ParseResult } from 'papaparse';
import type { DetectionsJson, ImageItem, LogEntry } from './types/panorama';
import { PanoramaView, IMAGE_WIDTH } from './components/PanoramaView';

// 選択可能なCSV一覧とそれぞれの画像フォルダ
export const CSV_OPTIONS = [
  { id: 'images_a_0_0.csv', label: '0_0', imageDir: '/images/left' },
  { id: 'images_a_50.csv', label: '50', imageDir: '/images/output(a_50)' },
  { id: 'images_a_100.csv', label: '100', imageDir: '/images/output' },
] as const;

// CSVのファイル名（_a_50.png や _a_100.png など）から left フォルダ内の元画像名（.jpg）を生成
const toOriginalLeftFilename = (filename: string): string => {
  return filename
    .replace(/_a_(0_0|50|100)(?=\.[^.]+$)/i, '')
    .replace(/\.png$/i, '.jpg');
};

const processImagesForPanorama = (
  logs: LogEntry[],
  width: number,
  imageDir: string
): ImageItem[] => {
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
      accumulatedX += stepX;
      accumulatedY += stepY;
    }

    const cleanFilename = log.filename.trim();
    const originalFilename = toOriginalLeftFilename(cleanFilename);

    result.push({
      id: i,
      url: `${imageDir}/${cleanFilename}`,
      originalUrl: `/images/left/${originalFilename}`,
      position: [accumulatedX, accumulatedY, 0],
      filename: cleanFilename,
    });
  }
  return result;
};

export default function App() {
  const [selectedCsv, setSelectedCsv] = useState<string>(CSV_OPTIONS[0].id);
  const [loadedData, setLoadedData] = useState<{ csvId: string; logs: LogEntry[] } | null>(null);
  const [detectionData, setDetectionData] = useState<DetectionsJson | null>(null);
  const [errorText, setErrorText] = useState<string>('');

  // 1. 選択されたCSVの読み込み
  useEffect(() => {
    let isCancelled = false;

    Papa.parse<LogEntry>(`/${selectedCsv}`, {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (results: ParseResult<LogEntry>) => {
        if (!isCancelled) {
          setLoadedData({ csvId: selectedCsv, logs: results.data });
        }
      },
      error: () => {
        if (!isCancelled) {
          setLoadedData({ csvId: selectedCsv, logs: [] });
        }
      },
    });

    return () => {
      isCancelled = true;
    };
  }, [selectedCsv]);

  // 2. detections.json の取得＆実行中のリアルタイム監視（1秒間隔）
  useEffect(() => {
    let isMounted = true;
    let timerId: number | undefined;

    const fetchDetections = async () => {
      try {
        const res = await fetch(`/detections.json?t=${Date.now()}`);
        if (!res.ok) throw new Error('detections.json が見つかりません');
        const json = (await res.json()) as DetectionsJson;

        if (!isMounted) return;
        setDetectionData(json);
        setErrorText('');

        if (json.status !== 'completed') {
          timerId = window.setTimeout(fetchDetections, 1000);
        }
      } catch {
        if (!isMounted) return;
        setErrorText('事前検出データなし (npm run detect を実行してください)');
        timerId = window.setTimeout(fetchDetections, 2000);
      }
    };

    fetchDetections();

    return () => {
      isMounted = false;
      if (timerId) clearTimeout(timerId);
    };
  }, []);

  const images = useMemo(() => {
    if (!loadedData || loadedData.csvId !== selectedCsv) return [];
    const opt = CSV_OPTIONS.find((o) => o.id === selectedCsv) || CSV_OPTIONS[0];
    return processImagesForPanorama(loadedData.logs, IMAGE_WIDTH, opt.imageDir);
  }, [loadedData, selectedCsv]);

  if (!loadedData) {
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
      selectedCsv={selectedCsv}
      onSelectCsv={setSelectedCsv}
    />
  );
}