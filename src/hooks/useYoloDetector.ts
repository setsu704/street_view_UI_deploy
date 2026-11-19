import { useState, useEffect } from 'react';
import * as ort from 'onnxruntime-web';
import type { DetectionBox, ImageItem, YoloProgress } from '../types/panorama';

const YOLO_MODEL_URL = '/models/yolo26n.onnx';
const YOLO_INPUT_SIZE = 640;
const NMS_IOU_THRESHOLD = 0.45;
const MIN_STORE_SCORE = 0.15;

// 重複するバウンディングボックスを除去するNMS
function applyNMS(boxes: DetectionBox[], iouThreshold: number): DetectionBox[] {
  if (boxes.length === 0) return [];
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const selected: DetectionBox[] = [];

  while (sorted.length > 0) {
    const current = sorted.shift()!;
    selected.push(current);

    for (let i = sorted.length - 1; i >= 0; i--) {
      const other = sorted[i];
      const interX1 = Math.max(current.x1, other.x1);
      const interY1 = Math.max(current.y1, other.y1);
      const interX2 = Math.min(current.x2, other.x2);
      const interY2 = Math.min(current.y2, other.y2);

      const interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
      const areaA = (current.x2 - current.x1) * (current.y2 - current.y1);
      const areaB = (other.x2 - other.x1) * (other.y2 - other.y1);
      const unionArea = areaA + areaB - interArea;

      if (unionArea > 0 && interArea / unionArea > iouThreshold) {
        sorted.splice(i, 1);
      }
    }
  }
  return selected;
}

// 1枚の画像に対する推論処理
async function runSingleDetection(
  session: ort.InferenceSession,
  imageUrl: string
): Promise<DetectionBox[]> {
  return new Promise((resolve) => {
    const img = new window.Image();
    img.onload = async () => {
      try {
        const origW = img.naturalWidth || img.width;
        const origH = img.naturalHeight || img.height;
        if (!origW || !origH) return resolve([]);

        const scale = Math.min(YOLO_INPUT_SIZE / origW, YOLO_INPUT_SIZE / origH);
        const newW = Math.round(origW * scale);
        const newH = Math.round(origH * scale);
        const padX = Math.floor((YOLO_INPUT_SIZE - newW) / 2);
        const padY = Math.floor((YOLO_INPUT_SIZE - newH) / 2);

        const canvas = document.createElement('canvas');
        canvas.width = YOLO_INPUT_SIZE;
        canvas.height = YOLO_INPUT_SIZE;
        const ctx = canvas.getContext('2d');
        if (!ctx) return resolve([]);

        ctx.fillStyle = 'rgb(114, 114, 114)';
        ctx.fillRect(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
        ctx.drawImage(img, padX, padY, newW, newH);

        const imageData = ctx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE).data;
        const float32Data = new Float32Array(3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE);
        const stride = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;

        for (let i = 0; i < stride; i++) {
          float32Data[i] = imageData[i * 4] / 255.0;
          float32Data[i + stride] = imageData[i * 4 + 1] / 255.0;
          float32Data[i + stride * 2] = imageData[i * 4 + 2] / 255.0;
        }

        const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]);
        const feeds: Record<string, ort.Tensor> = { [session.inputNames[0]]: inputTensor };

        const results = await session.run(feeds);
        const output = results[session.outputNames[0]];
        const data = output.data as Float32Array;
        const dims = output.dims;

        const rawBoxes: DetectionBox[] = [];

        const toNormalizedBox = (bx1: number, by1: number, bx2: number, by2: number, score: number): DetectionBox => ({
          x1: Math.max(0, Math.min(1, (bx1 - padX) / newW)),
          y1: Math.max(0, Math.min(1, (by1 - padY) / newH)),
          x2: Math.max(0, Math.min(1, (bx2 - padX) / newW)),
          y2: Math.max(0, Math.min(1, (by2 - padY) / newH)),
          score,
          label: 'tomato',
        });

        if (dims.length === 3 && dims[2] === 6) {
          const numDetections = dims[1];
          for (let i = 0; i < numDetections; i++) {
            const offset = i * 6;
            const score = data[offset + 4];
            if (score >= MIN_STORE_SCORE) {
              rawBoxes.push(
                toNormalizedBox(data[offset], data[offset + 1], data[offset + 2], data[offset + 3], score)
              );
            }
          }
        } else if (dims.length === 3) {
          const numChannels = dims[1];
          const numAnchors = dims[2];
          for (let i = 0; i < numAnchors; i++) {
            let maxScore = 0;
            for (let c = 4; c < numChannels; c++) {
              const s = data[c * numAnchors + i];
              if (s > maxScore) maxScore = s;
            }
            if (maxScore >= MIN_STORE_SCORE) {
              const cx = data[0 * numAnchors + i];
              const cy = data[1 * numAnchors + i];
              const w = data[2 * numAnchors + i];
              const h = data[3 * numAnchors + i];
              rawBoxes.push(
                toNormalizedBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, maxScore)
              );
            }
          }
        }

        resolve(applyNMS(rawBoxes, NMS_IOU_THRESHOLD));
      } catch (err) {
        console.error(`YOLO推論エラー (${imageUrl}):`, err);
        resolve([]);
      }
    };
    img.onerror = (e) => {
      console.error(`YOLO用画像の読み込みに失敗 (${imageUrl}):`, e);
      resolve([]);
    };
    img.src = imageUrl;
  });
}

// 複数画像を順次検出し、途中経過を出力するカスタムフック
export function useYoloDetector(images: ImageItem[]) {
  const [detectionsMap, setDetectionsMap] = useState<Record<string, DetectionBox[]>>({});
  const [progress, setProgress] = useState<YoloProgress>({
    status: 'idle',
    message: '待機中...',
    current: 0,
    total: 0,
    currentFilename: '',
    detectedImagesCount: 0,
    totalBoxesCount: 0,
    logs: [],
  });

  useEffect(() => {
    let isCancelled = false;

    async function detectAll() {
      if (images.length === 0) return;

      const addLog = (text: string) => {
        const timestamp = new Date().toLocaleTimeString('ja-JP');
        const logLine = `[${timestamp}] ${text}`;
        console.log(logLine); // コンソールにも途中経過を出力
        return logLine;
      };

      try {
        const startLog = addLog('YOLO26n モデルの読み込みを開始...');
        setProgress({
          status: 'loading_model',
          message: 'YOLO26n モデル初期化中...',
          current: 0,
          total: images.length,
          currentFilename: '',
          detectedImagesCount: 0,
          totalBoxesCount: 0,
          logs: [startLog],
        });

        const session = await ort.InferenceSession.create(YOLO_MODEL_URL, {
          executionProviders: ['wasm'],
        });

        if (isCancelled) return;
        const loadedLog = addLog(`モデル読込完了。全 ${images.length} 枚の検出を開始します。`);
        setProgress((prev) => ({
          ...prev,
          status: 'detecting',
          message: '検出開始...',
          logs: [...prev.logs, loadedLog],
        }));

        let completed = 0;
        let detectedImagesCount = 0;
        let totalBoxesCount = 0;

        for (const item of images) {
          if (isCancelled) break;

          const startTime = performance.now();
          setProgress((prev) => ({
            ...prev,
            message: `検出中 (${completed + 1}/${images.length}): ${item.filename}`,
            currentFilename: item.filename,
          }));

          const boxes = await runSingleDetection(session, item.url);
          const elapsed = Math.round(performance.now() - startTime);

          if (boxes.length > 0) detectedImagesCount++;
          totalBoxesCount += boxes.length;
          completed++;

          const stepLog = addLog(
            `(${completed}/${images.length}) ${item.filename}: ${boxes.length}個検出 (${elapsed}ms)`
          );

          if (!isCancelled) {
            setDetectionsMap((prev) => ({
              ...prev,
              [item.filename]: boxes,
            }));

            setProgress((prev) => ({
              ...prev,
              current: completed,
              detectedImagesCount,
              totalBoxesCount,
              logs: [...prev.logs.slice(-49), stepLog], // 最新50件のログを保持
            }));
          }
        }

        if (!isCancelled) {
          const finishLog = addLog(
            `全処理完了: ${detectedImagesCount}/${completed}枚で検出（計 ${totalBoxesCount}個）`
          );
          setProgress((prev) => ({
            ...prev,
            status: 'completed',
            message: `YOLO26n 検出完了 (${detectedImagesCount}/${completed}枚で反応 / 計${totalBoxesCount}個)`,
            currentFilename: '',
            logs: [...prev.logs.slice(-49), finishLog],
          }));
        }
      } catch (err) {
        console.error('YOLO26n モデルのロードに失敗しました:', err);
        if (!isCancelled) {
          const errLog = addLog('エラー: モデルの読み込みに失敗しました (/models/yolo26n.onnx を確認)');
          setProgress((prev) => ({
            ...prev,
            status: 'error',
            message: 'YOLO26n モデル読込エラー',
            logs: [...prev.logs, errLog],
          }));
        }
      }
    }

    detectAll();
    return () => {
      isCancelled = true;
    };
  }, [images]);

  return { detectionsMap, progress };
}