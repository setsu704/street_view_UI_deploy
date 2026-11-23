import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';
import * as ort from 'onnxruntime-node';
import sharp from 'sharp';

// --- 設定 ---
const CSV_PATH = path.resolve('public/images_a_0_0.csv');
const IMAGES_DIR = path.resolve('public/images/left');
const MODEL_PATH = path.resolve('public/models/yolo26m.onnx');
const OUTPUT_JSON_PATH = path.resolve('public/detections.json');

const YOLO_INPUT_SIZE = 640;
const MIN_STORE_SCORE = 0.15;
const NMS_IOU_THRESHOLD = 0.45;

// 誤検出防止用フィルタ設定
const MIN_ASPECT_RATIO = 0.60; // 幅/高さ の最小値（細長い茎・支柱を除外）
const MAX_ASPECT_RATIO = 1.65; // 幅/高さ の最大値
const MIN_BOX_AREA = 0.0005;   // 画像全体に対する最小面積比（0.05%未満のノイズを除外）
const MAX_BOX_AREA = 0.35;     // 画像全体に対する最大面積比（35%超の背景誤認を除外）

// COCO 80クラス名リスト
const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush',
];

// トマトとして採用するCOCOクラスID（32: sports ball, 47: apple, 49: orange）
// ※葉っぱ(58: potted plant, 50: broccoli)や人工物を除外
const TARGET_COCO_CLASS_IDS = new Set([32, 47, 49]);

// バウンディングボックスの形状・面積チェック
function isValidTomatoShape(box) {
  const w = box.x2 - box.x1;
  const h = box.y2 - box.y1;
  if (w <= 0 || h <= 0) return false;

  const aspect = w / h;
  const area = w * h;

  return (
    aspect >= MIN_ASPECT_RATIO &&
    aspect <= MAX_ASPECT_RATIO &&
    area >= MIN_BOX_AREA &&
    area <= MAX_BOX_AREA
  );
}

// NMS（重複枠の統合）
function applyNMS(boxes, iouThreshold) {
  if (boxes.length === 0) return [];
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const selected = [];

  while (sorted.length > 0) {
    const current = sorted.shift();
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

// 1枚の画像を推論
async function detectImage(session, imagePath) {
  const image = sharp(imagePath);
  const metadata = await image.metadata();
  const origW = metadata.width;
  const origH = metadata.height;

  if (!origW || !origH) return [];

  const scale = Math.min(YOLO_INPUT_SIZE / origW, YOLO_INPUT_SIZE / origH);
  const newW = Math.round(origW * scale);
  const newH = Math.round(origH * scale);
  const padX = Math.floor((YOLO_INPUT_SIZE - newW) / 2);
  const padY = Math.floor((YOLO_INPUT_SIZE - newH) / 2);

  const { data: rawBuffer } = await image
    .resize(newW, newH)
    .extend({
      top: padY,
      bottom: YOLO_INPUT_SIZE - newH - padY,
      left: padX,
      right: YOLO_INPUT_SIZE - newW - padX,
      background: { r: 114, g: 114, b: 114 },
    })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const stride = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;
  const float32Data = new Float32Array(3 * stride);
  for (let i = 0; i < stride; i++) {
    float32Data[i] = rawBuffer[i * 3] / 255.0;
    float32Data[i + stride] = rawBuffer[i * 3 + 1] / 255.0;
    float32Data[i + stride * 2] = rawBuffer[i * 3 + 2] / 255.0;
  }

  const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]);
  const results = await session.run({ [session.inputNames[0]]: inputTensor });
  const output = results[session.outputNames[0]];
  const data = output.data;
  const dims = output.dims;

  const rawBoxes = [];
  const toNormalizedBox = (bx1, by1, bx2, by2, score, rawLabel) => ({
    x1: Math.max(0, Math.min(1, (bx1 - padX) / newW)),
    y1: Math.max(0, Math.min(1, (by1 - padY) / newH)),
    x2: Math.max(0, Math.min(1, (bx2 - padX) / newW)),
    y2: Math.max(0, Math.min(1, (by2 - padY) / newH)),
    score: Number(score.toFixed(4)),
    label: 'tomato',
    rawLabel,
  });

  if (dims.length === 3 && dims[2] === 6) {
    // End-to-End 形式 [1, N, 6] (x1, y1, x2, y2, score, classId)
    const numDetections = dims[1];
    for (let i = 0; i < numDetections; i++) {
      const offset = i * 6;
      const score = data[offset + 4];
      const classId = Math.round(data[offset + 5]);

      if (score >= MIN_STORE_SCORE && TARGET_COCO_CLASS_IDS.has(classId)) {
        const box = toNormalizedBox(
          data[offset],
          data[offset + 1],
          data[offset + 2],
          data[offset + 3],
          score,
          COCO_CLASSES[classId] || `class_${classId}`
        );
        if (isValidTomatoShape(box)) rawBoxes.push(box);
      }
    }
  } else if (dims.length === 3) {
    // 標準 YOLO 出力形式 [1, 4 + numClasses, numAnchors]
    const numChannels = dims[1];
    const numAnchors = dims[2];
    const numClasses = numChannels - 4;
    const isCustomSingleClass = numClasses === 1;

    for (let i = 0; i < numAnchors; i++) {
      let maxScore = 0;
      let bestClassId = -1;

      for (let c = 0; c < numClasses; c++) {
        // COCO 80クラスモデルの場合は対象クラスのみ評価（独自1クラスモデルなら全て評価）
        if (!isCustomSingleClass && !TARGET_COCO_CLASS_IDS.has(c)) continue;

        const s = data[(c + 4) * numAnchors + i];
        if (s > maxScore) {
          maxScore = s;
          bestClassId = c;
        }
      }

      if (maxScore >= MIN_STORE_SCORE && bestClassId !== -1) {
        const cx = data[0 * numAnchors + i];
        const cy = data[1 * numAnchors + i];
        const w = data[2 * numAnchors + i];
        const h = data[3 * numAnchors + i];

        const rawLabel = isCustomSingleClass
          ? 'tomato'
          : COCO_CLASSES[bestClassId] || `class_${bestClassId}`;

        const box = toNormalizedBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, maxScore, rawLabel);
        if (isValidTomatoShape(box)) rawBoxes.push(box);
      }
    }
  }

  return applyNMS(rawBoxes, NMS_IOU_THRESHOLD);
}

// 途中経過をJSONファイルへ逐次書き出しする関数
function saveProgressJson(payload) {
  const tempPath = `${OUTPUT_JSON_PATH}.tmp`;
  fs.writeFileSync(tempPath, JSON.stringify(payload, null, 2), 'utf-8');
  fs.renameSync(tempPath, OUTPUT_JSON_PATH);
}

// メイン処理
async function main() {
  const logs = [];
  const addLog = (msg) => {
    const time = new Date().toLocaleTimeString('ja-JP');
    const line = `[${time}] ${msg}`;
    console.log(line);
    logs.push(line);
    if (logs.length > 50) logs.shift();
  };

  addLog('CSVファイルを読み込み中...');
  const csvText = fs.readFileSync(CSV_PATH, 'utf-8');
  const parsed = Papa.parse(csvText, { header: true, skipEmptyLines: true });
  const filenames = parsed.data.map((row) => row.filename?.trim()).filter(Boolean);

  addLog(`対象画像数: ${filenames.length} 枚。YOLO26m モデルをロード中...`);

  const detectionsMap = {};
  let detectedImagesCount = 0;
  let totalBoxesCount = 0;

  saveProgressJson({
    status: 'running',
    generatedAt: new Date().toISOString(),
    totalImages: filenames.length,
    processedImages: 0,
    detectedImagesCount: 0,
    totalBoxesCount: 0,
    logs: [...logs],
    detections: detectionsMap,
  });

  const session = await ort.InferenceSession.create(MODEL_PATH);
  addLog('モデルロード完了。推論を開始します...');

  for (let i = 0; i < filenames.length; i++) {
    const filename = filenames[i];
    const imagePath = path.join(IMAGES_DIR, filename);

    if (!fs.existsSync(imagePath)) {
      addLog(`[${i + 1}/${filenames.length}] スキップ (ファイルなし): ${filename}`);
      detectionsMap[filename] = [];
    } else {
      const start = performance.now();
      const boxes = await detectImage(session, imagePath);
      const elapsed = Math.round(performance.now() - start);

      detectionsMap[filename] = boxes;
      if (boxes.length > 0) detectedImagesCount++;
      totalBoxesCount += boxes.length;

      const summary = boxes.map((b) => `${b.rawLabel}:${Math.round(b.score * 100)}%`).join(', ');
      addLog(
        `[${i + 1}/${filenames.length}] ${filename}: ${boxes.length}個検出 (${elapsed}ms)${
          summary ? ` [${summary}]` : ''
        }`
      );
    }

    saveProgressJson({
      status: 'running',
      generatedAt: new Date().toISOString(),
      totalImages: filenames.length,
      processedImages: i + 1,
      detectedImagesCount,
      totalBoxesCount,
      logs: [...logs],
      detections: detectionsMap,
    });
  }

  addLog(`全処理完了! ${detectedImagesCount}/${filenames.length} 枚で検出（合計 ${totalBoxesCount} 個）`);

  saveProgressJson({
    status: 'completed',
    generatedAt: new Date().toISOString(),
    totalImages: filenames.length,
    processedImages: filenames.length,
    detectedImagesCount,
    totalBoxesCount,
    logs: [...logs],
    detections: detectionsMap,
  });

  console.log('--------------------------------------------------');
  console.log(`保存完了: ${OUTPUT_JSON_PATH}`);
}

main().catch(console.error);