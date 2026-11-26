import fs from 'node:fs';
import path from 'node:path';
import Papa from 'papaparse';
import sharp from 'sharp';
import * as ort from 'onnxruntime-node';

// --- 設定 ---
const CSV_SOURCES = [
  { csvPath: 'public/images_a_0_0.csv', imagesDir: 'public/images/left' },
  { csvPath: 'public/images_a_50.csv', imagesDir: 'public/images/output(a_50)' },
  { csvPath: 'public/images_a_100.csv', imagesDir: 'public/images/output' },
  { csvPath: 'public/images_20261119.csv', imagesDir: 'public/images02/2_flower_1_right' },
];
const MODEL_PATH = path.resolve('public/models/yolo26x.onnx');
const OUTPUT_PATH = path.resolve('public/detections.json');

const YOLO_INPUT_SIZE = 1280; // エクスポート時の imgsz と合わせる（640の場合は640に変更）
const CONFIDENCE_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;

// 誤検出防止フィルタ（COCO事前学習モデル使用時）
const ALLOWED_TOMATO_CLASS_IDS = new Set([47, 49, 32]); // 47: apple, 49: orange, 32: sports ball
const MIN_ASPECT_RATIO = 0.60;
const MAX_ASPECT_RATIO = 1.65;
const MIN_BOX_AREA_RATIO = 0.0005;
const MAX_BOX_AREA_RATIO = 0.18;

const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
];

function computeIoU(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - inter;
  return union <= 0 ? 0 : inter / union;
}

function applyNMS(boxes, iouThreshold) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const selected = [];
  for (const candidate of sorted) {
    let keep = true;
    for (const chosen of selected) {
      if (computeIoU(candidate, chosen) > iouThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) selected.push(candidate);
  }
  return selected;
}

async function preprocessImage(imagePath) {
  const image = sharp(imagePath);
  const metadata = await image.metadata();
  const origW = metadata.width;
  const origH = metadata.height;

  const scale = Math.min(YOLO_INPUT_SIZE / origW, YOLO_INPUT_SIZE / origH);
  const newW = Math.round(origW * scale);
  const newH = Math.round(origH * scale);
  const padX = Math.floor((YOLO_INPUT_SIZE - newW) / 2);
  const padY = Math.floor((YOLO_INPUT_SIZE - newH) / 2);

  const { data } = await image
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

  const pixelCount = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;
  const float32Data = new Float32Array(3 * pixelCount);

  for (let i = 0; i < pixelCount; i++) {
    float32Data[i] = data[i * 3] / 255.0;
    float32Data[pixelCount + i] = data[i * 3 + 1] / 255.0;
    float32Data[2 * pixelCount + i] = data[i * 3 + 2] / 255.0;
  }

  return {
    tensor: new ort.Tensor('float32', float32Data, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]),
    origW,
    origH,
    scale,
    padX,
    padY,
  };
}

function parseYoloOutput(outputTensor, origW, origH, scale, padX, padY) {
  const dims = outputTensor.dims;
  const data = outputTensor.data;
  const rawBoxes = [];

  if (dims.length === 3 && dims[2] === 6) {
    const numDetections = dims[1];
    for (let i = 0; i < numDetections; i++) {
      const offset = i * 6;
      const x1 = data[offset];
      const y1 = data[offset + 1];
      const x2 = data[offset + 2];
      const y2 = data[offset + 3];
      const score = data[offset + 4];
      const classId = Math.round(data[offset + 5]);

      if (score < CONFIDENCE_THRESHOLD) continue;
      if (!ALLOWED_TOMATO_CLASS_IDS.has(classId)) continue;

      const rawLabel = COCO_CLASSES[classId] || `class_${classId}`;
      rawBoxes.push({ x1, y1, x2, y2, score, classId, label: 'tomato', rawLabel });
    }
  } else if (dims.length === 3) {
    const numChannels = dims[1];
    const numAnchors = dims[2];
    const numClasses = numChannels - 4;
    const isCustomSingleClass = numClasses === 1;

    for (let i = 0; i < numAnchors; i++) {
      let maxScore = 0;
      let bestClassId = 0;
      for (let c = 0; c < numClasses; c++) {
        const score = data[(4 + c) * numAnchors + i];
        if (score > maxScore) {
          maxScore = score;
          bestClassId = c;
        }
      }

      if (maxScore < CONFIDENCE_THRESHOLD) continue;
      if (!isCustomSingleClass && !ALLOWED_TOMATO_CLASS_IDS.has(bestClassId)) continue;

      const cx = data[0 * numAnchors + i];
      const cy = data[1 * numAnchors + i];
      const w = data[2 * numAnchors + i];
      const h = data[3 * numAnchors + i];

      const rawLabel = isCustomSingleClass
        ? 'tomato'
        : COCO_CLASSES[bestClassId] || `class_${bestClassId}`;

      rawBoxes.push({
        x1: cx - w / 2,
        y1: cy - h / 2,
        x2: cx + w / 2,
        y2: cy + h / 2,
        score: maxScore,
        classId: bestClassId,
        label: 'tomato',
        rawLabel,
      });
    }
  }

  const normalized = [];
  for (const box of rawBoxes) {
    const x1 = Math.max(0, Math.min(origW, (box.x1 - padX) / scale)) / origW;
    const y1 = Math.max(0, Math.min(origH, (box.y1 - padY) / scale)) / origH;
    const x2 = Math.max(0, Math.min(origW, (box.x2 - padX) / scale)) / origW;
    const y2 = Math.max(0, Math.min(origH, (box.y2 - padY) / scale)) / origH;

    const boxW = x2 - x1;
    const boxH = y2 - y1;
    if (boxW <= 0.005 || boxH <= 0.005) continue;

    // 縦横比と面積による形状フィルタ
    const pixelW = boxW * origW;
    const pixelH = boxH * origH;
    const aspectRatio = pixelW / pixelH;
    const areaRatio = boxW * boxH;

    if (aspectRatio < MIN_ASPECT_RATIO || aspectRatio > MAX_ASPECT_RATIO) continue;
    if (areaRatio < MIN_BOX_AREA_RATIO || areaRatio > MAX_BOX_AREA_RATIO) continue;

    normalized.push({
      x1: Number(x1.toFixed(4)),
      y1: Number(y1.toFixed(4)),
      x2: Number(x2.toFixed(4)),
      y2: Number(y2.toFixed(4)),
      score: Number(box.score.toFixed(4)),
      classId: box.classId,
      label: box.label,
      rawLabel: box.rawLabel,
    });
  }

  return applyNMS(normalized, IOU_THRESHOLD);
}

function saveProgressAtomic(state) {
  const tempPath = `${OUTPUT_PATH}.tmp`;
  fs.writeFileSync(tempPath, JSON.stringify(state, null, 2), 'utf-8');
  fs.renameSync(tempPath, OUTPUT_PATH);
}

async function main() {
  const logs = [];
  const addLog = (msg) => {
    const line = `[${new Date().toLocaleTimeString('ja-JP')}] ${msg}`;
    console.log(line);
    logs.push(line);
  };

  addLog('CSVファイルを読み込み中...');
  const targetMap = new Map(); // filename -> { imagePath, csvNames: Set<string> }
  const csvStats = new Map(); // csvName -> { totalImages: number, detectedImages: number, totalBoxes: number }

  for (const source of CSV_SOURCES) {
    const csvName = path.basename(source.csvPath);
    const fullCsvPath = path.resolve(source.csvPath);
    if (!fs.existsSync(fullCsvPath)) {
      addLog(`警告: ${source.csvPath} が見つかりません`);
      continue;
    }

    const csvText = fs.readFileSync(fullCsvPath, 'utf-8');
    const parsed = Papa.parse(csvText, { header: true, skipEmptyLines: true });
    const uniqueInCsv = new Set();

    for (const row of parsed.data) {
      const name = row.filename?.trim();
      if (!name) continue;
      uniqueInCsv.add(name);

      if (!targetMap.has(name)) {
        targetMap.set(name, {
          imagePath: path.resolve(source.imagesDir, name),
          csvNames: new Set([csvName]),
        });
      } else {
        targetMap.get(name).csvNames.add(csvName);
      }
    }

    csvStats.set(csvName, {
      totalImages: uniqueInCsv.size,
      detectedImages: 0,
      totalBoxes: 0,
    });
  }

  const targets = Array.from(targetMap.entries());

  const state = {
    status: 'running',
    startedAt: new Date().toISOString(),
    completedAt: null,
    totalImages: targets.length,
    processedImages: 0,
    detectedImagesCount: 0,
    totalBoxesCount: 0,
    logs,
    detections: {},
  };

  saveProgressAtomic(state);

  if (!fs.existsSync(MODEL_PATH)) {
    addLog(`エラー: モデルファイルが見つかりません (${MODEL_PATH})`);
    state.status = 'error';
    saveProgressAtomic(state);
    process.exit(1);
  }

  addLog(`対象画像数: ${targets.length} 枚。${path.basename(MODEL_PATH)} モデルをロード中...`);
  saveProgressAtomic(state);

  const session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['cpu'],
  });
  const inputName = session.inputNames[0];

  addLog('モデルロード完了。推論を開始します...');
  saveProgressAtomic(state);

  for (let i = 0; i < targets.length; i++) {
    const [filename, info] = targets[i];
    const csvTag = Array.from(info.csvNames).join(', ');

    if (!fs.existsSync(info.imagePath)) {
      addLog(`[${i + 1}/${targets.length}] [${csvTag}] スキップ (画像なし): ${filename}`);
      state.processedImages = i + 1;
      saveProgressAtomic(state);
      continue;
    }

    const startTime = Date.now();
    const { tensor, origW, origH, scale, padX, padY } = await preprocessImage(info.imagePath);
    const results = await session.run({ [inputName]: tensor });
    const outputTensor = results[session.outputNames[0]];
    const boxes = parseYoloOutput(outputTensor, origW, origH, scale, padX, padY);
    const elapsed = Date.now() - startTime;

    state.detections[filename] = boxes;
    state.processedImages = i + 1;

    if (boxes.length > 0) {
      state.detectedImagesCount += 1;
      state.totalBoxesCount += boxes.length;

      for (const csvName of info.csvNames) {
        const stat = csvStats.get(csvName);
        if (stat) {
          stat.detectedImages += 1;
          stat.totalBoxes += boxes.length;
        }
      }
    }

    addLog(`[${i + 1}/${targets.length}] [${csvTag}] ${filename}: ${boxes.length}個検出 (${elapsed}ms)`);
    saveProgressAtomic(state);
  }

  state.status = 'completed';
  state.completedAt = new Date().toISOString();

  addLog('--- CSVファイル別 検出結果集計 ---');
  for (const [csvName, stat] of csvStats.entries()) {
    addLog(`・${csvName}: ${stat.totalBoxes}個検出 (${stat.detectedImages}/${stat.totalImages}枚で反応)`);
  }
  addLog(
    `完了！全体 ${targets.length} 枚中 ${state.detectedImagesCount} 枚から計 ${state.totalBoxesCount} 個検出しました。`
  );
  saveProgressAtomic(state);
}

main().catch((err) => {
  console.error('推論中にエラーが発生しました:', err);
  process.exit(1);
});