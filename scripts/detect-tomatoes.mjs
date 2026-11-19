import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';
import * as ort from 'onnxruntime-node';
import sharp from 'sharp';

// --- 設定 ---
const CSV_PATH = path.resolve('public/images_a_0_0.csv');
const IMAGES_DIR = path.resolve('public/images/left');
const MODEL_PATH = path.resolve('public/models/yolo26n.onnx');
const OUTPUT_JSON_PATH = path.resolve('public/detections.json');

const YOLO_INPUT_SIZE = 640;
const MIN_STORE_SCORE = 0.15; // スライダー絞り込み用に0.15以上を保存
const NMS_IOU_THRESHOLD = 0.45;

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

  // Letterboxリサイズ計算
  const scale = Math.min(YOLO_INPUT_SIZE / origW, YOLO_INPUT_SIZE / origH);
  const newW = Math.round(origW * scale);
  const newH = Math.round(origH * scale);
  const padX = Math.floor((YOLO_INPUT_SIZE - newW) / 2);
  const padY = Math.floor((YOLO_INPUT_SIZE - newH) / 2);

  // 640x640 RGBバッファ生成（余白はYOLO標準の114）
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

  // HWC -> CHW & 0.0-1.0正規化
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
  const toNormalizedBox = (bx1, by1, bx2, by2, score) => ({
    x1: Math.max(0, Math.min(1, (bx1 - padX) / newW)),
    y1: Math.max(0, Math.min(1, (by1 - padY) / newH)),
    x2: Math.max(0, Math.min(1, (bx2 - padX) / newW)),
    y2: Math.max(0, Math.min(1, (by2 - padY) / newH)),
    score: Number(score.toFixed(4)),
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

  return applyNMS(rawBoxes, NMS_IOU_THRESHOLD);
}

// メイン処理
async function main() {
  console.log('1. CSVファイルを読み込み中...');
  const csvText = fs.readFileSync(CSV_PATH, 'utf-8');
  const parsed = Papa.parse(csvText, { header: true, skipEmptyLines: true });
  const filenames = parsed.data
    .map((row) => row.filename?.trim())
    .filter(Boolean);

  console.log(`対象画像数: ${filenames.length} 枚`);
  console.log('2. YOLO26n モデルをロード中...');
  const session = await ort.InferenceSession.create(MODEL_PATH);

  const detectionsMap = {};
  let detectedImagesCount = 0;
  let totalBoxesCount = 0;

  console.log('3. 検出処理を開始します...');
  for (let i = 0; i < filenames.length; i++) {
    const filename = filenames[i];
    const imagePath = path.join(IMAGES_DIR, filename);

    if (!fs.existsSync(imagePath)) {
      console.warn(`[${i + 1}/${filenames.length}] スキップ (ファイルなし): ${filename}`);
      detectionsMap[filename] = [];
      continue;
    }

    const start = performance.now();
    const boxes = await detectImage(session, imagePath);
    const elapsed = Math.round(performance.now() - start);

    detectionsMap[filename] = boxes;
    if (boxes.length > 0) detectedImagesCount++;
    totalBoxesCount += boxes.length;

    // 途中経過の出力
    console.log(
      `[${i + 1}/${filenames.length}] ${filename} -> ${boxes.length}個検出 (${elapsed}ms)`
    );
  }

  // JSON保存
  const outputPayload = {
    generatedAt: new Date().toISOString(),
    totalImages: filenames.length,
    detectedImagesCount,
    totalBoxesCount,
    detections: detectionsMap,
  };

  fs.writeFileSync(OUTPUT_JSON_PATH, JSON.stringify(outputPayload, null, 2), 'utf-8');
  console.log('--------------------------------------------------');
  console.log(`完了! 保存先: ${OUTPUT_JSON_PATH}`);
  console.log(`結果: ${detectedImagesCount}/${filenames.length} 枚で検出（合計 ${totalBoxesCount} 個）`);
}

main().catch(console.error);