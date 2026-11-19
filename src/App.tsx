import React, { useMemo, useRef, useState, useEffect, Suspense } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { Image as DreiImage, OrbitControls, PerspectiveCamera, Grid, Line, Text } from '@react-three/drei';
import * as THREE from 'three';
import Papa, { type ParseResult } from 'papaparse';
import * as ort from 'onnxruntime-web';

// 時刻表示
const BUILD_TIME = new Date().toLocaleString("ja-JP", {
  timeZone: "Asia/Tokyo",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit"
});

// --- 1. 設定 & 型定義 ---
const IMAGE_WIDTH = 3.2; 
const IMAGE_HEIGHT = 2.4;

const OPACITY_VISIBLE = 0.9;
const OPACITY_INVISIBLE = 0.5;
const CURVE_FACTOR = 0.000015;

// YOLO26n モデル設定
const YOLO_MODEL_URL = '/models/yolo26n.onnx';
const YOLO_INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.45; // 検出信頼度のしきい値

export type ViewMode = 'plane' | 'cylinder' | 'tunnel';

export interface LogEntry {
  filename: string;
  dx: string | number;
  dy: string | number;
}

// 検出されたトマトのバウンディングボックス（0.0〜1.0の正規化座標）
export interface DetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  label: string;
}

// --- 2. YOLO26n (ONNX) 推論ユーティリティ ---
async function runYolo26Detection(
  session: ort.InferenceSession,
  imageUrl: string
): Promise<DetectionBox[]> {
  return new Promise((resolve) => {
    const img = new window.Image();
    img.crossOrigin = 'Anonymous';
    img.onload = async () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = YOLO_INPUT_SIZE;
        canvas.height = YOLO_INPUT_SIZE;
        const ctx = canvas.getContext('2d');
        if (!ctx) return resolve([]);

        ctx.drawImage(img, 0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
        const imageData = ctx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE).data;

        const float32Data = new Float32Array(3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE);
        const stride = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;
        for (let i = 0; i < stride; i++) {
          float32Data[i] = imageData[i * 4] / 255.0;               // R
          float32Data[i + stride] = imageData[i * 4 + 1] / 255.0;  // G
          float32Data[i + stride * 2] = imageData[i * 4 + 2] / 255.0; // B
        }

        const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]);
        const feeds: Record<string, ort.Tensor> = {};
        feeds[session.inputNames[0]] = inputTensor;

        const results = await session.run(feeds);
        const output = results[session.outputNames[0]];
        const data = output.data as Float32Array;
        const dims = output.dims;

        const boxes: DetectionBox[] = [];

        if (dims.length === 3 && dims[2] === 6) {
          const numDetections = dims[1];
          for (let i = 0; i < numDetections; i++) {
            const offset = i * 6;
            const score = data[offset + 4];
            if (score >= CONFIDENCE_THRESHOLD) {
              boxes.push({
                x1: Math.max(0, data[offset] / YOLO_INPUT_SIZE),
                y1: Math.max(0, data[offset + 1] / YOLO_INPUT_SIZE),
                x2: Math.min(1, data[offset + 2] / YOLO_INPUT_SIZE),
                y2: Math.min(1, data[offset + 3] / YOLO_INPUT_SIZE),
                score,
                label: 'tomato'
              });
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
            if (maxScore >= CONFIDENCE_THRESHOLD) {
              const cx = data[0 * numAnchors + i];
              const cy = data[1 * numAnchors + i];
              const w = data[2 * numAnchors + i];
              const h = data[3 * numAnchors + i];
              boxes.push({
                x1: Math.max(0, (cx - w / 2) / YOLO_INPUT_SIZE),
                y1: Math.max(0, (cy - h / 2) / YOLO_INPUT_SIZE),
                x2: Math.min(1, (cx + w / 2) / YOLO_INPUT_SIZE),
                y2: Math.min(1, (cy + h / 2) / YOLO_INPUT_SIZE),
                score: maxScore,
                label: 'tomato'
              });
            }
          }
        }

        resolve(boxes);
      } catch (err) {
        console.error(`YOLO推論エラー (${imageUrl}):`, err);
        resolve([]);
      }
    };
    img.onerror = () => resolve([]);
    img.src = imageUrl;
  });
}

// --- 3. データ処理 (dx, dyから2D累積位置を計算) ---
const processImagesForPanorama = (logs: LogEntry[], width: number) => {
  const result = [];
  let accumulatedX = 0;
  let accumulatedY = 0;

  for (let i = 0; i < logs.length; i++) {
    const log = logs[i];
    const dxNum = typeof log.dx === 'string' ? parseFloat(log.dx) : (log.dx || 0);
    const dyNum = typeof log.dy === 'string' ? parseFloat(log.dy) : (log.dy || 0);

    if (i > 0) {
      const stepX = dxNum ? (dxNum / 100) : width;
      const stepY = dyNum ? (dyNum / 100) : 0;
      accumulatedX -= stepX;
      accumulatedY += stepY;
    }

    const featurePoints: [number, number][] = [
      [(Math.random() - 0.5) * width * 0.8, (Math.random() - 0.5) * IMAGE_HEIGHT * 0.8],
      [(Math.random() - 0.5) * width * 0.8, (Math.random() - 0.5) * IMAGE_HEIGHT * 0.8],
      [(Math.random() - 0.5) * width * 0.8, (Math.random() - 0.5) * IMAGE_HEIGHT * 0.8]
    ];

    result.push({
      id: i,
      url: `/images/left/${log.filename}`,
      position: [accumulatedX, accumulatedY, 0] as [number, number, number],
      filename: log.filename,
      featurePoints
    });
  }
  return result;
}; 

// --- 4. 3D サブコンポーネント ---

const CameraRig = ({ targetX, targetY }: { targetX: number; targetY: number }) => {
  const { camera, controls } = useThree();
  
  useFrame((_, delta) => {
    const dampSpeed = 5 * delta;
    const nextX = THREE.MathUtils.lerp(camera.position.x, targetX, dampSpeed);
    const nextY = THREE.MathUtils.lerp(camera.position.y, targetY, dampSpeed);
    camera.position.set(nextX, nextY, camera.position.z);

    if (controls) {
      const orbit = controls as unknown as { target: THREE.Vector3; update: () => void };
      const nextTargetX = THREE.MathUtils.lerp(orbit.target.x, targetX, dampSpeed);
      const nextTargetY = THREE.MathUtils.lerp(orbit.target.y, targetY, dampSpeed);
      orbit.target.set(nextTargetX, nextTargetY, orbit.target.z);
      orbit.update();
    }
  });

  return null;
};

// 検出されたトマトを赤枠で囲むコンポーネント
const TomatoBoundingBoxes = ({ detections }: { detections: DetectionBox[] }) => {
  if (!detections || detections.length === 0) return null;

  return (
    <group position={[0, 0, 0.02]}>
      {detections.map((box, idx) => {
        const left = (box.x1 - 0.5) * IMAGE_WIDTH;
        const right = (box.x2 - 0.5) * IMAGE_WIDTH;
        const top = (0.5 - box.y1) * IMAGE_HEIGHT;
        const bottom = (0.5 - box.y2) * IMAGE_HEIGHT;

        const rectPoints: [number, number, number][] = [
          [left, top, 0],
          [right, top, 0],
          [right, bottom, 0],
          [left, bottom, 0],
          [left, top, 0]
        ];

        return (
          <group key={idx}>
            <Line
              points={rectPoints}
              color="#ff1a1a"
              lineWidth={2.5}
            />
            <Text
              position={[left, top + 0.06, 0.01]}
              fontSize={0.09}
              color="#ff3333"
              anchorX="left"
              anchorY="bottom"
              outlineWidth={0.01}
              outlineColor="#000000"
            >
              {`${box.label} ${(box.score * 100).toFixed(0)}%`}
            </Text>
          </group>
        );
      })}
    </group>
  );
};

const ImagePanel = ({ 
  url, 
  position, 
  index, 
  mode = 'cylinder', 
  curveFactor = CURVE_FACTOR,
  detections = [],
  showDetections = true
}: { 
  url: string, 
  position: [number, number, number], 
  index: number,
  mode?: ViewMode,
  curveFactor?: number,
  detections?: DetectionBox[],
  showDetections?: boolean
}) => {
  const groupRef = useRef<THREE.Group>(null!);
  const meshRef = useRef<THREE.Mesh>(null!);

  useFrame(({ camera }) => {
    if (!groupRef.current || !meshRef.current) return;

    const distanceX = Math.abs(position[0] - camera.position.x);

    let dynamicZ = 0;
    if (mode === 'cylinder') {
      dynamicZ = -Math.pow(distanceX, 2) * curveFactor;
    } else if (mode === 'tunnel') {
      dynamicZ = -Math.pow(distanceX, 2.5) * (curveFactor * 10);
    } else {
      dynamicZ = 0;
    }
    
    groupRef.current.position.set(position[0], position[1], dynamicZ);

    const isFront = distanceX < (IMAGE_WIDTH * 0.5);
    groupRef.current.renderOrder = isFront ? 999 : index;

    const targetOpacity = distanceX < (IMAGE_WIDTH * 0.6) ? OPACITY_VISIBLE : OPACITY_INVISIBLE;
    const material = meshRef.current.material;
    if (material && 'opacity' in material) {
      (material as THREE.Material & { opacity: number }).opacity = THREE.MathUtils.lerp(
        (material as THREE.Material & { opacity: number }).opacity,
        targetOpacity,
        0.1
      );
    }
  });

  return (
    <group ref={groupRef} position={[position[0], position[1], position[2]]} renderOrder={index}>
      <DreiImage 
        ref={meshRef}
        url={url} 
        scale={[IMAGE_WIDTH, IMAGE_HEIGHT]}
        transparent
        opacity={OPACITY_INVISIBLE}
        side={THREE.DoubleSide}
      />
      {showDetections && <TomatoBoundingBoxes detections={detections} />}
    </group>
  );
};

const FeatureMatchLines = ({ items }: { items: ReturnType<typeof processImagesForPanorama> }) => {
  const lines = useMemo(() => {
    const result: { id: string; points: THREE.Vector3[] }[] = [];

    for (let i = 0; i < items.length - 1; i++) {
      const imgA = items[i];
      const imgB = items[i + 1];

      imgA.featurePoints.forEach((ptA, ptIdx) => {
        const ptB = imgB.featurePoints[ptIdx] || imgB.featurePoints[0];

        const start = new THREE.Vector3(imgA.position[0] + ptA[0], imgA.position[1] + ptA[1], 0.1);
        const end = new THREE.Vector3(imgB.position[0] + ptB[0], imgB.position[1] + ptB[1], 0.1);

        const midX = (start.x + end.x) / 2;
        const midY = (start.y + end.y) / 2;
        const midZ = 0.8;

        const curve = new THREE.QuadraticBezierCurve3(start, new THREE.Vector3(midX, midY, midZ), end);
        result.push({
          id: `${i}-${ptIdx}`,
          points: curve.getPoints(20)
        });
      });
    }
    return result;
  }, [items]);

  return (
    <group>
      {lines.map((line) => (
        <Line
          key={line.id}
          points={line.points}
          color="#00ffcc"
          lineWidth={1.5}
          transparent
          opacity={0.6}
        />
      ))}
    </group>
  );
};

// --- CSS Styles ---
const uiContainerStyle: React.CSSProperties = {
  position: 'absolute',
  bottom: '30px',
  left: '50%',
  transform: 'translateX(-50%)',
  display: 'flex',
  alignItems: 'center',
  gap: '20px',
  zIndex: 10,
  background: 'rgba(0,0,0,0.7)',
  padding: '10px 20px',
  borderRadius: '30px',
  border: '1px solid #444'
};

const arrowButtonStyle: React.CSSProperties = {
  width: '50px',
  height: '50px',
  borderRadius: '50%',
  border: '2px solid #fff',
  background: '#333',
  color: '#fff',
  fontSize: '24px',
  cursor: 'pointer',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  transition: 'background 0.2s'
};

const labelStyle: React.CSSProperties = {
  color: '#fff',
  fontFamily: 'monospace',
  textAlign: 'center',
  minWidth: '120px'
};

const controlPanelStyle: React.CSSProperties = {
  position: 'absolute',
  top: '50px',
  right: '20px',
  zIndex: 100,
  background: 'rgba(0, 0, 0, 0.75)',
  padding: '16px',
  borderRadius: '12px',
  border: '1px solid #444',
  color: '#fff',
  width: '240px',
  fontFamily: 'sans-serif',
  fontSize: '12px'
};

// --- 5. 3D パノラマ表示コンポーネント ---
function PanoramaView({ rawLogs }: { rawLogs: LogEntry[] }) {
  const processedData = useMemo(() => processImagesForPanorama(rawLogs, IMAGE_WIDTH), [rawLogs]);
  const initialPos = processedData[0]?.position || [0, 0, 0];

  const [targetCameraPos, setTargetCameraPos] = useState<[number, number]>([initialPos[0], initialPos[1]]);
  const [viewMode, setViewMode] = useState<ViewMode>('cylinder');
  const [curveFactor, setCurveFactor] = useState(CURVE_FACTOR);
  const showMatches = true;
  const [showDetections, setShowDetections] = useState(true);

  const [detectionsMap, setDetectionsMap] = useState<Record<string, DetectionBox[]>>({});
  const [yoloStatus, setYoloStatus] = useState<string>('YOLO26n モデル読込中...');

  useEffect(() => {
    let isCancelled = false;

    async function detectAllImages() {
      if (processedData.length === 0) return;

      try {
        setYoloStatus('YOLO26n 初期化中...');
        const session = await ort.InferenceSession.create(YOLO_MODEL_URL, {
          executionProviders: ['wasm']
        });

        let completed = 0;
        for (const item of processedData) {
          if (isCancelled) break;
          setYoloStatus(`YOLO26n 検出中 (${completed + 1}/${processedData.length})`);
          
          const boxes = await runYolo26Detection(session, item.url);
          if (!isCancelled) {
            setDetectionsMap((prev) => ({
              ...prev,
              [item.filename]: boxes
            }));
          }
          completed++;
        }

        if (!isCancelled) {
          setYoloStatus(`YOLO26n 検出完了 (${completed}枚)`);
        }
      } catch (err) {
        console.error('YOLO26n モデルのロードに失敗しました:', err);
        if (!isCancelled) {
          setYoloStatus('YOLO26n モデル読込エラー (/models/yolo26n.onnx を確認)');
        }
      }
    }

    detectAllImages();
    return () => {
      isCancelled = true;
    };
  }, [processedData]);

  useEffect(() => {
    if (processedData.length > 0) {
      setTargetCameraPos([processedData[0].position[0], processedData[0].position[1]]);
    }
  }, [processedData]);

  const currentImageIdx = useMemo(() => {
    const idx = processedData.findIndex(d => Math.abs(d.position[0] - targetCameraPos[0]) < 0.001);
    return idx === -1 ? 0 : idx;
  }, [processedData, targetCameraPos]);

  const moveLeft = () => {
    if (currentImageIdx > 0) {
      const prev = processedData[currentImageIdx - 1];
      setTargetCameraPos([prev.position[0], prev.position[1]]);
    }
  };

  const moveRight = () => {
    if (currentImageIdx < processedData.length - 1) {
      const next = processedData[currentImageIdx + 1];
      setTargetCameraPos([next.position[0], next.position[1]]);
    }
  };

  const currentDetectionsCount = detectionsMap[processedData[currentImageIdx]?.filename]?.length || 0;

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#000', overflow: 'hidden', position: 'relative' }}>
      <header style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        background: 'rgba(230, 255, 250, 0.95)',
        borderBottom: '2px solid #319795',
        padding: '10px 0',
        textAlign: 'center',
        color: '#234e52',
        zIndex: 999999,
        pointerEvents: 'none',
        fontFamily: 'sans-serif'
      }}>
        <p style={{ margin: 0, fontSize: '14px', fontWeight: 'bold' }}>
          検証用ビルド日時: {BUILD_TIME} ｜ {yoloStatus}
        </p>
      </header>

      <div style={controlPanelStyle}>
        <div style={{ fontWeight: 'bold', marginBottom: '10px', color: '#00ffcc' }}>3D View Controls</div>
        
        <label style={{ display: 'block', marginBottom: '4px' }}>Depth Mode:</label>
        <select 
          value={viewMode} 
          onChange={(e) => setViewMode(e.target.value as ViewMode)}
          style={{ width: '100%', padding: '6px', background: '#222', color: '#fff', border: '1px solid #555', borderRadius: '4px', marginBottom: '12px' }}
        >
          <option value="plane">Plane (Flat)</option>
          <option value="cylinder">Cylinder (Curved)</option>
          <option value="tunnel">Tunnel (Deep)</option>
        </select>

        {viewMode !== 'plane' && (
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '4px' }}>Curve Factor:</label>
            <input 
              type="range" 
              min="0.000005" 
              max="0.000050" 
              step="0.000005" 
              value={curveFactor}
              onChange={(e) => setCurveFactor(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>
        )}

        <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', gap: '8px', color: '#ff6666', fontWeight: 'bold' }}>
          <input 
            type="checkbox" 
            checked={showDetections} 
            onChange={(e) => setShowDetections(e.target.checked)} 
          />
          Show Tomato Boxes (YOLO26n)
        </label>
      </div>

      <Canvas>
        <PerspectiveCamera 
          makeDefault 
          position={[initialPos[0], initialPos[1], IMAGE_WIDTH * 1.5]} 
          fov={60} 
          far={100000} 
        />
        
        <OrbitControls 
          makeDefault
          enableDamping 
          dampingFactor={0.1}
          target={[initialPos[0], initialPos[1], 0]} 
          enablePan={false} 
        />

        <CameraRig targetX={targetCameraPos[0]} targetY={targetCameraPos[1]} />

        <color attach="background" args={['#111']} />
        <ambientLight intensity={2} />

        <Grid 
          position={[initialPos[0], -IMAGE_HEIGHT / 2 - 0.2, 0]} 
          args={[2000, 2000]} 
          cellSize={1}     
          cellThickness={0.2}
          cellColor="#444444"
          sectionSize={5} 
          sectionThickness={0.5}
          sectionColor="#777777"
          fadeDistance={50}  
          infiniteGrid      
        />

        <Suspense fallback={null}>
          {processedData.map((img, idx) => (
            <ImagePanel 
              key={img.id} 
              index={idx}
              url={img.url} 
              position={img.position}
              mode={viewMode}
              curveFactor={curveFactor}
              detections={detectionsMap[img.filename]}
              showDetections={showDetections}
            />
          ))}

          {showMatches && <FeatureMatchLines items={processedData} />}
        </Suspense>

      </Canvas>

      <div style={uiContainerStyle}>
        <button style={arrowButtonStyle} onClick={moveLeft}>←</button>
        <div style={labelStyle}>
          Target X: {Math.round(targetCameraPos[0])} <br/>
          <span style={{fontSize:'11px', color:'#ff6666'}}>
            検出トマト数: {currentDetectionsCount}個
          </span>
        </div>
        <button style={arrowButtonStyle} onClick={moveRight}>→</button>
      </div>
    </div>
  );  
}

// --- 6. メインAppコンポーネント (images_a_50.csv 読み込み) ---
export default function App() {
  const [data, setData] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Papa.parse<LogEntry>('/images_a_0_0.csv', {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (results: ParseResult<LogEntry>) => {
        console.log('読み込んだデータ:', results.data);
        setData(results.data);
        setLoading(false);
      },
      error: (error: Error) => {
        console.error('CSV読み込みエラー:', error);
        setLoading(false);
      }
    });
  }, []);

  if (loading) {
    return <div style={{ color: '#fff', padding: '20px' }}>CSVデータを読み込み中...</div>;
  }

  return <PanoramaView rawLogs={data} />;
}