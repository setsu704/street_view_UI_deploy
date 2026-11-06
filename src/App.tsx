import React, { useMemo, useRef, useState, useEffect, Suspense } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { Image as DreiImage, OrbitControls, PerspectiveCamera, Grid, Line } from '@react-three/drei';
import * as THREE from 'three';
import Papa from 'papaparse';

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
const MOVE_STEP = 100; 

const OPACITY_VISIBLE = 0.9;
const OPACITY_INVISIBLE = 0.5;
const CURVE_FACTOR = 0.000015;

export type ViewMode = 'plane' | 'cylinder' | 'tunnel';

export interface LogEntry {
  filename: string;
  dx: string | number;
  dy: string | number;
}

// --- 2. データ処理 (dx, dyから2D累積位置を計算) ---
const processImagesForPanorama = (logs: LogEntry[], width: number) => {
  const result = [];
  let accumulatedX = 0;
  let accumulatedY = 0;

  for (let i = 0; i < logs.length; i++) {
    const log = logs[i];
    const dxNum = typeof log.dx === 'string' ? parseFloat(log.dx) : (log.dx || 0);
    const dyNum = typeof log.dy === 'string' ? parseFloat(log.dy) : (log.dy || 0);

    // 1番目の画像(i=0)は基準点(0,0)に配置
    if (i > 0) {
      // 基準ピクセル移動量を3D空間スケールに変換
      const stepX = dxNum ? (dxNum / 100) : width;
      const stepY = dyNum ? (dyNum / 100) : 0;
      
      accumulatedX -= stepX;
      accumulatedY += stepY; // 3D空間のY軸正方向と合わせる
    }

    // 可視化用の特徴点（ダミー表示用）
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

// --- 3. 3D サブコンポーネント ---

const CameraRig = ({ targetX, targetY }: { targetX: number; targetY: number }) => {
  const { camera, controls } = useThree();
  
  useFrame((_, delta) => {
    const dampSpeed = 5 * delta;
    const currentX = camera.position.x;
    const currentY = camera.position.y;
    
    const nextX = THREE.MathUtils.lerp(currentX, targetX, dampSpeed);
    const nextY = THREE.MathUtils.lerp(currentY, targetY, dampSpeed);
    
    camera.position.set(nextX, nextY, camera.position.z);

    if (controls) {
      const orbit = controls as unknown as { target: THREE.Vector3; update: () => void };
      const currentTargetX = orbit.target.x;
      const currentTargetY = orbit.target.y;
      
      const nextTargetX = THREE.MathUtils.lerp(currentTargetX, targetX, dampSpeed);
      const nextTargetY = THREE.MathUtils.lerp(currentTargetY, targetY, dampSpeed);
      
      orbit.target.set(nextTargetX, nextTargetY, orbit.target.z);
      orbit.update();
    }
  });

  return null;
};

const ImagePanel = ({ url, position, index, mode = 'cylinder', curveFactor = CURVE_FACTOR }: { 
  url: string, 
  position: [number, number, number], 
  index: number,
  mode?: ViewMode,
  curveFactor?: number
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
    
    // X, Y, Z の位置を設定
    groupRef.current.position.set(position[0], position[1], dynamicZ);

    const isFront = distanceX < (IMAGE_WIDTH * 0.5);
    groupRef.current.renderOrder = isFront ? 999 : index;

    const targetOpacity = distanceX < (IMAGE_WIDTH * 0.6) ? OPACITY_VISIBLE : OPACITY_INVISIBLE;
    const material = meshRef.current.material;
    if (material && 'opacity' in material) {
      (material as any).opacity = THREE.MathUtils.lerp((material as any).opacity, targetOpacity, 0.1);
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
  width: '220px',
  fontFamily: 'sans-serif',
  fontSize: '12px'
};

// --- 4. 3D パノラマ表示コンポーネント ---
function PanoramaView({ rawLogs }: { rawLogs: LogEntry[] }) {
  const processedData = useMemo(() => processImagesForPanorama(rawLogs, IMAGE_WIDTH), [rawLogs]);
  const initialPos = processedData[0]?.position || [0, 0, 0];

  const [targetCameraPos, setTargetCameraPos] = useState<[number, number]>( [initialPos[0], initialPos[1]] );
  const [viewMode, setViewMode] = useState<ViewMode>('cylinder');
  const [curveFactor, setCurveFactor] = useState(CURVE_FACTOR);
  const [showMatches, setShowMatches] = useState(true);

  // データ更新時にカメラ位置をリセット
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
          検証用ビルド日時: {BUILD_TIME}
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

        <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', gap: '8px' }}>
          <input 
            type="checkbox" 
            checked={showMatches} 
            onChange={(e) => setShowMatches(e.target.checked)} 
          />
          Show AKAZE Matches
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
            />
          ))}

          {showMatches && <FeatureMatchLines items={processedData} />}
        </Suspense>

      </Canvas>

      <div style={uiContainerStyle}>
        <button style={arrowButtonStyle} onClick={moveLeft}>←</button>
        <div style={labelStyle}>
          Target X: {Math.round(targetCameraPos[0])} <br/>
          <span style={{fontSize:'10px', color:'#888'}}>Step: {MOVE_STEP}px</span>
        </div>
        <button style={arrowButtonStyle} onClick={moveRight}>→</button>
      </div>
    </div>
  );  
}

// --- 5. メインAppコンポーネント (images_a_50.csv 読み込み) ---
export default function App() {
  const [data, setData] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 読み込み対象を images_a_50.csv に指定
    Papa.parse('/images_a_50.csv', {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        console.log('読み込んだデータ:', results.data);
        setData(results.data as LogEntry[]);
        setLoading(false);
      },
      error: (error) => {
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