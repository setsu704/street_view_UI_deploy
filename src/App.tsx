import React, { useMemo, useRef, useState, useLayoutEffect, Suspense } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { Image as DreiImage, OrbitControls, PerspectiveCamera, Grid, Line } from '@react-three/drei';
import * as THREE from 'three';

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

// --- 1. 設定 ---
const IMAGE_WIDTH = 3.2; 
const IMAGE_HEIGHT = 2.4;
const MOVE_STEP = 100; 

const OPACITY_VISIBLE = 0.9;
const OPACITY_INVISIBLE = 0.5;
const CURVE_FACTOR = 0.000015;

export type ViewMode = 'plane' | 'cylinder' | 'tunnel';

// AKAZEログデータ
const RAW_LOGS = [
  { filename: 'left_20241219_114330_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114331_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114332_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114333_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114335_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114336_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114337_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114338_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114340_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114341_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114342_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114343_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114345_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114346_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114347_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114348_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114349_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114351_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114352_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114353_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114354_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114356_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114357_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114358_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114359_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114401_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114402_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114403_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114404_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114406_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114407_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114408_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114409_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114411_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114412_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114413_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114414_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114416_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114417_rgba.png', dx: 0, dy: 0 },
  { filename: 'left_20241219_114418_rgba.png', dx: 0, dy: 0 },
];

// --- 2. データ処理 ---
const processImagesForPanorama = (logs: typeof RAW_LOGS, width: number) => {
  const result = [];
  let accumulatedX = 0;
  for (let i = 0; i < logs.length; i++) {
    const log = logs[i];
    
    const featurePoints: [number, number][] = [
      [(Math.random() - 0.5) * width * 0.8, (Math.random() - 0.5) * IMAGE_HEIGHT * 0.8],
      [(Math.random() - 0.5) * width * 0.8, (Math.random() - 0.5) * IMAGE_HEIGHT * 0.8],
      [(Math.random() - 0.5) * width * 0.8, (Math.random() - 0.5) * IMAGE_HEIGHT * 0.8]
    ];

    result.push({
      id: i,
      url: `/images/left/${log.filename}`,
      position: [accumulatedX, 0, 0] as [number, number, number],
      filename: log.filename,
      featurePoints
    });
    const stepX = log.dx ? (log.dx / 100) : width;
    accumulatedX -= stepX;
  }
  return result;
}; 

// --- 3. 表示コンポーネント ---

const CameraRig = ({ targetX }: { targetX: number }) => {
  const { camera, controls } = useThree();
  
  useFrame((_, delta) => {
    const dampSpeed = 5 * delta;
    const currentX = camera.position.x;
    const nextX = THREE.MathUtils.lerp(currentX, targetX, dampSpeed);
    camera.position.setX(nextX);

    if (controls) {
      const orbit = controls as unknown as { target: THREE.Vector3; update: () => void };
      const currentTargetX = orbit.target.x;
      const nextTargetX = THREE.MathUtils.lerp(currentTargetX, targetX, dampSpeed);
      
      orbit.target.setX(nextTargetX);
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
    groupRef.current.position.setZ(dynamicZ);

    const isFront = distanceX < (IMAGE_WIDTH * 0.5);
    groupRef.current.renderOrder = isFront ? 999 : index;

    const targetOpacity = distanceX < (IMAGE_WIDTH * 0.6) ? OPACITY_VISIBLE : OPACITY_INVISIBLE;
    const material = meshRef.current.material;
    if (material && 'opacity' in material) {
      (material as any).opacity = THREE.MathUtils.lerp((material as any).opacity, targetOpacity, 0.1);
    }
  });

  return (
    <group ref={groupRef} position={[position[0], position[1], 0]} renderOrder={index}>
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

        const start = new THREE.Vector3(imgA.position[0] + ptA[0], ptA[1], 0.1);
        const end = new THREE.Vector3(imgB.position[0] + ptB[0], ptB[1], 0.1);

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

// --- 4. メインコンポーネント ---

export default function PanoramaView() {
  const processedData = useMemo(() => processImagesForPanorama(RAW_LOGS, IMAGE_WIDTH), []);
  const initialX = processedData[0]?.position[0] || 0;

  const [targetCameraX, setTargetCameraX] = useState(initialX);
  const [viewMode, setViewMode] = useState<ViewMode>('cylinder');
  const [curveFactor, setCurveFactor] = useState(CURVE_FACTOR);
  const [showMatches, setShowMatches] = useState(true);

  const currentImageIdx = useMemo(() => {
    const idx = processedData.findIndex(d => Math.abs(d.position[0] - targetCameraX) < 0.001);
    return idx === -1 ? 0 : idx;
  }, [processedData, targetCameraX]);

  const moveLeft = () => {
    if (currentImageIdx > 0) {
      setTargetCameraX(processedData[currentImageIdx - 1].position[0]);
    }
  };

  const moveRight = () => {
    if (currentImageIdx < processedData.length - 1) {
      setTargetCameraX(processedData[currentImageIdx + 1].position[0]);
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
          position={[initialX, 0, IMAGE_WIDTH * 1.5]} 
          fov={60} 
          far={100000} 
        />
        
        <OrbitControls 
          makeDefault
          enableDamping 
          dampingFactor={0.1}
          target={[initialX, 0, 0]} 
          enablePan={false} 
        />

        <CameraRig targetX={targetCameraX} />

        <color attach="background" args={['#111']} />

        <ambientLight intensity={2} />

        {/* 🟢 修正：Grid の線幅（cellThickness）を 5 から 0.2 に修正 */}
        <Grid 
          position={[initialX, -IMAGE_HEIGHT / 2 - 0.2, 0]} 
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
          Target: {Math.round(targetCameraX)} <br/>
          <span style={{fontSize:'10px', color:'#888'}}>Step: {MOVE_STEP}px</span>
        </div>
        <button style={arrowButtonStyle} onClick={moveRight}>→</button>
      </div>

    </div>
  );  
}