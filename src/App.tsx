import React, { useMemo, useRef, useState, useLayoutEffect, Suspense } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { Image as DreiImage, OrbitControls, PerspectiveCamera, Grid, Line } from '@react-three/drei'; // Line を追加
import * as THREE from 'three';

//時刻表示
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

// 移動量
const MOVE_STEP = 100; 
//一枚ずつの移動にしたい時
//const MOVE_STEP = IMAGE_WIDTH

// カメラに「映っている」とみなす範囲
// const VISIBLE_THRESHOLD = IMAGE_WIDTH * 0.8; 

// 透明度設定
const OPACITY_VISIBLE = 0.9;
const OPACITY_INVISIBLE = 0.5;

//湾曲の強さ 数値を大きくすると奥に回り込む
const CURVE_FACTOR = 0.000015;

// 【追加】表示モードの型定義
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
// ---改良版を下に書いたため、コメントアウト---
const processImagesForPanorama = (logs: typeof RAW_LOGS, width: number) => {
  const result = [];
  let accumulatedX = 0;
  for (let i = 0; i < logs.length; i++) {
    const log = logs[i];
   
    // 【追加】AKAZE特徴点の模擬データ（画像ローカル座標 -0.5〜0.5 相当）
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
      featurePoints // 【追加】
    });
    //2枚目以降の位置決定のために累積
    //dxが０のときは等間隔に配置されるように設定
    const stepX = log.dx ? (log.dx / 100) : width;
    accumulatedX -= stepX;
  }
  return result;
}; 

// 改良版（imagewidth）のエラーが出たため、いったん保留
{/*interface Log {
  filename: string;
  dx: number; // 前の画像からの相対的な移動量（ピクセルまたは3D単位）
}

const processImagesForPanorama = (logs: Log[], imageWidth: number) => {
  const result = [];
  let accumulatedX = 0; // 1枚目は必ず 0 からスタート

  for (let i = 0; i < logs.length; i++) {
    const log = logs[i];

    // ① 初項（1枚目）は 0 のまま配置し、2枚目以降のループの「最後」または「次」で累積する
    result.push({
      id: i,
      url: `/images/left/${log.filename}`,
      position: [accumulatedX, 0, 0] as [number, number, number],
      filename: log.filename
    });

    // ② 次の画像のための配置座標を更新（画像幅と移動量を考慮）
    // dx の仕様に合わせて、以下のいずれかの数式を選択
    // パターンA (純粋な移動量):
    accumulatedX -= log.dx; 
    
    // パターンB (重複率などを考慮する場合、後述のロジックに差し替え):
    // accumulatedX -= (imageWidth - log.dx);
  }

  return result;
}; */}
// --- 3. 表示コンポーネント ---

// カメラ制御用コンポーネント（滑らかな移動を追加）
const CameraRig = ({ targetX }: { targetX: number }) => {
  const { camera, controls } = useThree();
  
  // 毎フレーム実行されるループ
  useFrame((_, delta) => {
    // 現在のX座標から目標のX座標へ、少しずつ近づける（線形補間: Lerp）
    // 第3引数の数値を大きくすると速く、小さくすると遅く（粘り強く）なります
    const dampSpeed = 5 * delta;
    
    // カメラの移動
    const currentX = camera.position.x;
    const nextX = THREE.MathUtils.lerp(currentX, targetX, dampSpeed);
    camera.position.setX(nextX);

    // Controls（視点ターゲット）の移動
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

// 【追加パラメータ対応】mode, curveFactor を受け取れるように拡張
const ImagePanel = ({ url, position, index, mode = 'cylinder', curveFactor = CURVE_FACTOR }: { 
  url: string, 
  position: [number, number, number], 
  index: number,
  mode?: ViewMode, // 【追加】
  curveFactor?: number // 【追加】
}) => {
  const groupRef = useRef<THREE.Group>(null!);
  const meshRef = useRef<THREE.Mesh>(null!);

  useLayoutEffect(() => {
    if (meshRef.current && meshRef.current.material) {
      const material = meshRef.current.material as THREE.Material;
      material.depthWrite = false;
    }
  }, []);

  // 滑らかなカメラ移動に合わせて、毎フレーム透明度を計算する
  // (Propsでカメラ位置を受け取ると再レンダリングが頻発するため、refで直接操作する)
  useFrame(({ camera }) => {
    if (!groupRef.current || !meshRef.current) return;

    // カメラと画像の距離を計算
    const distanceX = Math.abs(position[0] - camera.position.x);
    const absDistanceX = Math.abs(distanceX);

    // 【追加】モードに応じた動的Z軸位置計算
    let dynamicZ = 0;
    if (mode === 'cylinder') {
      dynamicZ = -Math.pow(distanceX, 2) * curveFactor;
    } else if (mode === 'tunnel') {
      dynamicZ = -Math.pow(distanceX, 2.5) * (curveFactor * 10);
    } else {
      dynamicZ = 0; // plane モード
    }
    groupRef.current.position.setZ(dynamicZ);

    //Z軸に下がるだけで小さく見えるが、より強調したい時に有効化する
    // const dynamicScale = 1 / (1 + absDistanceX * 0.0005);
    // groupRef.current.scale.set(dynamicScale, dynamicScale, 1);

    const isFront = absDistanceX < (IMAGE_WIDTH * 0.5);
    groupRef.current.renderOrder = isFront ? 999 : index;
    // 距離に応じた目標透明度を決定,正面付近ならくっきり、離れると薄く
    // 瞬時に切り替えたい場合は meshRef.current.material.opacity = targetOpacity; だけでもOK
    const targetOpacity = absDistanceX < (IMAGE_WIDTH * 0.6) ? OPACITY_VISIBLE : OPACITY_INVISIBLE;
    const material = meshRef.current.material;
    if (material && 'opacity' in material) {
     // 🟢 括弧の位置、スペル、カンマをすべて修正
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
        // 初期値
        opacity={OPACITY_INVISIBLE}
        side={THREE.DoubleSide}
      />
    </group>
  );
};

// 【追加】AKAZE 3D マッチング線描画コンポーネント
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

// 【追加】右上コントロールパネル用スタイル
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
  
  // 初期位置（最初の画像がある場所。今回のロジックでは必ず 0）
  const initialX = processedData[0]?.position[0] || 0;

  // 目標位置（Target）を管理するState。
  // カメラの実際の座標（Current）はCameraRig内でLerp計算される。
  // カメラの移動目標（初期値は1枚目の座標 = 0）
  const [targetCameraX, setTargetCameraX] = useState(initialX);

  // 【追加】インタラクションコントロール用State
  const [viewMode, setViewMode] = useState<ViewMode>('cylinder');
  const [curveFactor, setCurveFactor] = useState(CURVE_FACTOR);
  const [showMatches, setShowMatches] = useState(true);

  // 🟢 インデックスベースで移動を制御するためのヘルパー（境界を超えない防衛策）
  const currentImageIdx = useMemo(() => {
    const idx = processedData.findIndex(d => Math.abs(d.position[0] -targetCameraX) < 0.001);
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

  //const minX = Math.min(...processedData.map(d => d.position[0]));
  //const maxX = Math.max(...processedData.map(d => d.position[0]));
  //const centerX = (minX + maxX) / 2;
  //const totalWidth = Math.abs(maxX - minX) + IMAGE_WIDTH;

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#000', overflow: 'hidden', position: 'relative' }}>
      
      {/* 🟢 追加：検証用ビルド日時ヘッダー（Canvasの外側・最前面に配置） */}
      <header style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        background: 'rgba(230, 255, 250, 0.95)', // 背景が黒なので少し透過させて馴染ませる
        borderBottom: '2px solid #319795',
        padding: '10px 0',
        textAlign: 'center',
        color: '#234e52',
        zIndex: 999999, // 3D空間や他のUIより絶対に前に出す
        pointerEvents: 'none', // マウス操作を透過させ、3D操作の邪魔をしない
        fontFamily: 'sans-serif'
      }}>
        <p style={{ margin: 0, fontSize: '14px', fontWeight: 'bold' }}>
          検証用ビルド日時: {BUILD_TIME}
        </p>
      </header>

      {/* 【追加】3D表示制御パネル（UI） */}
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
          // 初期位置はminXに設定
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

        {/* State(目標値)を渡して、内部で滑らかに移動させる */}
        <CameraRig targetX={targetCameraX} />

        <color attach="background" args={['#111']} />

        <ambientLight intensity={2} />

        <Grid 
          position={[initialX, -IMAGE_HEIGHT / 2 - 0.2, 0]} // 画像の底辺より少し下に配置（高さ調節）
          args={[2000, 2000]} // グリッドの全体のサイズ [幅, 奥行き]
          cellSize={5}    // 細かいマスのサイズ
          cellThickness={5}
          cellColor="#333333"
          sectionSize={2.5} // 太線のマスのサイズ
          sectionThickness={1}
          sectionColor="#555555"
          fadeDistance={20}  // 奥にいくほどグリッドが薄れていく距離
          infiniteGrid      // 無限に床が続いているように見せる
        />

        <Suspense fallback={null}>
          {processedData.map((img, idx) => (
          <ImagePanel 
            key={img.id} 
            index={idx}
            url={img.url} 
            position={img.position}
            mode={viewMode} // 【追加】
            curveFactor={curveFactor} // 【追加】
            // ImagePanel自体がuseFrameでカメラ位置を取得するため、currentCameraXのProps渡しは不要
          />
        ))}

        {/* 【追加】AKAZE特徴点マッチング線のトグル表示 */}
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