import React, { useMemo, useRef, useState, useLayoutEffect } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { Image as DreiImage, OrbitControls, PerspectiveCamera } from '@react-three/drei'; // Grid, 
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
const IMAGE_WIDTH = 320; 
const IMAGE_HEIGHT = 240;

// 移動量
const MOVE_STEP = 100; 
//一枚ずつの移動にしたい時
//const MOVE_STEP = IMAGE_WIDTH

// カメラに「映っている」とみなす範囲
const VISIBLE_THRESHOLD = IMAGE_WIDTH * 0.8; 

// 透明度設定
const OPACITY_VISIBLE = 0.8;
const OPACITY_INVISIBLE = 0.2;

//湾曲の強さ　数値を大きくすると奥に回り込む
const CURVE_FACTOR = 0.0015

// AKAZEログデータ
const RAW_LOGS = [
  { filename: 'left_20241219_114330.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114331.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114332.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114333.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114335.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114336.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114337.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114338.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114340.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114341.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114342.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114343.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114345.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114346.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114347.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114348.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114349.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114351.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114352.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114353.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114354.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114356.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114357.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114358.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114359.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114401.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114402.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114403.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114404.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114406.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114407.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114408.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114409.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114411.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114412.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114413.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114414.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114416.jpg', dx: 0, dy: 0 },
  { filename: 'left_20241219_114417.jpg', dx: 0, dy: 0},
  { filename: 'left_20241219_114418.jpg', dx: 0, dy: 0},


  //{ filename: 'left_20241219_114330.jpg', dx: 0, dy: 0 },
  //{ filename: 'left_20241219_114331.jpg', dx: -93.43, dy: -13.05 },
  //{ filename: 'left_20241219_114332.jpg', dx: 28.67, dy: -34.71 },
  //{ filename: 'left_20241219_114333.jpg', dx: -32.99, dy: -39.81 },
  //{ filename: 'left_20241219_114335.jpg', dx: -72.88, dy: -23.30 },
  //{ filename: 'left_20241219_114336.jpg', dx: 16.47, dy: 14.57 },
  //{ filename: 'left_20241219_114337.jpg', dx: -66.36, dy: -22.38 },
  //{ filename: 'left_20241219_114338.jpg', dx: -12.62, dy: 15.27 },
  //{ filename: 'left_20241219_114340.jpg', dx: -68.39, dy: -17.63 },
  //{ filename: 'left_20241219_114341.jpg', dx: -64.25, dy: -13.84 },
  //{ filename: 'left_20241219_114342.jpg', dx: 0, dy: 0 },
  //{ filename: 'left_20241219_114343.jpg', dx: -68.14, dy: -23.11 },
  //{ filename: 'left_20241219_114345.jpg', dx: -83.02, dy: -22.41 },
  //{ filename: 'left_20241219_114346.jpg', dx: -16.69, dy: -3.11 },
  //{ filename: 'left_20241219_114347.jpg', dx: -88.54, dy: -6.09 },
  //{ filename: 'left_20241219_114348.jpg', dx: -110.16, dy: -23.39 },
  //{ filename: 'left_20241219_114349.jpg', dx: 12.11, dy: 38.70 },
  // { filename: 'left_20241219_114351.jpg', dx: -62.50, dy: -2.81 },
  //{ filename: 'left_20241219_114352.jpg', dx: -110.61, dy: -13.17 },
  //{ filename: 'left_20241219_114353.jpg', dx: -30.15, dy: 0.62 },
  //{ filename: 'left_20241219_114354.jpg', dx: -45.59, dy: -28.42 },
  // { filename: 'left_20241219_114356.jpg', dx: -127.01, dy: -4.18 },
  //{ filename: 'left_20241219_114357.jpg', dx: -0.12, dy: -29.26 },
  //{ filename: 'left_20241219_114358.jpg', dx: -76.21, dy: -9.96 },
  //{ filename: 'left_20241219_114359.jpg', dx: -153.54, dy: -26.94 },
  //{ filename: 'left_20241219_114401.jpg', dx: -63.78, dy: -16.74 },
  //{ filename: 'left_20241219_114402.jpg', dx: 79.41, dy: 21.44 },
  //{ filename: 'left_20241219_114403.jpg', dx: -66.49, dy: -12.54 },
  //{ filename: 'left_20241219_114404.jpg', dx: -22.49, dy: -6.05 },
  //{ filename: 'left_20241219_114406.jpg', dx: -41.09, dy: -23.54 },
  //{ filename: 'left_20241219_114407.jpg', dx: -69.53, dy: -18.17 },
  //{ filename: 'left_20241219_114408.jpg', dx: -63.83, dy: -14.66 },
  //{ filename: 'left_20241219_114409.jpg', dx: -71.35, dy: -8.76 },
  //{ filename: 'left_20241219_114411.jpg', dx: -93.96, dy: -23.74 },
  //{ filename: 'left_20241219_114412.jpg', dx: -51.44, dy: -7.57 },
  //{ filename: 'left_20241219_114413.jpg', dx: -122.79, dy: -17.78 },
  //{ filename: 'left_20241219_114414.jpg', dx: 1.95, dy: -0.95 },
  //{ filename: 'left_20241219_114416.jpg', dx: -80.39, dy: -24.22 },
  //{ filename: 'left_20241219_114417.jpg', dx: 75.91, dy: 14.56 },
  //{ filename: 'left_20241219_114418.jpg', dx: -108.06, dy: -32.35 },
];

// --- 2. データ処理 ---
// ---改良版を下に書いたため、コメントアウト---
const processImagesForPanorama = (logs: typeof RAW_LOGS, width: number) => {
  const result = [];
  let accumulatedX = 0;
  for (let i = 0; i < logs.length; i++) {
    const log = logs[i];
   
    result.push({
      id: i,
      url: `/images/left/${log.filename}`,
      position: [accumulatedX, 0, 0] as [number, number, number],
      filename: log.filename
    });
    //2枚目以降の位置決定のために累積
    //dxが０のときは等間隔に配置されるように設定
    const stepX = log.dx ! ? log.dx : width;
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

const ImagePanel = ({ url, position, index }: { 
  url: string, 
  position: [number, number, number], 
  index: number
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

    //カメラの正面（distanceX）に近いほど手前に、離れるほど奥に
    //二次関数を用いてなめらかなカーブを作る
    const dynamicZ = -Math.pow(distanceX, 2) * CURVE_FACTOR;
    groupRef.current.position.setZ(dynamicZ);

    //Z軸に下がるだけで小さく見えるが、より強調したい時に有効化する
    // const dynamicScale = 1 / (1 + absDistanceX * 0.0005);
    // groupRef.current.scale.set(dynamicScale, dynamicScale, 1);


    // 距離に応じた目標透明度を決定,正面付近ならくっきり、離れると薄く
    // 瞬時に切り替えたい場合は meshRef.current.material.opacity = targetOpacity; だけでもOK
    const targetOpacity = absDistanceX < (IMAGE_WIDTH * 0.6) ? OPACITY_VISIBLE : OPACITY_INVISIBLE;
    const material = meshRef.current.material as THREE.Material;
    material.opacity = THREE.MathUtils.lerp(material.opacity, targetOpacity, 0.1);
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
// --- 4. メインコンポーネント ---

export default function PanoramaView() {
  const processedData = useMemo(() => processImagesForPanorama(RAW_LOGS, IMAGE_WIDTH), []);
  
  // 初期位置（最初の画像がある場所。今回のロジックでは必ず 0）
  const initialX = processedData[0]?.position[0] || 0;

  // 目標位置（Target）を管理するState。
  // カメラの実際の座標（Current）はCameraRig内でLerp計算される。
  // カメラの移動目標（初期値は1枚目の座標 = 0）
  const [targetCameraX, setTargetCameraX] = useState(initialX);

  // 🟢 インデックスベースで移動を制御するためのヘルパー（境界を超えない防衛策）
  const currentImageIdx = processedData.findIndex(d => d.position[0] === targetCameraX);

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

      <Canvas>
        <PerspectiveCamera 
          makeDefault 
          // 初期位置はminXに設定
          position={[initialX, 0, IMAGE_WIDTH * 1.2]} 
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

        {/* <Grid 
          position={[centerX, -IMAGE_HEIGHT / 2 - 200, 0]} 
          args={[totalWidth * 2, IMAGE_HEIGHT * 2]} 
          cellColor="#333" 
          sectionColor="#555" 
          infiniteGrid 
          fadeDistance={50000}
          cellSize={100} 
          sectionSize={1000} 
        /> */}

        {processedData.map((img, idx) => (
          <ImagePanel 
            key={img.id} 
            index={idx}
            url={img.url} 
            position={img.position}
            // ImagePanel自体がuseFrameでカメラ位置を取得するため、currentCameraXのProps渡しは不要
          />
        ))}
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