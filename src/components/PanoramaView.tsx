import React, { useMemo, useRef, useState, useEffect, Suspense } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { Image as DreiImage, OrbitControls, PerspectiveCamera, Grid, Line, Text } from '@react-three/drei';
import * as THREE from 'three';
import type { DetectionBox, DetectionsJson, ImageItem } from '../types/panorama';

const BUILD_TIME = new Date().toLocaleString('ja-JP', {
  timeZone: 'Asia/Tokyo',
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
});

export const IMAGE_WIDTH = 3.2;
export const IMAGE_HEIGHT = 2.4;
const ROW_GAP = 0.25; // 上段（検出画像）と下段（元画像）の隙間

const OPACITY_VISIBLE = 0.95;
const OPACITY_INVISIBLE = 0.5;
const CURVE_FACTOR = 0.000015;
const DEFAULT_CONFIDENCE_THRESHOLD = 0.25;

type DisplayMode = 'detection_only' | 'compare' | 'original_only';

// カメラのスムーズな追従制御（ユーザー操作時は妨げない）
const CameraRig = ({
  targetX,
  targetY,
  targetZ,
  navTrigger,
}: {
  targetX: number;
  targetY: number;
  targetZ: number;
  navTrigger: number;
}) => {
  const { camera, controls } = useThree();
  const isTransitioning = useRef(false);
  const lastTrigger = useRef(navTrigger);

  // 矢印移動・表示モード切替・視点リセットが発生したときだけ自動移動をONにする
  useEffect(() => {
    if (navTrigger !== lastTrigger.current) {
      lastTrigger.current = navTrigger;
      isTransitioning.current = true;
    }
  }, [navTrigger]);

  // ユーザーがマウスホイールやドラッグ操作を始めたら自動移動を即座に解除
  useEffect(() => {
    const orbit = controls as unknown as {
      addEventListener?: (event: string, listener: () => void) => void;
      removeEventListener?: (event: string, listener: () => void) => void;
    } | null;

    if (!orbit || !orbit.addEventListener || !orbit.removeEventListener) return;

    const handleUserInteract = () => {
      isTransitioning.current = false;
    };

    orbit.addEventListener('start', handleUserInteract);
    return () => {
      orbit.removeEventListener?.('start', handleUserInteract);
    };
  }, [controls]);

  useFrame((_, delta) => {
    if (!isTransitioning.current) return;

    const dampSpeed = Math.min(1, 6 * delta);
    const orbit = controls as unknown as { target: THREE.Vector3; update: () => void } | null;

    const nextX = THREE.MathUtils.lerp(camera.position.x, targetX, dampSpeed);
    const nextY = THREE.MathUtils.lerp(camera.position.y, targetY, dampSpeed);
    const nextZ = THREE.MathUtils.lerp(camera.position.z, targetZ, dampSpeed);
    camera.position.set(nextX, nextY, nextZ);

    if (orbit) {
      const nextTargetX = THREE.MathUtils.lerp(orbit.target.x, targetX, dampSpeed);
      const nextTargetY = THREE.MathUtils.lerp(orbit.target.y, targetY, dampSpeed);
      const nextTargetZ = THREE.MathUtils.lerp(orbit.target.z, 0, dampSpeed);
      orbit.target.set(nextTargetX, nextTargetY, nextTargetZ);
      orbit.update();
    }

    // 目標位置に十分近づいたら自動追従を終了し、自由操作状態にする
    const dist =
      Math.abs(camera.position.x - targetX) +
      Math.abs(camera.position.y - targetY) +
      Math.abs(camera.position.z - targetZ);

    if (dist < 0.01) {
      camera.position.set(targetX, targetY, targetZ);
      if (orbit) {
        orbit.target.set(targetX, targetY, 0);
        orbit.update();
      }
      isTransitioning.current = false;
    }
  });

  return null;
};

// 検出されたトマトのバウンディングボックス描画
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
          [left, top, 0],
        ];

        return (
          <group key={idx}>
            <Line points={rectPoints} color="#ff1a1a" lineWidth={2.5} />
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

// 3D空間上の各画像パネル（上段：メイン画像 / 下段：比較用元画像）
const ImagePanel = ({
  url,
  position,
  index,
  detections = [],
  showDetections = true,
  showOriginalBelow = false,
}: {
  url: string;
  position: [number, number, number];
  index: number;
  detections?: DetectionBox[];
  showDetections?: boolean;
  showOriginalBelow?: boolean;
}) => {
  const groupRef = useRef<THREE.Group>(null!);
  const topMeshRef = useRef<THREE.Mesh>(null!);
  const bottomMeshRef = useRef<THREE.Mesh>(null!);

  useFrame(({ camera }) => {
    if (!groupRef.current || !topMeshRef.current) return;

    const distanceX = Math.abs(position[0] - camera.position.x);
    const dynamicZ = -Math.pow(distanceX, 2) * CURVE_FACTOR;

    groupRef.current.position.set(position[0], position[1], dynamicZ);

    const isFront = distanceX < IMAGE_WIDTH * 0.5;
    groupRef.current.renderOrder = isFront ? 999 : index;

    const targetOpacity = distanceX < IMAGE_WIDTH * 0.6 ? OPACITY_VISIBLE : OPACITY_INVISIBLE;

    // 上段の透明度アニメーション
    const topMat = topMeshRef.current.material;
    if (topMat && 'opacity' in topMat) {
      (topMat as THREE.Material & { opacity: number }).opacity = THREE.MathUtils.lerp(
        (topMat as THREE.Material & { opacity: number }).opacity,
        targetOpacity,
        0.1
      );
    }

    // 下段（元画像）の透明度アニメーション
    if (bottomMeshRef.current) {
      const bottomMat = bottomMeshRef.current.material;
      if (bottomMat && 'opacity' in bottomMat) {
        (bottomMat as THREE.Material & { opacity: number }).opacity = THREE.MathUtils.lerp(
          (bottomMat as THREE.Material & { opacity: number }).opacity,
          targetOpacity,
          0.1
        );
      }
    }
  });

  const bottomOffsetY = -(IMAGE_HEIGHT + ROW_GAP);

  return (
    <group ref={groupRef} position={[position[0], position[1], position[2]]} renderOrder={index}>
      {/* 上段: メイン画像（検出モードまたは比較モード時は赤枠あり、元画像のみモード時は赤枠なし） */}
      <group position={[0, 0, 0]}>
        <DreiImage
          ref={topMeshRef}
          url={url}
          scale={[IMAGE_WIDTH, IMAGE_HEIGHT]}
          transparent
          opacity={OPACITY_INVISIBLE}
          side={THREE.DoubleSide}
        />
        {showDetections && <TomatoBoundingBoxes detections={detections} />}
      </group>

      {/* 下段: 比較用の元画像（枠なし・比較モードのときのみ描画） */}
      {showOriginalBelow && (
        <group position={[0, bottomOffsetY, 0]}>
          <DreiImage
            ref={bottomMeshRef}
            url={url}
            scale={[IMAGE_WIDTH, IMAGE_HEIGHT]}
            transparent
            opacity={OPACITY_INVISIBLE}
            side={THREE.DoubleSide}
          />
        </group>
      )}
    </group>
  );
};

// --- スタイル定義 ---
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
  border: '1px solid #444',
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
};

const labelStyle: React.CSSProperties = {
  color: '#fff',
  fontFamily: 'monospace',
  textAlign: 'center',
  minWidth: '140px',
};

const controlPanelStyle: React.CSSProperties = {
  position: 'absolute',
  top: '55px',
  right: '20px',
  zIndex: 100,
  background: 'rgba(0, 0, 0, 0.8)',
  padding: '16px',
  borderRadius: '12px',
  border: '1px solid #444',
  color: '#fff',
  width: '260px',
  fontFamily: 'sans-serif',
  fontSize: '12px',
};

const resetButtonStyle: React.CSSProperties = {
  width: '100%',
  marginTop: '14px',
  padding: '8px 12px',
  background: '#2c5282',
  color: '#fff',
  border: '1px solid #4299e1',
  borderRadius: '6px',
  fontWeight: 'bold',
  fontSize: '12px',
  cursor: 'pointer',
  transition: 'background 0.2s',
};

const progressPanelStyle: React.CSSProperties = {
  position: 'absolute',
  top: '55px',
  left: '20px',
  zIndex: 100,
  background: 'rgba(0, 0, 0, 0.82)',
  padding: '14px',
  borderRadius: '12px',
  border: '1px solid #319795',
  color: '#e2e8f0',
  width: '340px',
  fontFamily: 'monospace',
  fontSize: '11px',
};

interface PanoramaViewProps {
  images: ImageItem[];
  detectionData: DetectionsJson | null;
  errorText: string;
}

export function PanoramaView({ images, detectionData, errorText }: PanoramaViewProps) {
  const initialPos = images[0]?.position || [0, 0, 0];

  const [targetCameraPos, setTargetCameraPos] = useState<[number, number]>([initialPos[0], initialPos[1]]);
  const [navTrigger, setNavTrigger] = useState(0);
  const [displayMode, setDisplayMode] = useState<DisplayMode>('detection_only');
  const [confThreshold, setConfThreshold] = useState(DEFAULT_CONFIDENCE_THRESHOLD);
  const [showLogs, setShowLogs] = useState(true);

  const showDetections = displayMode !== 'original_only';
  const showOriginalBelow = displayMode === 'compare';

  const logContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [detectionData?.logs]);

  useEffect(() => {
    if (images.length > 0) {
      setTargetCameraPos([images[0].position[0], images[0].position[1]]);
      setNavTrigger((prev) => prev + 1);
    }
  }, [images]);

  const currentImageIdx = useMemo(() => {
    const idx = images.findIndex((d) => Math.abs(d.position[0] - targetCameraPos[0]) < 0.001);
    return idx === -1 ? 0 : idx;
  }, [images, targetCameraPos]);

  // 左矢印（←）：画面左側の画像へ移動（X座標がマイナス方向のためインデックスを増やす）
  const moveLeft = () => {
    if (currentImageIdx < images.length - 1) {
      const next = images[currentImageIdx + 1];
      setTargetCameraPos([next.position[0], next.position[1]]);
      setNavTrigger((n) => n + 1);
    }
  };

  // 右矢印（→）：画面右側の画像へ移動（X座標がプラス方向のためインデックスを減らす）
  const moveRight = () => {
    if (currentImageIdx > 0) {
      const prev = images[currentImageIdx - 1];
      setTargetCameraPos([prev.position[0], prev.position[1]]);
      setNavTrigger((n) => n + 1);
    }
  };

  const handleChangeDisplayMode = (mode: DisplayMode) => {
    setDisplayMode(mode);
    setNavTrigger((n) => n + 1);
  };

  // 現在選択中の画像の正面・初期距離にカメラ視点をリセット
  const handleResetView = () => {
    const currentImg = images[currentImageIdx];
    if (currentImg) {
      setTargetCameraPos([currentImg.position[0], currentImg.position[1]]);
    }
    setNavTrigger((n) => n + 1);
  };

  const filteredDetectionsMap = useMemo(() => {
    const result: Record<string, DetectionBox[]> = {};
    const detections = detectionData?.detections || {};
    for (const [filename, boxes] of Object.entries(detections)) {
      result[filename] = boxes.filter((b) => b.score >= confThreshold);
    }
    return result;
  }, [detectionData, confThreshold]);

  const currentDetectionsCount = filteredDetectionsMap[images[currentImageIdx]?.filename]?.length || 0;

  const totalImages = detectionData?.totalImages || images.length || 0;
  const processedImages = detectionData?.processedImages ?? totalImages;
  const progressPercent = totalImages > 0 ? Math.round((processedImages / totalImages) * 100) : 0;
  const isRunning = detectionData?.status === 'running';

  const headerStatus = errorText
    ? errorText
    : isRunning
    ? `YOLO26 検出実行中... (${processedImages}/${totalImages}枚)`
    : detectionData
    ? `事前検出データ読込済 (${detectionData.detectedImagesCount}/${totalImages}枚で反応 / 計${detectionData.totalBoxesCount}個)`
    : '検出データ待機中...';

  const verticalCenterOffset = showOriginalBelow ? -(IMAGE_HEIGHT + ROW_GAP) / 2 : 0;
  const targetCameraZ = showOriginalBelow ? IMAGE_WIDTH * 2.1 : IMAGE_WIDTH * 1.5;
  const gridY = showOriginalBelow
    ? -IMAGE_HEIGHT * 1.5 - ROW_GAP - 0.2
    : -IMAGE_HEIGHT / 2 - 0.2;

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#000', overflow: 'hidden', position: 'relative' }}>
      {/* ヘッダー */}
      <header
        style={{
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
          fontFamily: 'sans-serif',
        }}
      >
        <p style={{ margin: 0, fontSize: '14px', fontWeight: 'bold' }}>
          検証用ビルド日時: {BUILD_TIME} ｜ {headerStatus}
        </p>
      </header>

      {/* 左上: YOLO検出の途中経過＆結果ログパネル */}
      <div style={progressPanelStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontWeight: 'bold', color: isRunning ? '#f6e05e' : '#4fd1c5', fontSize: '12px' }}>
            {isRunning ? `● 検出処理中 (${progressPercent}%)` : `✓ 検出ログ (${progressPercent}%)`}
          </span>
          <button
            onClick={() => setShowLogs(!showLogs)}
            style={{
              background: '#2d3748',
              color: '#fff',
              border: '1px solid #4a5568',
              borderRadius: '4px',
              padding: '2px 6px',
              cursor: 'pointer',
              fontSize: '10px',
            }}
          >
            {showLogs ? 'ログを隠す' : 'ログを表示'}
          </button>
        </div>

        {/* プログレスバー */}
        <div style={{ width: '100%', height: '6px', background: '#2d3748', borderRadius: '3px', overflow: 'hidden', marginBottom: '8px' }}>
          <div
            style={{
              width: `${progressPercent}%`,
              height: '100%',
              background: errorText ? '#f56565' : isRunning ? '#ecc94b' : '#38b2ac',
              transition: 'width 0.3s ease',
            }}
          />
        </div>

        {detectionData && (
          <div style={{ marginBottom: '6px', color: '#cbd5e0' }}>
            進捗: {processedImages} / {totalImages} 枚 ｜ 反応画像: {detectionData.detectedImagesCount} 枚 (計 {detectionData.totalBoxesCount} 個)
          </div>
        )}

        {showLogs && (
          <div
            ref={logContainerRef}
            style={{
              maxHeight: '160px',
              overflowY: 'auto',
              background: 'rgba(0, 0, 0, 0.65)',
              padding: '6px',
              borderRadius: '6px',
              border: '1px solid #2d3748',
              lineHeight: '1.4',
            }}
          >
            {detectionData?.logs && detectionData.logs.length > 0 ? (
              detectionData.logs.map((log, i) => (
                <div key={i} style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {log}
                </div>
              ))
            ) : (
              <div style={{ color: '#a0aec0' }}>
                {errorText || 'ログデータがありません'}
              </div>
            )}
          </div>
        )}
      </div>

      {/* 右上: 表示コントロール */}
      <div style={controlPanelStyle}>
        <div style={{ fontWeight: 'bold', marginBottom: '10px', color: '#00ffcc' }}>Display Controls</div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '12px' }}>
          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              cursor: 'pointer',
              color: '#ff6666',
              fontWeight: 'bold',
            }}
          >
            <input
              type="radio"
              name="displayMode"
              checked={displayMode === 'detection_only'}
              onChange={() => handleChangeDisplayMode('detection_only')}
            />
            検出画像のみ（赤枠あり）
          </label>

          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              cursor: 'pointer',
              color: '#63b3ed',
              fontWeight: 'bold',
            }}
          >
            <input
              type="radio"
              name="displayMode"
              checked={displayMode === 'compare'}
              onChange={() => handleChangeDisplayMode('compare')}
            />
            上下で比較（上:検出 / 下:元画像）
          </label>

          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              cursor: 'pointer',
              color: '#e2e8f0',
            }}
          >
            <input
              type="radio"
              name="displayMode"
              checked={displayMode === 'original_only'}
              onChange={() => handleChangeDisplayMode('original_only')}
            />
            元画像のみ（赤枠なし）
          </label>
        </div>

        {showDetections && (
          <div>
            <label style={{ display: 'block', marginBottom: '4px', color: '#ccc' }}>
              Confidence Threshold: {Math.round(confThreshold * 100)}%
            </label>
            <input
              type="range"
              min="0.15"
              max="0.80"
              step="0.05"
              value={confThreshold}
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>
        )}

        <button
          type="button"
          onClick={handleResetView}
          style={resetButtonStyle}
          onMouseEnter={(e) => (e.currentTarget.style.background = '#3182ce')}
          onMouseLeave={(e) => (e.currentTarget.style.background = '#2c5282')}
        >
          視点をリセット (Reset View)
        </button>
      </div>

      {/* 3D空間キャンバス */}
      <Canvas>
        <PerspectiveCamera
          makeDefault
          position={[initialPos[0], initialPos[1] + verticalCenterOffset, targetCameraZ]}
          fov={60}
          far={100000}
        />

        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.1}
          target={[initialPos[0], initialPos[1] + verticalCenterOffset, 0]}
          enableZoom={true}
          enableRotate={true}
          enablePan={true}
          minDistance={0.5}
          maxDistance={50}
        />

        <CameraRig
          targetX={targetCameraPos[0]}
          targetY={targetCameraPos[1] + verticalCenterOffset}
          targetZ={targetCameraZ}
          navTrigger={navTrigger}
        />

        <color attach="background" args={['#111']} />
        <ambientLight intensity={2} />

        <Grid
          position={[initialPos[0], gridY, 0]}
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
          {images.map((img, idx) => (
            <ImagePanel
              key={img.id}
              index={idx}
              url={img.url}
              position={img.position}
              detections={filteredDetectionsMap[img.filename]}
              showDetections={showDetections}
              showOriginalBelow={showOriginalBelow}
            />
          ))}
        </Suspense>
      </Canvas>

      {/* 下部ナビゲーションUI */}
      <div style={uiContainerStyle}>
        <button style={arrowButtonStyle} onClick={moveLeft}>
          ←
        </button>
        <div style={labelStyle}>
          Image: {currentImageIdx + 1} / {images.length} <br />
          <span style={{ fontSize: '11px', color: '#ff6666' }}>
            検出トマト数: {currentDetectionsCount}個
          </span>
        </div>
        <button style={arrowButtonStyle} onClick={moveRight}>
          →
        </button>
      </div>
    </div>
  );
}