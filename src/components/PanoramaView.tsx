import React, { useMemo, useRef, useState, useEffect, Suspense } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { Image as DreiImage, OrbitControls, PerspectiveCamera, Grid, Line, Text } from '@react-three/drei';
import * as THREE from 'three';
import type { DetectionBox, DetectionsJson, ImageItem } from '../types/panorama';
import { CSV_OPTIONS } from '../App';

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

  useEffect(() => {
    if (navTrigger !== lastTrigger.current) {
      lastTrigger.current = navTrigger;
      isTransitioning.current = true;
    }
  }, [navTrigger]);

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

        const labelText =
          box.rawLabel && box.rawLabel !== box.label
            ? `#${idx + 1} ${box.label}(${box.rawLabel}) ${(box.score * 100).toFixed(1)}%`
            : `#${idx + 1} ${box.label} ${(box.score * 100).toFixed(1)}%`;

        return (
          <group key={idx}>
            <Line points={rectPoints} color="#ff1a1a" lineWidth={2.5} />
            <Text
              position={[left, top + 0.06, 0.01]}
              fontSize={0.085}
              color="#ff3333"
              anchorX="left"
              anchorY="bottom"
              outlineWidth={0.01}
              outlineColor="#000000"
            >
              {labelText}
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
  originalUrl,
  position,
  index,
  detections = [],
  showDetections = true,
  showOriginalBelow = false,
}: {
  url: string;
  originalUrl: string;
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

    const topMat = topMeshRef.current.material;
    if (topMat && 'opacity' in topMat) {
      (topMat as THREE.Material & { opacity: number }).opacity = THREE.MathUtils.lerp(
        (topMat as THREE.Material & { opacity: number }).opacity,
        targetOpacity,
        0.1
      );
    }

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
  // 「元画像のみ」モードの場合は上段も left の元画像を表示
  const topImageUrl = showDetections ? url : originalUrl;

  return (
    <group ref={groupRef} position={[position[0], position[1], position[2]]} renderOrder={index}>
      <group position={[0, 0, 0]}>
        <DreiImage
          ref={topMeshRef}
          url={topImageUrl}
          scale={[IMAGE_WIDTH, IMAGE_HEIGHT]}
          transparent
          opacity={OPACITY_INVISIBLE}
          side={THREE.DoubleSide}
        />
        {showDetections && <TomatoBoundingBoxes detections={detections} />}
      </group>

      {showOriginalBelow && (
        <group position={[0, bottomOffsetY, 0]}>
          <DreiImage
            ref={bottomMeshRef}
            url={originalUrl}
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
  background: 'rgba(0, 0, 0, 0.82)',
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
  background: 'rgba(0, 0, 0, 0.85)',
  padding: '14px',
  borderRadius: '12px',
  border: '1px solid #319795',
  color: '#e2e8f0',
  width: '380px',
  fontFamily: 'monospace',
  fontSize: '11px',
};

interface PanoramaViewProps {
  images: ImageItem[];
  detectionData: DetectionsJson | null;
  errorText: string;
  selectedCsv: string;
  onSelectCsv: (csv: string) => void;
}

export function PanoramaView({
  images,
  detectionData,
  errorText,
  selectedCsv,
  onSelectCsv,
}: PanoramaViewProps) {
  const initialPos = images[0]?.position || [0, 0, 0];

  const [targetCameraPos, setTargetCameraPos] = useState<[number, number]>([initialPos[0], initialPos[1]]);
  const [navTrigger, setNavTrigger] = useState(0);
  const [displayMode, setDisplayMode] = useState<DisplayMode>('detection_only');
  const [confThreshold, setConfThreshold] = useState(DEFAULT_CONFIDENCE_THRESHOLD);
  const [showDetailsPanel, setShowDetailsPanel] = useState(true);
  const [leftTab, setLeftTab] = useState<'current' | 'logs'>('current');

  const logContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logContainerRef.current && leftTab === 'logs') {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [detectionData?.logs, leftTab]);

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

  // 左矢印（←）：画面左側の画像（前の画像）へ移動
  const moveLeft = () => {
    if (currentImageIdx > 0) {
      const prev = images[currentImageIdx - 1];
      setTargetCameraPos([prev.position[0], prev.position[1]]);
      setNavTrigger((n) => n + 1);
    }
  };

  // 右矢印（→）：画面右側の画像（次の画像）へ移動
  const moveRight = () => {
    if (currentImageIdx < images.length - 1) {
      const next = images[currentImageIdx + 1];
      setTargetCameraPos([next.position[0], next.position[1]]);
      setNavTrigger((n) => n + 1);
    }
  };

  const handleChangeDisplayMode = (mode: DisplayMode) => {
    setDisplayMode(mode);
    setNavTrigger((n) => n + 1);
  };

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

  const currentImage = images[currentImageIdx];
  const currentBoxes = currentImage ? filteredDetectionsMap[currentImage.filename] || [] : [];
  const currentDetectionsCount = currentBoxes.length;

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

  const showDetections = displayMode !== 'original_only';
  const showOriginalBelow = displayMode === 'compare';

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

      {/* 左上: 検出データ詳細 ＆ 実行ログパネル */}
      <div style={progressPanelStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontWeight: 'bold', color: isRunning ? '#f6e05e' : '#4fd1c5', fontSize: '12px' }}>
            {isRunning ? `● 検出処理中 (${progressPercent}%)` : `✓ YOLO検出データ (${progressPercent}%)`}
          </span>
          <button
            onClick={() => setShowDetailsPanel(!showDetailsPanel)}
            style={{
              background: '#2d3748',
              color: '#fff',
              border: '1px solid #4a5568',
              borderRadius: '4px',
              padding: '2px 8px',
              cursor: 'pointer',
              fontSize: '10px',
            }}
          >
            {showDetailsPanel ? 'パネルを隠す' : '詳細を表示'}
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
          <div style={{ marginBottom: '8px', color: '#cbd5e0' }}>
            全体: {processedImages}/{totalImages}枚 ｜ 反応: {detectionData.detectedImagesCount}枚 (計 {detectionData.totalBoxesCount}個)
          </div>
        )}

        {showDetailsPanel && (
          <>
            {/* タブ切り替えボタン */}
            <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
              <button
                onClick={() => setLeftTab('current')}
                style={{
                  flex: 1,
                  padding: '5px 8px',
                  borderRadius: '6px',
                  border: '1px solid #4fd1c5',
                  background: leftTab === 'current' ? '#285e61' : '#1a202c',
                  color: '#fff',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  fontSize: '11px',
                }}
              >
                現在の画像の検出詳細 ({currentDetectionsCount}個)
              </button>
              <button
                onClick={() => setLeftTab('logs')}
                style={{
                  flex: 1,
                  padding: '5px 8px',
                  borderRadius: '6px',
                  border: '1px solid #4a5568',
                  background: leftTab === 'logs' ? '#2d3748' : '#1a202c',
                  color: '#cbd5e0',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  fontSize: '11px',
                }}
              >
                全体ログ ({detectionData?.logs?.length || 0}行)
              </button>
            </div>

            {leftTab === 'current' ? (
              /* 現在表示中の画像の各ボックス詳細 */
              <div
                style={{
                  maxHeight: '220px',
                  overflowY: 'auto',
                  background: 'rgba(0, 0, 0, 0.7)',
                  padding: '8px',
                  borderRadius: '6px',
                  border: '1px solid #2d3748',
                }}
              >
                <div style={{ color: '#81e6d9', marginBottom: '6px', fontWeight: 'bold' }}>
                  対象: {currentImage?.filename || '-'} (しきい値 {Math.round(confThreshold * 100)}% 以上)
                </div>

                {currentBoxes.length === 0 ? (
                  <div style={{ color: '#a0aec0', padding: '6px 0' }}>
                    この画像で検出されたトマトはありません
                  </div>
                ) : (
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid #4a5568', color: '#a0aec0', textAlign: 'left' }}>
                        <th style={{ padding: '4px 2px' }}>No</th>
                        <th style={{ padding: '4px 2px' }}>ラベル</th>
                        <th style={{ padding: '4px 2px' }}>類似度</th>
                        <th style={{ padding: '4px 2px' }}>大きさ(幅×高)</th>
                        <th style={{ padding: '4px 2px' }}>中心位置</th>
                      </tr>
                    </thead>
                    <tbody>
                      {currentBoxes.map((box, idx) => {
                        const widthPct = ((box.x2 - box.x1) * 100).toFixed(1);
                        const heightPct = ((box.y2 - box.y1) * 100).toFixed(1);
                        const centerX = (((box.x1 + box.x2) / 2) * 100).toFixed(0);
                        const centerY = (((box.y1 + box.y2) / 2) * 100).toFixed(0);

                        return (
                          <tr key={idx} style={{ borderBottom: '1px solid #2d3748' }}>
                            <td style={{ padding: '5px 2px', color: '#fc8181', fontWeight: 'bold' }}>#{idx + 1}</td>
                            <td style={{ padding: '5px 2px', color: '#fff' }}>
                              {box.label}
                              {box.rawLabel && box.rawLabel !== box.label && (
                                <span style={{ color: '#90cdf4', fontSize: '10px' }}> ({box.rawLabel})</span>
                              )}
                            </td>
                            <td style={{ padding: '5px 2px', color: '#68d391', fontWeight: 'bold' }}>
                              {(box.score * 100).toFixed(1)}%
                            </td>
                            <td style={{ padding: '5px 2px', color: '#cbd5e0' }}>
                              {widthPct}% × {heightPct}%
                            </td>
                            <td style={{ padding: '5px 2px', color: '#cbd5e0' }}>
                              ({centerX}%, {centerY}%)
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                )}
              </div>
            ) : (
              /* 全体実行ログ */
              <div
                ref={logContainerRef}
                style={{
                  maxHeight: '220px',
                  overflowY: 'auto',
                  overflowX: 'auto',
                  background: 'rgba(0, 0, 0, 0.7)',
                  padding: '8px',
                  borderRadius: '6px',
                  border: '1px solid #2d3748',
                  lineHeight: '1.5',
                }}
              >
                {detectionData?.logs && detectionData.logs.length > 0 ? (
                  detectionData.logs.map((log, i) => (
                    <div key={i} style={{ whiteSpace: 'nowrap' }}>
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
          </>
        )}
      </div>

      {/* 右上: 表示コントロール */}
      <div style={controlPanelStyle}>
        <div style={{ fontWeight: 'bold', marginBottom: '10px', color: '#00ffcc' }}>Display Controls</div>

        {/* CSVデータセット切り替え用ミニヘッダー */}
        <div
          style={{
            marginBottom: '12px',
            paddingBottom: '10px',
            borderBottom: '1px solid #444',
          }}
        >
          <div style={{ fontSize: '10px', color: '#a0aec0', marginBottom: '6px' }}>
            CSVデータセット ({selectedCsv})
          </div>
          <div
            style={{
              display: 'flex',
              background: '#1a202c',
              borderRadius: '6px',
              padding: '2px',
              border: '1px solid #4a5568',
              gap: '2px',
            }}
          >
            {CSV_OPTIONS.map((opt) => {
              const isSelected = selectedCsv === opt.id;
              return (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => onSelectCsv(opt.id)}
                  style={{
                    flex: 1,
                    padding: '4px 0',
                    fontSize: '11px',
                    fontWeight: isSelected ? 'bold' : 'normal',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    background: isSelected ? '#319795' : 'transparent',
                    color: isSelected ? '#fff' : '#cbd5e0',
                    transition: 'all 0.15s',
                  }}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* 表示モード選択ラジオボタン */}
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
              key={`${selectedCsv}-${img.id}`}
              index={idx}
              url={img.url}
              originalUrl={img.originalUrl || img.url}
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