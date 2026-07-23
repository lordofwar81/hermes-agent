import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import { RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import type { MemoryStats, MemoryGalaxy, MemoryGalaxyPoint } from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";

// ── 3D Galaxy ────────────────────────────────────────────────────────────────
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";

// ═══════════════════════════════════════════════════════════════════════════════
// THE PERSISTENCE OF MEMORY — Interstellar-inspired observatory palette.
//
// The thesis: deep void, a galaxy of varying-brightness facts, warm amber
// reserved exclusively for high-trust so they burn through the cool field
// like Gargantua's accretion disk. Every warm pixel in the page is earned.
// ═══════════════════════════════════════════════════════════════════════════════
const C = {
  void:       "#05070D",  // the space outside the window — deeper than generic #0A0A0A
  dust:       "#1A2138",  // faint blue-white of low-trust mass / hairlines
  star:       "#E8EDF5",  // warm white for mid-trust facts and body text
  accretion:  "#FFB347",  // THE signature — high-trust amber. Used nowhere else.
  signal:     "#5EEAD4",  // teal-cyan for active UI (hover, selection, links)
  readout:    "#64748B",  // TARS/CASE telemetry gray for labels
  readoutHi:  "#94A3B8",  // brighter readout for secondary text
  hairline:   "#131A2A",  // dividers — barely visible, like a scratch on glass
  panel:      "#0A0E1A",  // raised surface — one shade up from void
};

// Category tints — deliberately DESATURATED. 90% of facts are `conversation`,
// so it gets the dimmest treatment (near-invisible dust). Minority categories
// get faint tints so they read as structure emerging from the mass, not as a
// rainbow. The real visual hierarchy is trust→warmth, not category→hue.
const CATEGORY_TINT: Record<string, [number, number, number]> = {
  // [r, g, b] 0-1, will be dimmed by trust/retrieval encoding downstream
  conversation:  [0.30, 0.36, 0.50],  // cool slate-gray — the dust cloud
  zcode:         [0.37, 0.92, 0.83],  // faint teal
  user_pref:     [0.99, 0.70, 0.28],  // warm amber tint (these matter)
  infrastructure:[0.20, 0.83, 0.60],  // faint emerald
  project:       [0.65, 0.55, 0.98],  // faint violet
  health:        [0.98, 0.44, 0.52],  // faint rose
  opencode:      [0.98, 0.57, 0.24],  // faint orange
  "claude-code": [0.38, 0.64, 0.98],  // faint blue
  tool:          [0.13, 0.83, 0.93],  // faint cyan
  document:      [0.91, 0.47, 0.98],  // faint magenta
  general:       [0.58, 0.64, 0.72],  // neutral
  uncategorized: [0.39, 0.45, 0.55],  // neutral dim
};
const DUST_TINT: [number, number, number] = [0.30, 0.36, 0.50];

function categoryTint(cat: string): [number, number, number] {
  return CATEGORY_TINT[cat] ?? DUST_TINT;
}

// Display font: the page already loads "Collapse" (wide geometric sans).
// We reference it by family name; if unavailable, falls back to the monospace
// stack that the rest of the telemetry uses.
const DISPLAY_FONT = '"Collapse", "JetBrains Mono", ui-monospace, monospace';
const TELEMETRY_FONT = '"JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace';

const POLL_MS = 30_000;

// ═══════════════════════════════════════════════════════════════════════════════
// CAMERA INTRO — the "coming out of cryosleep" dolly. Starts at z=55, eases
// to z=22. The galaxy reveals as you approach, not all at once.
// ═══════════════════════════════════════════════════════════════════════════════
function CameraIntro() {
  const started = useRef(false);
  useFrame((state) => {
    if (!started.current) {
      started.current = true;
      state.camera.position.set(0, 0, 55);
    }
    const target = 22;
    if (state.camera.position.z > target) {
      // Slower ease than before — the reveal should take ~4 seconds
      state.camera.position.z = THREE.MathUtils.lerp(state.camera.position.z, target, 0.015);
    }
  });
  return null;
}

// ═══════════════════════════════════════════════════════════════════════════════
// THE GALAXY — a single Points cloud where each fact is a star.
//
// Visual encoding (data → light):
//   trust > 0.85  → full ACCRETION amber, additive glow (the 47 that matter)
//   trust 0.5–0.85 → warm white STAR, brightness scaled by retrieval
//   trust ≤ 0.5   → cool DUST, dim, barely there (the 1266 mass)
//   retrieval     → point size (log) + opacity — recalled facts burn brighter
//   category      → subtle hue tint on dust/stars (amber points keep amber)
//
// This replaces the old uniform-color-per-category approach where 90% of points
// were the same color. Now the galaxy has real topography: a dim dust cloud
// with brighter landmarks and a few amber giants.
// ═══════════════════════════════════════════════════════════════════════════════

// A circular sprite texture for soft round points (procedurally generated,
// no asset dependency). Sharp square points read as "data viz"; soft round
// points read as "stars".
function useStarTexture(): THREE.Texture {
  return useMemo(() => {
    const size = 64;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d")!;
    const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    grad.addColorStop(0.0, "rgba(255,255,255,1)");
    grad.addColorStop(0.2, "rgba(255,255,255,0.85)");
    grad.addColorStop(0.5, "rgba(255,255,255,0.25)");
    grad.addColorStop(1.0, "rgba(255,255,255,0)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);
    const tex = new THREE.CanvasTexture(canvas);
    tex.needsUpdate = true;
    return tex;
  }, []);
}

function GalaxyScene({
  points, selectedId, onSelect,
}: {
  points: MemoryGalaxyPoint[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
}) {
  const pointsRef = useRef<THREE.Points>(null);
  const { pointer, camera } = useThree();
  const starTexture = useStarTexture();
  const [hovered, setHovered] = useState<number | null>(null);
  const hoveredRef = useRef<number | null>(null);
  hoveredRef.current = hovered;

  // Respect reduced motion (kill the ambient rotation/pulse).
  const reducedMotion = useMemo(
    () => typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches,
    []
  );

  const raycaster = useMemo(() => {
    const r = new THREE.Raycaster();
    r.params.Points = { threshold: 0.5 };
    return r;
  }, []);

  // Build geometry: positions, per-point colors (trust→warmth), per-point sizes
  // (retrieval→magnitude), and per-point opacity.
  const { geometry, factIds } = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const n = points.length;
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
    const opacities = new Float32Array(n);
    const ids = new Int32Array(n);

    for (let i = 0; i < n; i++) {
      const p = points[i];
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
      ids[i] = p.fact_id;

      const trust = p.trust_score;
      const retr = p.retrieval_count;
      // Magnitude: log-scale retrieval so the top facts (600+) don't dwarf all.
      // Most facts have retr=0, so they get the minimum.
      const mag = Math.log(retr + 1) / Math.log(700); // 0..1 where 700 recalls = 1.0
      const tint = categoryTint(p.category);

      let r: number, g: number, b: number;
      if (trust > 0.85) {
        // ACCRETION — full amber, the signature warm points.
        r = 1.0; g = 0.70; b = 0.28;
      } else if (trust > 0.5) {
        // STAR — warm white tinted slightly by category.
        const w = (trust - 0.5) / 0.35; // 0..1 within the star band
        r = THREE.MathUtils.lerp(tint[0] * 0.6 + 0.4, 0.91, w);
        g = THREE.MathUtils.lerp(tint[1] * 0.6 + 0.4, 0.93, w);
        b = THREE.MathUtils.lerp(tint[2] * 0.6 + 0.4, 0.96, w);
      } else {
        // DUST — cool, dim, category-tinted. The mass.
        const d = trust / 0.5; // 0..1 within dust band
        r = tint[0] * (0.3 + d * 0.3);
        g = tint[1] * (0.3 + d * 0.3);
        b = tint[2] * (0.3 + d * 0.3);
      }
      colors[i * 3] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;

      // Size: base + magnitude contribution. High-trust gets a floor bonus.
      const baseSize = trust > 0.85 ? 0.45 : trust > 0.5 ? 0.25 : 0.12;
      sizes[i] = baseSize + mag * (trust > 0.85 ? 0.6 : 0.3);

      // Opacity: trust-scaled + magnitude boost. Dust is barely visible.
      const baseOp = trust > 0.85 ? 0.95 : trust > 0.5 ? 0.7 : 0.35;
      opacities[i] = Math.min(1, baseOp + mag * 0.3);
    }

    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.setAttribute("size", new THREE.BufferAttribute(sizes, 1));
    geo.setAttribute("opacity", new THREE.BufferAttribute(opacities, 1));
    return { geometry: geo, factIds: ids };
  }, [points]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  // Custom shader material for per-point size + opacity + round-star texture.
  // PointsMaterial can't do per-point size or opacity; this is minimal: just
  // enough to honor the size/opacity attributes and sample the star sprite.
  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        uTexture: { value: starTexture },
        uPixelRatio: { value: typeof window !== "undefined" ? window.devicePixelRatio : 1 },
      },
      vertexShader: `
        attribute float size;
        attribute float opacity;
        varying vec3 vColor;
        varying float vOpacity;
        uniform float uPixelRatio;
        void main() {
          vColor = color;
          vOpacity = opacity;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * 300.0 * uPixelRatio / -mvPosition.z;
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform sampler2D uTexture;
        varying vec3 vColor;
        varying float vOpacity;
        void main() {
          vec4 tex = texture2D(uTexture, gl_PointCoord);
          gl_FragColor = vec4(vColor, tex.a * vOpacity);
          if (gl_FragColor.a < 0.01) discard;
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexColors: true,
    });
  }, [starTexture]);

  useEffect(() => () => material.dispose(), [material]);

  // Slow ambient rotation + amber-point pulse (unless reduced motion).
  useFrame((_, delta) => {
    if (!reducedMotion && pointsRef.current) {
      pointsRef.current.rotation.y += delta * 0.015; // ~1 deg / 4s — imperceptible until noticed
    }
    // Raycast for hover.
    if (!pointsRef.current || points.length === 0) return;
    raycaster.setFromCamera(pointer, camera);
    let newHovered: number | null = null;
    try {
      const intersects = raycaster.intersectObject(pointsRef.current);
      if (intersects.length > 0 && intersects[0].index !== undefined) {
        newHovered = factIds[intersects[0].index];
      }
    } catch {
      // non-fatal
    }
    if (newHovered !== hoveredRef.current) {
      hoveredRef.current = newHovered;
      setHovered(newHovered);
      document.body.style.cursor = newHovered !== null ? "pointer" : "default";
    }
  });

  const hoveredPoint = useMemo(() => {
    if (hovered === null) return null;
    return points.find((p) => p.fact_id === hovered) ?? null;
  }, [hovered, points]);

  // The lensing halo texture — soft warm radial gradient (Gargantua's core).
  // Computed once; the mesh sits behind the galaxy as the single atmosphere.
  const haloTexture = useMemo(() => {
    const s = 512;
    const cv = document.createElement("canvas");
    cv.width = s; cv.height = s;
    const ctx = cv.getContext("2d")!;
    const g = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    g.addColorStop(0.0, "rgba(255,179,71,0.35)");
    g.addColorStop(0.15, "rgba(255,140,50,0.15)");
    g.addColorStop(0.4, "rgba(60,40,80,0.06)");
    g.addColorStop(1.0, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, s, s);
    const tex = new THREE.CanvasTexture(cv);
    tex.needsUpdate = true;
    return tex;
  }, []);

  useEffect(() => () => haloTexture.dispose(), [haloTexture]);

  return (
    <>
      {/* The lensing halo — the ONE atmospheric element. Warm amber core
          where high-trust facts cluster, fading to void. */}
      <mesh position={[0, 0, -3]}>
        <planeGeometry args={[80, 80]} />
        <meshBasicMaterial transparent opacity={0.55} depthWrite={false} blending={THREE.AdditiveBlending}
          map={haloTexture} />
      </mesh>

      <ambientLight intensity={0.3} />

      {/* The galaxy itself */}
      <points ref={pointsRef} geometry={geometry} material={material}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(hovered !== null && hovered !== selectedId ? hovered : null);
        }}
      />

      {/* Hover emphasis: a soft glow sprite + minimal floating label.
          Everything stays in-scene (no DOM overlay) so it moves and scales
          naturally with the galaxy. The label shows only the preview — full
          detail lives in the side panel on click. */}
      {hoveredPoint && (
        <>
          {/* Glow halo around the hovered point */}
          <sprite position={[hoveredPoint.x, hoveredPoint.y, hoveredPoint.z]} scale={[3.5, 3.5, 1]}>
            <spriteMaterial
              map={starTexture}
              color={hoveredPoint.trust_score > 0.85 ? C.accretion : C.signal}
              transparent
              opacity={0.4}
              depthWrite={false}
              blending={THREE.AdditiveBlending}
            />
          </sprite>
          {/* Minimal label — just the preview text, no box, no border.
              Left-aligned, offset to the right of the point so it doesn't
              cover the glow. Scales with distance like everything else. */}
          <Html
            position={[hoveredPoint.x + 1.2, hoveredPoint.y + 0.3, hoveredPoint.z]}
            distanceFactor={14}
            center={false}
            zIndexRange={[100, 0]}
            style={{ pointerEvents: "none" }}
          >
            <div style={{
              color: C.star,
              fontFamily: TELEMETRY_FONT,
              fontSize: "11px",
              lineHeight: 1.4,
              maxWidth: "200px",
              whiteSpace: "normal",
              textShadow: `0 0 8px ${C.void}, 0 0 4px ${C.void}, 0 1px 2px ${C.void}`,
              // No background, no border — the text floats in space, legible
              // via the dark text-shadow halo against the galaxy.
            }}>
              {hoveredPoint.preview}
              <div style={{
                fontSize: "8px", color: hoveredPoint.trust_score > 0.85 ? C.accretion : C.readout,
                marginTop: "2px", letterSpacing: "0.1em",
                textShadow: `0 0 6px ${C.void}`,
              }}>
                {hoveredPoint.trust_score > 0.85 ? "◆ VERIFIED" : `trust ${hoveredPoint.trust_score.toFixed(2)}`}
                {"  ·  "}
                <span style={{ color: C.signal }}>×{hoveredPoint.retrieval_count}</span>
              </div>
            </div>
          </Html>
        </>
      )}

      {/* Selection reticle — a thin targeting bracket, not a solid ring.
          Four short arcs that frame the point without enclosing it. */}
      {selectedId !== null && (() => {
        const sp = points.find((p) => p.fact_id === selectedId);
        if (!sp) return null;
        const ringColor = sp.trust_score > 0.85 ? C.accretion : C.signal;
        return (
          <group position={[sp.x, sp.y, sp.z]}>
            {/* Soft outer glow */}
            <sprite scale={[4, 4, 1]}>
              <spriteMaterial map={starTexture} color={ringColor} transparent opacity={0.2}
                depthWrite={false} blending={THREE.AdditiveBlending} />
            </sprite>
            {/* Thin reticle ring */}
            <mesh>
              <ringGeometry args={[0.7, 0.74, 64]} />
              <meshBasicMaterial color={ringColor} transparent opacity={0.8} side={THREE.DoubleSide} />
            </mesh>
          </group>
        );
      })()}

      <CameraIntro />
      <OrbitControls
        enablePan={true}
        autoRotate={!reducedMotion}
        autoRotateSpeed={0.15}
        minDistance={8}
        maxDistance={60}
        enableDamping
        dampingFactor={0.06}
      />
    </>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TELEMETRY STRIP — instrument readout. Thin, monospace, hairline-divided.
// Each readout is a single number + label, the way a spacecraft HUD shows
// O2 or distance. No icons, no decoration — the number is the content.
// ═══════════════════════════════════════════════════════════════════════════════
function Telemetry({ stats, galaxy }: { stats: MemoryStats | null; galaxy: MemoryGalaxy | null }) {
  const readouts = [
    { label: "FACTS", value: stats?.t4?.facts, color: C.star },
    { label: "GALAXY PTS", value: galaxy?.points?.length, color: C.accretion },
    { label: "VECTORS", value: stats?.t3?.vector_rows, color: C.readoutHi },
    { label: "ENTITIES", value: stats?.t4?.entities, color: C.readoutHi },
    { label: "RECALLS", value: stats?.activity?.total_retrievals, color: C.signal },
    { label: "QUERIES 24H", value: stats?.queries?.last_24h, color: C.signal },
    { label: "CATEGORIES", value: galaxy?.categories ? Object.keys(galaxy.categories).length : undefined, color: C.readoutHi },
  ];
  return (
    <div style={{
      display: "flex", gap: 0, borderBottom: `1px solid ${C.hairline}`,
      background: C.panel, overflowX: "auto",
    }}>
      {readouts.map((r, i) => (
        <div key={r.label} style={{
          padding: "0.7rem 1.4rem", flex: "1 1 auto", minWidth: "100px",
          borderRight: i < readouts.length - 1 ? `1px solid ${C.hairline}` : "none",
        }}>
          <div style={{
            fontFamily: TELEMETRY_FONT, fontSize: "1.4rem", fontWeight: 600, color: r.color,
            lineHeight: 1, letterSpacing: "-0.02em", fontVariantNumeric: "tabular-nums",
          }}>
            {(r.value ?? 0).toLocaleString()}
          </div>
          <div style={{
            fontFamily: TELEMETRY_FONT, fontSize: "8px", color: C.readout,
            letterSpacing: "0.16em", marginTop: "5px", textTransform: "uppercase",
          }}>
            {r.label}
          </div>
        </div>
      ))}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// DISTRIBUTION PANEL — hairline-divided columns. Bars are thin (2px), glowing,
// against a near-invisible track. Reads as a spectrograph, not a bar chart.
// ═══════════════════════════════════════════════════════════════════════════════
function DistPanel({
  title, data, colorMap, defaultColor,
}: {
  title: string; data: Record<string, number> | undefined;
  colorMap?: Record<string, string>; defaultColor?: string;
}) {
  const entries = data ? Object.entries(data).filter(([, v]) => v > 0) : [];
  const max = entries.length ? Math.max(...entries.map(([, v]) => v), 1) : 1;
  return (
    <div style={{ padding: "1rem 1.4rem", borderRight: `1px solid ${C.hairline}` }}>
      <div style={{
        fontFamily: TELEMETRY_FONT, fontSize: "8px", color: C.readout,
        letterSpacing: "0.18em", marginBottom: "0.8rem", textTransform: "uppercase",
      }}>
        {title}
      </div>
      {entries.length === 0 ? (
        <div style={{ fontSize: "11px", color: C.readout, fontFamily: TELEMETRY_FONT }}>—</div>
      ) : (
        entries.map(([label, value]) => {
          const color = colorMap?.[label] ?? defaultColor ?? C.signal;
          const pct = (value / max) * 100;
          return (
            <div key={label} style={{ marginBottom: "8px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", marginBottom: "3px" }}>
                <span style={{ color: C.readoutHi, fontFamily: TELEMETRY_FONT }}>{label}</span>
                <span style={{ fontFamily: TELEMETRY_FONT, color: C.readout, fontVariantNumeric: "tabular-nums" }}>
                  {value.toLocaleString()}
                </span>
              </div>
              <div style={{ height: "2px", background: C.dust, borderRadius: "0", overflow: "hidden" }}>
                <div style={{
                  width: `${pct}%`, height: "100%", background: color,
                  borderRadius: "0", transition: "width 0.5s ease",
                  boxShadow: `0 0 3px ${color}88`,
                }} />
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}

// Faint coordinate grid — the "observatory window" feel. Barely visible.
function GridOverlay() {
  return (
    <div style={{
      position: "absolute", inset: 0, pointerEvents: "none", opacity: 0.12,
      backgroundImage: `
        linear-gradient(${C.dust}33 1px, transparent 1px),
        linear-gradient(90deg, ${C.dust}33 1px, transparent 1px)
      `,
      backgroundSize: "64px 64px",
    }} />
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═══════════════════════════════════════════════════════════════════════════════
export default function MemoryVizPage() {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [galaxy, setGalaxy] = useState<MemoryGalaxy | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<number | null>(null);
  const [lastUpdate, setLastUpdate] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [s, g] = await Promise.all([api.getMemoryStats(), api.getMemoryGalaxy()]);
      setStats(s); setGalaxy(g); setLastUpdate(Date.now()); setError(null);
    } catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    fetchData();
    timerRef.current = setInterval(fetchData, POLL_MS);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [fetchData]);

  const selectedData = useMemo(() => {
    if (selectedPoint === null || !galaxy) return null;
    return galaxy.points.find((p) => p.fact_id === selectedPoint) ?? null;
  }, [selectedPoint, galaxy]);

  const legendEntries = useMemo(() => {
    if (!galaxy?.categories) return [];
    return Object.entries(galaxy.categories).sort(([, a], [, b]) => b - a).slice(0, 10);
  }, [galaxy?.categories]);

  const ageSeconds = lastUpdate ? Math.floor((Date.now() - lastUpdate) / 1000) : 0;

  if (loading && !stats) {
    return (
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center",
        padding: "4rem", background: C.void, minHeight: "100vh" }}>
        <Spinner className="size-6" style={{ color: C.accretion }} />
      </div>
    );
  }

  const factCount = (stats?.t4?.facts ?? 0).toLocaleString();
  const ptCount = (galaxy?.points?.length ?? 0).toLocaleString();

  return (
    <div style={{ background: C.void, color: C.star, minHeight: "100vh" }}>
      {/* ── HERO: the galaxy viewport ─────────────────────────────────────── */}
      <div style={{
        position: "relative", width: "100%", height: "68vh", minHeight: "480px",
        borderBottom: `1px solid ${C.hairline}`, overflow: "hidden",
        background: `radial-gradient(ellipse at 45% 50%, ${C.dust}22 0%, ${C.void} 70%)`,
      }}>
        <GridOverlay />
        {(galaxy?.points?.length ?? 0) > 0 ? (
          <Canvas camera={{ position: [0, 0, 55], fov: 50 }} style={{ background: "transparent" }} dpr={[1, 2]}>
            <GalaxyScene points={galaxy!.points} selectedId={selectedPoint} onSelect={setSelectedPoint} />
          </Canvas>
        ) : (
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "center", height: "100%",
            color: C.readout, fontFamily: TELEMETRY_FONT, fontSize: "11px", letterSpacing: "0.1em",
          }}>
            NO MEMORY DATA
          </div>
        )}

        {/* ── Top-left: the title block ────────────────────────────────────── */}
        <div style={{
          position: "absolute", top: "1.5rem", left: "2rem", pointerEvents: "none",
          zIndex: 10,
        }}>
          <div style={{
            fontFamily: DISPLAY_FONT, fontSize: "13px", fontWeight: 700,
            letterSpacing: "0.32em", color: C.star, textTransform: "uppercase",
            marginBottom: "0.4rem",
            textShadow: `0 0 20px ${C.void}`,
          }}>
            Memory Palace
          </div>
          <div style={{
            fontFamily: TELEMETRY_FONT, fontSize: "10px", color: C.readout,
            letterSpacing: "0.06em",
            textShadow: `0 0 12px ${C.void}`,
          }}>
            {factCount} facts · {ptCount} plotted · semantic cluster
          </div>
        </div>

        {/* ── Top-right: status + refresh ──────────────────────────────────── */}
        <div style={{
          position: "absolute", top: "1.5rem", right: "2rem",
          display: "flex", alignItems: "center", gap: "0.8rem", zIndex: 10,
        }}>
          {galaxy?.stale && (
            <span style={{ fontFamily: TELEMETRY_FONT, fontSize: "8px", color: C.accretion, letterSpacing: "0.14em" }}>
              ◇ STALE
            </span>
          )}
          {galaxy?.cached && (
            <span style={{ fontFamily: TELEMETRY_FONT, fontSize: "8px", color: C.readout, letterSpacing: "0.14em" }}>
              ◇ CACHED
            </span>
          )}
          {lastUpdate > 0 && (
            <span style={{ fontFamily: TELEMETRY_FONT, fontSize: "8px", color: C.readout, letterSpacing: "0.1em" }}>
              Δ {ageSeconds}s
            </span>
          )}
          <Button size="sm" onClick={() => { setLoading(true); fetchData(); }}
            style={{ background: "transparent", border: `1px solid ${C.hairline}`, padding: "4px 8px" }}>
            <RefreshCw size={12} color={C.readoutHi} />
          </Button>
        </div>

        {/* ── Bottom-left: category legend ─────────────────────────────────── */}
        <div style={{
          position: "absolute", bottom: "1.2rem", left: "2rem",
          display: "flex", gap: "0.7rem", flexWrap: "wrap", maxWidth: "62%",
          zIndex: 10,
        }}>
          {legendEntries.map(([cat, count]) => {
            const tint = categoryTint(cat);
            const isAmber = cat === "user_pref"; // only category with warm tint
            const colorStr = isAmber
              ? C.accretion
              : `rgb(${Math.round(tint[0]*255)},${Math.round(tint[1]*255)},${Math.round(tint[2]*255)})`;
            return (
              <span key={cat} style={{
                display: "flex", alignItems: "center", gap: "4px",
                fontFamily: TELEMETRY_FONT, fontSize: "8px", color: C.readout,
                letterSpacing: "0.1em",
              }}>
                <span style={{
                  width: "5px", height: "5px", borderRadius: "50%", background: colorStr,
                  boxShadow: `0 0 5px ${colorStr}aa`, display: "inline-block",
                }} />
                {cat.toUpperCase()} <span style={{ color: C.readout, opacity: 0.6 }}>{count}</span>
              </span>
            );
          })}
        </div>

        {/* ── Bottom-right: trust legend (the encoding key) ────────────────── */}
        <div style={{
          position: "absolute", bottom: "1.2rem", right: "2rem",
          display: "flex", flexDirection: "column", gap: "4px",
          zIndex: 10, alignItems: "flex-end",
        }}>
          <span style={{
            fontFamily: TELEMETRY_FONT, fontSize: "7px", color: C.readout,
            letterSpacing: "0.16em", marginBottom: "2px", textTransform: "uppercase",
          }}>
            Trust Encoding
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "5px", fontFamily: TELEMETRY_FONT, fontSize: "8px", color: C.accretion, letterSpacing: "0.08em" }}>
            <span style={{ width: "5px", height: "5px", borderRadius: "50%", background: C.accretion, boxShadow: `0 0 6px ${C.accretion}`, display: "inline-block" }} />
            VERIFIED
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "5px", fontFamily: TELEMETRY_FONT, fontSize: "8px", color: C.star, letterSpacing: "0.08em", opacity: 0.7 }}>
            <span style={{ width: "4px", height: "4px", borderRadius: "50%", background: C.star, display: "inline-block" }} />
            STATED
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "5px", fontFamily: TELEMETRY_FONT, fontSize: "8px", color: C.readout, letterSpacing: "0.08em" }}>
            <span style={{ width: "3px", height: "3px", borderRadius: "50%", background: C.dust, display: "inline-block" }} />
            DUST
          </span>
        </div>

        {/* ── Selected fact detail (right overlay) ─────────────────────────── */}
        {selectedData && (
          <div style={{
            position: "absolute", top: "4.5rem", right: "2rem", maxWidth: "300px",
            background: `rgba(5,7,13,0.96)`, border: `1px solid ${C.hairline}`,
            padding: "0.9rem 1.1rem", borderRadius: "1px", zIndex: 10,
            boxShadow: selectedData.trust_score > 0.85
              ? `0 0 24px ${C.accretion}22, 0 4px 16px rgba(0,0,0,0.6)`
              : `0 0 16px rgba(0,0,0,0.5)`,
          }}>
            <div style={{
              fontFamily: TELEMETRY_FONT, fontSize: "7px", color: C.readout,
              letterSpacing: "0.18em", marginBottom: "0.5rem", textTransform: "uppercase",
            }}>
              Selected · Fact #{selectedData.fact_id}
            </div>
            <div style={{
              fontFamily: TELEMETRY_FONT, fontSize: "11px", color: C.star,
              marginBottom: "0.7rem", lineHeight: 1.45,
            }}>
              {selectedData.preview}
            </div>
            <div style={{
              display: "flex", justifyContent: "space-between",
              fontSize: "9px", fontFamily: TELEMETRY_FONT, marginBottom: "0.4rem",
            }}>
              <span style={{ color: C.readout }}>trust
                <span style={{
                  color: selectedData.trust_score > 0.85 ? C.accretion : C.star,
                  marginLeft: "4px", fontWeight: 600,
                }}>
                  {selectedData.trust_score.toFixed(3)}
                </span>
              </span>
              <span style={{ color: C.readout }}>recall
                <span style={{ color: C.signal, marginLeft: "4px", fontWeight: 600 }}>×{selectedData.retrieval_count}</span>
              </span>
            </div>
            <div style={{
              display: "flex", justifyContent: "space-between",
              fontSize: "8px", fontFamily: TELEMETRY_FONT, color: C.readout,
              letterSpacing: "0.08em",
            }}>
              <span>{selectedData.category.toUpperCase()}</span>
              <span>{selectedData.epistemic_status.toUpperCase()}</span>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div style={{
          padding: "0.5rem 2rem", fontFamily: TELEMETRY_FONT, fontSize: "10px",
          color: "#FB7185", borderBottom: `1px solid ${C.hairline}`, background: C.panel,
        }}>
          {error}
        </div>
      )}

      {/* ── Telemetry strip ──────────────────────────────────────────────── */}
      <Telemetry stats={stats} galaxy={galaxy} />

      {/* ── Distribution panels ──────────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}>
        <DistPanel title="Trust Spectrum · T4" data={stats?.t4?.trust_bands}
          colorMap={{ "0.85-1.0": C.accretion, "0.5-0.85": C.star, "0.2-0.5": C.readoutHi, "0-0.2": C.dust }} />
        <DistPanel title="Chunk Lifecycle · T2" data={stats?.t2?.by_lifecycle}
          colorMap={{ sealed: C.signal, admitted: C.star, dropped: "#FB7185" }} />
        <DistPanel title="Score Distribution · T2" data={stats?.t2?.score_bands}
          colorMap={{ "0.85-1.0": C.accretion, "0.65-0.85": C.signal, "0.5-0.65": C.readoutHi, "0.15-0.5": C.readout, "0-0.15": C.dust }} />
        <DistPanel title="Fact Categories · T4" data={stats?.t4?.by_category} defaultColor={C.signal} />
        <DistPanel title="Epistemic Status · T4" data={stats?.t4?.epistemic}
          colorMap={{ verified: C.accretion, stated: C.star, inferred: C.readoutHi, contradicted: "#FB7185" }} />
        <DistPanel title="Source Kind · T2" data={stats?.t2?.by_source}
          colorMap={{ chat: C.signal, document: C.readoutHi }} />
      </div>

      {/* ── Activity telemetry ───────────────────────────────────────────── */}
      {(stats?.activity || stats?.queries) && (
        <div style={{
          display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          borderTop: `1px solid ${C.hairline}`,
        }}>
          {stats?.activity?.top_retrieved && stats.activity.top_retrieved.length > 0 && (
            <div style={{ padding: "1rem 1.4rem", borderRight: `1px solid ${C.hairline}` }}>
              <div style={{
                fontFamily: TELEMETRY_FONT, fontSize: "8px", color: C.readout,
                letterSpacing: "0.18em", marginBottom: "0.8rem", textTransform: "uppercase",
              }}>
                Most Recalled · Activity
              </div>
              {stats.activity.top_retrieved.map((f, i) => (
                <div key={f.fact_id} style={{
                  display: "flex", alignItems: "baseline", gap: "0.5rem", marginBottom: "7px", fontSize: "10px",
                }}>
                  <span style={{ fontFamily: TELEMETRY_FONT, color: C.readout, fontSize: "8px", minWidth: "16px" }}>
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span style={{ color: C.readoutHi, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontFamily: TELEMETRY_FONT }}>
                    {f.preview}
                  </span>
                  <span style={{ fontFamily: TELEMETRY_FONT, color: C.signal, fontSize: "9px", minWidth: "30px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                    ×{f.retrievals}
                  </span>
                  {f.helpful > 0 && (
                    <span style={{ fontFamily: TELEMETRY_FONT, color: C.accretion, fontSize: "8px", minWidth: "22px", textAlign: "right" }}>
                      +{f.helpful}
                    </span>
                  )}
                </div>
              ))}
              <div style={{
                marginTop: "0.7rem", paddingTop: "0.5rem", borderTop: `1px solid ${C.hairline}`,
                display: "flex", justifyContent: "space-between", fontSize: "9px",
                fontFamily: TELEMETRY_FONT, color: C.readout,
              }}>
                <span>avg trust <span style={{ color: C.accretion }}>{stats.activity.avg_trust?.toFixed(3)}</span></span>
                <span>{stats.activity.facts_recalled} recalled of {stats.t4?.facts ?? 0}</span>
              </div>
            </div>
          )}

          {stats?.queries?.by_tool && Object.keys(stats.queries.by_tool).length > 0 && (
            <div style={{ padding: "1rem 1.4rem", borderRight: `1px solid ${C.hairline}` }}>
              <div style={{
                fontFamily: TELEMETRY_FONT, fontSize: "8px", color: C.readout,
                letterSpacing: "0.18em", marginBottom: "0.8rem", textTransform: "uppercase",
              }}>
                Agent Queries · Last 24h
              </div>
              {Object.entries(stats.queries.by_tool).sort(([, a], [, b]) => b - a).map(([tool, count]) => {
                const max = Math.max(...Object.values(stats.queries!.by_tool), 1);
                const pct = (count / max) * 100;
                const color = tool.includes("semantic") ? C.signal : tool.includes("hybrid") ? C.accretion : C.star;
                return (
                  <div key={tool} style={{ marginBottom: "8px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", marginBottom: "3px" }}>
                      <span style={{ color: C.readoutHi, fontFamily: TELEMETRY_FONT }}>{tool}</span>
                      <span style={{ fontFamily: TELEMETRY_FONT, color: C.readout, fontVariantNumeric: "tabular-nums" }}>{count}</span>
                    </div>
                    <div style={{ height: "2px", background: C.dust, overflow: "hidden" }}>
                      <div style={{ width: `${pct}%`, height: "100%", background: color, transition: "width 0.5s ease", boxShadow: `0 0 3px ${color}88` }} />
                    </div>
                  </div>
                );
              })}
              {stats.queries.by_hour && Object.keys(stats.queries.by_hour).length > 0 && (
                <div style={{ marginTop: "0.8rem" }}>
                  <div style={{ fontFamily: TELEMETRY_FONT, fontSize: "7px", color: C.readout, letterSpacing: "0.16em", marginBottom: "5px" }}>
                    HOURLY
                  </div>
                  <div style={{ display: "flex", alignItems: "flex-end", gap: "1px", height: "22px" }}>
                    {Array.from({ length: 24 }, (_, h) => {
                      const hour = String(h).padStart(2, "0");
                      const count = stats.queries!.by_hour[hour] ?? 0;
                      const maxH = Math.max(...Object.values(stats.queries!.by_hour), 1);
                      const pct = (count / maxH) * 100;
                      return (
                        <div key={h} title={`${hour}:00 — ${count} queries`} style={{
                          flex: 1, height: `${Math.max(pct, count > 0 ? 8 : 2)}%`,
                          background: count > 0 ? `${C.signal}66` : C.dust,
                          transition: "height 0.5s ease", minHeight: "2px",
                        }} />
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <div style={{
        padding: "0.9rem 2rem", borderTop: `1px solid ${C.hairline}`,
        fontFamily: TELEMETRY_FONT, fontSize: "8px", color: C.readout,
        letterSpacing: "0.18em", textTransform: "uppercase",
        display: "flex", justifyContent: "space-between", background: C.panel,
      }}>
        <span>Hermes Memory System · Galaxy PCA · {ptCount} plotted of {factCount}</span>
        <span>poll {POLL_MS / 1000}s</span>
      </div>
    </div>
  );
}
