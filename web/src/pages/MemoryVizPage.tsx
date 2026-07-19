import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import { Brain, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import type { MemoryStats, MemoryGraph } from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";

// ── 3D Graph ─────────────────────────────────────────────────────────────────
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Line, Html } from "@react-three/drei";
import * as THREE from "three";

// Design tokens — "memory palace / observatory" palette.
// Deep cool space, synapse-cyan accent, myelin-gold for high-trust.
// Deliberately NOT the near-black + acid-green AI default.
const C = {
  space: "#080B14",       // background — deep, cool, bluer than generic #0A0A0A
  surface: "#111827",     // raised panel
  hairline: "#1E293B",    // dividers
  synapse: "#7DD3FC",     // primary accent — electric neural cyan
  myelin: "#FCD34D",      // high-trust / verified — warm against cool field
  resting: "#94A3B8",     // secondary text, dormant data
  text: "#E2E8F0",
  textDim: "#64748B",
};
// Nodal type colors — semantic, muted to sit in the field not pop out of it
const TYPE_COLORS: Record<string, string> = {
  concept: C.synapse,
  infrastructure: "#34D399",
  health: "#FB7185",
  event: C.myelin,
  note: "#A78BFA",
  "hydra-result": "#FB923C",
  unknown: C.resting,
};
const POLL_MS = 30_000;

// ── Force simulation types ───────────────────────────────────────────────────
interface SimNode {
  id: string; title: string; type: string; updated_at: string;
  pos: THREE.Vector3; vel: THREE.Vector3;
}
interface SimLink { source: string; target: string; type: string; context: string; }

function nodeColor(type: string): string { return TYPE_COLORS[type] || TYPE_COLORS.unknown; }

function initPositions(nodes: SimNode[]): void {
  const n = nodes.length;
  nodes.forEach((node, i) => {
    const phi = Math.acos(-1 + (2 * i) / Math.max(n, 1));
    const theta = Math.sqrt(n * Math.PI) * phi;
    const r = 6;
    node.pos.set(r * Math.cos(theta) * Math.sin(phi), r * Math.sin(theta) * Math.sin(phi), r * Math.cos(phi));
    node.vel.set(0, 0, 0);
  });
}

// ── Camera intro (slow dolly-in on mount — the signature moment) ─────────────
function CameraIntro() {
  const started = useRef(false);
  useFrame((state) => {
    if (!started.current) {
      started.current = true;
      state.camera.position.set(0, 0, 40);
    }
    const target = 15;
    if (state.camera.position.z > target) {
      state.camera.position.z = THREE.MathUtils.lerp(state.camera.position.z, target, 0.025);
    }
  });
  return null;
}

// ── The graph scene ──────────────────────────────────────────────────────────
function GraphScene({
  nodes, links, selectedId, onSelect, pulseRef,
}: {
  nodes: SimNode[]; links: SimLink[]; selectedId: string | null;
  onSelect: (id: string | null) => void; pulseRef: React.MutableRefObject<number>;
}) {
  const nodeMap = useMemo(() => {
    const m = new Map<string, SimNode>();
    nodes.forEach((n) => m.set(n.id, n));
    return m;
  }, [nodes]);
  const [hovered, setHovered] = useState<string | null>(null);

  useFrame(() => {
    if (nodes.length === 0) return;
    const repulsion = 3.0;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const diff = a.pos.clone().sub(b.pos);
        const distSq = Math.max(diff.lengthSq(), 0.5);
        const force = repulsion / distSq;
        diff.normalize().multiplyScalar(force * 0.3);
        a.vel.add(diff); b.vel.sub(diff);
      }
    }
    const spring = 0.02, restLen = 4.0;
    for (const link of links) {
      const a = nodeMap.get(link.source), b = nodeMap.get(link.target);
      if (!a || !b) continue;
      const diff = b.pos.clone().sub(a.pos);
      const dist = diff.length();
      diff.normalize().multiplyScalar((dist - restLen) * spring);
      a.vel.add(diff); b.vel.sub(diff);
    }
    const gravity = 0.008, damping = 0.85;
    for (const n of nodes) {
      n.vel.add(n.pos.clone().multiplyScalar(-gravity));
      n.vel.multiplyScalar(damping);
      n.pos.add(n.vel.clone().multiplyScalar(0.3));
    }
  });

  // Build edge line points
  const edgePoints = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    for (const link of links) {
      const s = nodeMap.get(link.source), t = nodeMap.get(link.target);
      if (s && t) pts.push(s.pos.clone(), t.pos.clone());
    }
    return pts;
  }, [links, nodeMap]);

  const highlightedLinks = useMemo(() => {
    if (!selectedId) return [];
    return links.filter((l) => l.source === selectedId || l.target === selectedId);
  }, [selectedId, links]);

  return (
    <>
      <ambientLight intensity={0.35} />
      <pointLight position={[12, 12, 12]} intensity={0.7} color={C.synapse} />
      <pointLight position={[-12, -8, -12]} intensity={0.3} color={C.myelin} />

      {/* Nodes */}
      {nodes.map((node) => (
        <NodeMesh key={node.id} node={node} isHovered={hovered === node.id}
          isSelected={selectedId === node.id}
          onHover={(h) => { setHovered(h ? node.id : null); document.body.style.cursor = h ? "pointer" : "default"; }}
          onClick={() => onSelect(selectedId === node.id ? null : node.id)}
          pulseRef={pulseRef}
        />
      ))}

      {/* All edges (faint) */}
      {edgePoints.length > 0 && (
        <Line points={edgePoints} color={C.hairline} lineWidth={1} transparent opacity={0.35} />
      )}
      {/* Highlighted edges (selected node's connections) */}
      {highlightedLinks.map((link, i) => {
        const s = nodeMap.get(link.source), t = nodeMap.get(link.target);
        if (!s || !t) return null;
        return <Line key={`hl-${i}`} points={[s.pos, t.pos]} color={C.synapse} lineWidth={1.5} transparent opacity={0.85} />;
      })}

      <CameraIntro />
      <OrbitControls enablePan={false} autoRotate autoRotateSpeed={0.25} minDistance={5} maxDistance={35}
        enableDamping dampingFactor={0.08} />
    </>
  );
}

function NodeMesh({
  node, isHovered, isSelected, onHover, onClick, pulseRef,
}: {
  node: SimNode; isHovered: boolean; isSelected: boolean;
  onHover: (h: boolean) => void; onClick: () => void;
  pulseRef: React.MutableRefObject<number>;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const groupRef = useRef<THREE.Group>(null);
  const isHighTrust = node.type === "event" || node.updated_at > "2026-07-01"; // proxy: recent/event = high-signal

  useFrame(() => {
    if (groupRef.current) groupRef.current.position.copy(node.pos);
    if (meshRef.current) {
      const target = isSelected ? 1.5 : isHovered ? 1.25 : 1.0;
      meshRef.current.scale.lerp(new THREE.Vector3(target, target, target), 0.15);
      // Gentle pulse on high-trust nodes — the "living" quality
      if (isHighTrust && !isSelected) {
        const pulse = 1 + Math.sin(pulseRef.current * 1.5 + node.pos.x) * 0.04;
        meshRef.current.scale.multiplyScalar(pulse);
      }
    }
  });

  const color = nodeColor(node.type);
  return (
    <group ref={groupRef}>
      <mesh
        ref={meshRef}
        onPointerOver={(e) => { e.stopPropagation(); onHover(true); }}
        onPointerOut={(e) => { e.stopPropagation(); onHover(false); }}
        onClick={(e) => { e.stopPropagation(); onClick(); }}
      >
        <sphereGeometry args={[0.3, 24, 24]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isSelected ? 0.7 : isHovered ? 0.5 : isHighTrust ? 0.35 : 0.15}
          roughness={0.35} metalness={0.4}
        />
      </mesh>
      {(isHovered || isSelected) && (
        <Html distanceFactor={10} position={[0, 0.55, 0]} center>
          <div style={{
            background: "rgba(8,11,20,0.92)", color: C.text, padding: "4px 8px",
            fontFamily: "monospace", fontSize: "11px", whiteSpace: "nowrap",
            border: `1px solid ${C.hairline}`, borderRadius: "2px", pointerEvents: "none",
          }}>
            {node.title}
          </div>
        </Html>
      )}
    </group>
  );
}

// ── Telemetry strip (replaces card-heavy stat grid) ──────────────────────────
function Telemetry({ stats, graph }: { stats: MemoryStats | null; graph: MemoryGraph | null }) {
  const readouts = [
    { label: "FACTS", value: stats?.t4?.facts, color: C.synapse },
    { label: "CHUNKS", value: stats?.t2?.chunks, color: "#34D399" },
    { label: "VECTORS", value: stats?.t3?.vector_rows, color: "#A78BFA" },
    { label: "ENTITIES", value: stats?.t4?.entities, color: "#FB7185" },
    { label: "RECALLS", value: stats?.activity?.total_retrievals, color: "#60A5FA" },
    { label: "FEEDBACK", value: stats?.activity?.total_helpful, color: C.myelin },
    { label: "QUERIES 24H", value: stats?.queries?.last_24h, color: "#34D399" },
    { label: "WIKI NODES", value: graph?.nodes?.length, color: C.myelin },
    { label: "GRAPH EDGES", value: graph?.links?.length, color: "#FB923C" },
  ];
  return (
    <div style={{ display: "flex", gap: 0, borderBottom: `1px solid ${C.hairline}`, overflowX: "auto" }}>
      {readouts.map((r, i) => (
        <div key={r.label} style={{
          padding: "0.75rem 1.25rem", flex: "1 1 auto", minWidth: "110px",
          borderRight: i < readouts.length - 1 ? `1px solid ${C.hairline}` : "none",
        }}>
          <div style={{
            fontFamily: "monospace", fontSize: "1.5rem", fontWeight: 700, color: r.color,
            lineHeight: 1, letterSpacing: "-0.02em",
          }}>
            {(r.value ?? 0).toLocaleString()}
          </div>
          <div style={{
            fontFamily: "monospace", fontSize: "9px", color: C.textDim,
            letterSpacing: "0.12em", marginTop: "4px", textTransform: "uppercase",
          }}>
            {r.label}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Distribution panel (thin, hairline-divided) ──────────────────────────────
function DistPanel({
  title, data, colorMap, defaultColor,
}: {
  title: string; data: Record<string, number> | undefined;
  colorMap?: Record<string, string>; defaultColor?: string;
}) {
  const entries = data ? Object.entries(data).filter(([, v]) => v > 0) : [];
  const max = entries.length ? Math.max(...entries.map(([, v]) => v), 1) : 1;
  return (
    <div style={{ padding: "1rem 1.25rem", borderRight: `1px solid ${C.hairline}` }}>
      <div style={{
        fontFamily: "monospace", fontSize: "9px", color: C.textDim,
        letterSpacing: "0.14em", marginBottom: "0.75rem", textTransform: "uppercase",
      }}>
        {title}
      </div>
      {entries.length === 0 ? (
        <div style={{ fontSize: "12px", color: C.textDim }}>—</div>
      ) : (
        entries.map(([label, value]) => {
          const color = colorMap?.[label] ?? defaultColor ?? C.synapse;
          const pct = (value / max) * 100;
          return (
            <div key={label} style={{ marginBottom: "7px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "2px" }}>
                <span style={{ color: C.resting }}>{label}</span>
                <span style={{ fontFamily: "monospace", color: C.textDim }}>{value.toLocaleString()}</span>
              </div>
              <div style={{ height: "3px", background: "rgba(100,116,139,0.12)", borderRadius: "1px", overflow: "hidden" }}>
                <div style={{
                  width: `${pct}%`, height: "100%", background: color,
                  borderRadius: "1px", transition: "width 0.4s ease",
                  boxShadow: `0 0 4px ${color}66`,
                }} />
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}

// ── Coordinate grid overlay (faint, the observatory feel) ────────────────────
function GridOverlay() {
  return (
    <div style={{
      position: "absolute", inset: 0, pointerEvents: "none", opacity: 0.25,
      backgroundImage: `
        linear-gradient(${C.hairline}22 1px, transparent 1px),
        linear-gradient(90deg, ${C.hairline}22 1px, transparent 1px)
      `,
      backgroundSize: "48px 48px",
    }} />
  );
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function MemoryVizPage() {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [graph, setGraph] = useState<MemoryGraph | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState(0);
  const pulseRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [s, g] = await Promise.all([api.getMemoryStats(), api.getMemoryGraph()]);
      setStats(s); setGraph(g); setLastUpdate(Date.now()); setError(null);
    } catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    fetchData();
    timerRef.current = setInterval(fetchData, POLL_MS);
    const animTimer = setInterval(() => { pulseRef.current += 0.05; }, 50);
    return () => { if (timerRef.current) clearInterval(timerRef.current); clearInterval(animTimer); };
  }, [fetchData]);

  const simNodes: SimNode[] = useMemo(() => {
    if (!graph?.nodes) return [];
    const nodes: SimNode[] = graph.nodes.map((n) => ({
      id: n.id, title: n.title, type: n.type, updated_at: n.updated_at,
      pos: new THREE.Vector3(), vel: new THREE.Vector3(),
    }));
    initPositions(nodes);
    return nodes;
  }, [graph?.nodes]);

  const simLinks: SimLink[] = useMemo(() => {
    if (!graph?.links) return [];
    return graph.links.map((l) => ({ source: l.source, target: l.target, type: l.type, context: l.context }));
  }, [graph?.links]);

  const selectedData = useMemo(() => {
    if (!selectedNode || !graph) return null;
    const node = graph.nodes.find((n) => n.id === selectedNode);
    if (!node) return null;
    return {
      node,
      outLinks: graph.links.filter((l) => l.source === selectedNode),
      inLinks: graph.links.filter((l) => l.target === selectedNode),
    };
  }, [selectedNode, graph]);

  const ageSeconds = lastUpdate ? Math.floor((Date.now() - lastUpdate) / 1000) : 0;

  if (loading && !stats) {
    return <div style={{ display: "flex", justifyContent: "center", padding: "4rem", background: C.space, minHeight: "60vh" }}><Spinner className="size-6" /></div>;
  }

  return (
    <div style={{ background: C.space, color: C.text, minHeight: "100vh" }}>
      {/* Hero: full-bleed 3D graph viewport */}
      <div style={{ position: "relative", width: "100%", height: "62vh", minHeight: "440px", borderBottom: `1px solid ${C.hairline}` }}>
        <GridOverlay />
        {simNodes.length > 0 ? (
          <Canvas camera={{ position: [0, 0, 40], fov: 50 }} style={{ background: "transparent" }} dpr={[1, 2]}>
            <GraphScene nodes={simNodes} links={simLinks} selectedId={selectedNode} onSelect={setSelectedNode} pulseRef={pulseRef} />
          </Canvas>
        ) : (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: C.textDim, fontFamily: "monospace", fontSize: "12px" }}>
            NO GRAPH DATA
          </div>
        )}

        {/* Top-left readout (the single composite headline) */}
        <div style={{ position: "absolute", top: "1.25rem", left: "1.5rem", pointerEvents: "none" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
            <Brain size={16} color={C.synapse} />
            <span style={{ fontFamily: "monospace", fontSize: "10px", letterSpacing: "0.18em", color: C.resting, textTransform: "uppercase" }}>
              Memory Palace
            </span>
          </div>
          <div style={{ fontFamily: "monospace", fontSize: "13px", color: C.textDim }}>
            {(stats?.t4?.facts ?? 0).toLocaleString()} facts · {(stats?.t2?.chunks ?? 0).toLocaleString()} chunks · {graph?.nodes?.length ?? 0} wiki nodes
          </div>
        </div>

        {/* Top-right status */}
        <div style={{ position: "absolute", top: "1.25rem", right: "1.5rem", display: "flex", alignItems: "center", gap: "0.75rem" }}>
          {graph?.stale && (
            <span style={{ fontFamily: "monospace", fontSize: "9px", color: C.myelin, letterSpacing: "0.1em" }}>◇ STALE</span>
          )}
          {lastUpdate > 0 && (
            <span style={{ fontFamily: "monospace", fontSize: "9px", color: C.textDim, letterSpacing: "0.08em" }}>Δ {ageSeconds}s</span>
          )}
          <Button size="sm" onClick={() => { setLoading(true); fetchData(); }} style={{ background: "transparent", border: `1px solid ${C.hairline}` }}>
            <RefreshCw size={13} color={C.resting} />
          </Button>
        </div>

        {/* Legend (bottom-left, over the graph) */}
        <div style={{ position: "absolute", bottom: "1rem", left: "1.5rem", display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          {Object.entries(TYPE_COLORS).filter(([k]) => k !== "unknown").map(([type, color]) => (
            <span key={type} style={{ display: "flex", alignItems: "center", gap: "4px", fontFamily: "monospace", fontSize: "9px", color: C.textDim, letterSpacing: "0.08em" }}>
              <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: color, boxShadow: `0 0 6px ${color}99`, display: "inline-block" }} />
              {type.toUpperCase()}
            </span>
          ))}
        </div>

        {/* Selected node detail (bottom-right overlay) */}
        {selectedData && (
          <div style={{
            position: "absolute", bottom: "1rem", right: "1.5rem", maxWidth: "320px",
            background: `rgba(8,11,20,0.94)`, border: `1px solid ${C.hairline}`, padding: "0.85rem 1rem",
          }}>
            <div style={{ fontFamily: "monospace", fontSize: "12px", color: C.text, marginBottom: "0.4rem" }}>
              <span style={{ color: nodeColor(selectedData.node.type) }}>● </span>
              {selectedData.node.title}
            </div>
            <div style={{ fontFamily: "monospace", fontSize: "9px", color: C.textDim, marginBottom: "0.6rem", letterSpacing: "0.06em" }}>
              {selectedData.node.type.toUpperCase()} · {selectedData.node.id}
            </div>
            {selectedData.outLinks.length > 0 && (
              <div style={{ marginBottom: "0.4rem" }}>
                <div style={{ fontFamily: "monospace", fontSize: "8px", color: C.textDim, letterSpacing: "0.12em", marginBottom: "3px" }}>→ OUTBOUND ({selectedData.outLinks.length})</div>
                {selectedData.outLinks.slice(0, 4).map((l, i) => (
                  <div key={i} style={{ fontSize: "10px", color: C.resting, marginBottom: "2px", lineHeight: 1.3 }}>
                    <span style={{ color: C.synapse }}>{l.type}</span> → {l.target.split("/").pop()}
                    {l.context && <div style={{ color: C.textDim, paddingLeft: "6px", fontSize: "9px" }}>{l.context.slice(0, 65)}</div>}
                  </div>
                ))}
              </div>
            )}
            {selectedData.inLinks.length > 0 && (
              <div>
                <div style={{ fontFamily: "monospace", fontSize: "8px", color: C.textDim, letterSpacing: "0.12em", marginBottom: "3px" }}>← INBOUND ({selectedData.inLinks.length})</div>
                {selectedData.inLinks.slice(0, 3).map((l, i) => (
                  <div key={i} style={{ fontSize: "10px", color: C.resting, marginBottom: "2px" }}>
                    <span style={{ color: "#34D399" }}>{l.type}</span> ← {l.source.split("/").pop()}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {error && (
        <div style={{ padding: "0.5rem 1.5rem", fontFamily: "monospace", fontSize: "11px", color: "#FB7185", borderBottom: `1px solid ${C.hairline}` }}>
          {error}
        </div>
      )}

      {/* Telemetry strip */}
      <Telemetry stats={stats} graph={graph} />

      {/* Distribution panels (thin, hairline-divided columns) */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
      }}>
        <DistPanel
          title="Trust Spectrum · T4"
          data={stats?.t4?.trust_bands}
          colorMap={{ "0.85-1.0": C.myelin, "0.5-0.85": C.synapse, "0.2-0.5": C.resting, "0-0.2": "#FB7185" }}
        />
        <DistPanel
          title="Chunk Lifecycle · T2"
          data={stats?.t2?.by_lifecycle}
          colorMap={{ sealed: "#34D399", admitted: C.synapse, dropped: "#FB7185" }}
        />
        <DistPanel
          title="Score Distribution · T2"
          data={stats?.t2?.score_bands}
          colorMap={{ "0.85-1.0": C.myelin, "0.65-0.85": C.synapse, "0.5-0.65": C.resting, "0.15-0.5": C.textDim, "0-0.15": "#FB7185" }}
        />
        <DistPanel
          title="Fact Categories · T4"
          data={stats?.t4?.by_category}
          defaultColor="#A78BFA"
        />
        <DistPanel
          title="Epistemic Status · T4"
          data={stats?.t4?.epistemic}
          colorMap={{ verified: C.myelin, stated: C.synapse, inferred: C.resting, contradicted: "#FB7185" }}
        />
        <DistPanel
          title="Source Kind · T2"
          data={stats?.t2?.by_source}
          colorMap={{ chat: C.synapse, document: C.myelin }}
        />
      </div>

      {/* Activity telemetry — retrieval + query analytics */}
      {(stats?.activity || stats?.queries) && (
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          borderTop: `1px solid ${C.hairline}`,
        }}>
          {/* Top retrieved facts */}
          {stats?.activity?.top_retrieved && stats.activity.top_retrieved.length > 0 && (
            <div style={{ padding: "1rem 1.25rem", borderRight: `1px solid ${C.hairline}` }}>
              <div style={{
                fontFamily: "monospace", fontSize: "9px", color: C.textDim,
                letterSpacing: "0.14em", marginBottom: "0.75rem", textTransform: "uppercase",
              }}>
                Most Recalled · Activity
              </div>
              {stats.activity.top_retrieved.map((f, i) => (
                <div key={f.fact_id} style={{
                  display: "flex", alignItems: "baseline", gap: "0.5rem",
                  marginBottom: "6px", fontSize: "11px",
                }}>
                  <span style={{ fontFamily: "monospace", color: C.textDim, fontSize: "9px", minWidth: "18px" }}>
                    {i + 1}.
                  </span>
                  <span style={{ color: C.resting, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {f.preview}
                  </span>
                  <span style={{ fontFamily: "monospace", color: "#60A5FA", fontSize: "10px", minWidth: "32px", textAlign: "right" }}>
                    ×{f.retrievals}
                  </span>
                  {f.helpful > 0 && (
                    <span style={{ fontFamily: "monospace", color: C.myelin, fontSize: "9px", minWidth: "24px", textAlign: "right" }}>
                      +{f.helpful}
                    </span>
                  )}
                </div>
              ))}
              <div style={{
                marginTop: "0.6rem", paddingTop: "0.5rem", borderTop: `1px solid ${C.hairline}`,
                display: "flex", justifyContent: "space-between", fontSize: "10px",
                fontFamily: "monospace", color: C.textDim,
              }}>
                <span>avg trust {stats.activity.avg_trust?.toFixed(3)}</span>
                <span>{stats.activity.facts_recalled} recalled of {stats.t4?.facts ?? 0}</span>
              </div>
            </div>
          )}

          {/* Query activity by tool */}
          {stats?.queries?.by_tool && Object.keys(stats.queries.by_tool).length > 0 && (
            <div style={{ padding: "1rem 1.25rem", borderRight: `1px solid ${C.hairline}` }}>
              <div style={{
                fontFamily: "monospace", fontSize: "9px", color: C.textDim,
                letterSpacing: "0.14em", marginBottom: "0.75rem", textTransform: "uppercase",
              }}>
                Agent Queries · Last 24h
              </div>
              {Object.entries(stats.queries.by_tool)
                .sort(([, a], [, b]) => b - a)
                .map(([tool, count]) => {
                  const max = Math.max(...Object.values(stats.queries!.by_tool), 1);
                  const pct = (count / max) * 100;
                  const color = tool.includes("semantic") ? "#A78BFA" : tool.includes("hybrid") ? "#34D399" : C.synapse;
                  return (
                    <div key={tool} style={{ marginBottom: "7px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "2px" }}>
                        <span style={{ color: C.resting, fontFamily: "monospace" }}>{tool}</span>
                        <span style={{ fontFamily: "monospace", color: C.textDim }}>{count}</span>
                      </div>
                      <div style={{ height: "3px", background: "rgba(100,116,139,0.12)", borderRadius: "1px", overflow: "hidden" }}>
                        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: "1px", transition: "width 0.4s ease", boxShadow: `0 0 4px ${color}66` }} />
                      </div>
                    </div>
                  );
                })}
              {/* Hourly sparkline */}
              {stats.queries.by_hour && Object.keys(stats.queries.by_hour).length > 0 && (
                <div style={{ marginTop: "0.75rem" }}>
                  <div style={{ fontFamily: "monospace", fontSize: "8px", color: C.textDim, letterSpacing: "0.12em", marginBottom: "4px" }}>
                    HOURLY
                  </div>
                  <div style={{ display: "flex", alignItems: "flex-end", gap: "1px", height: "24px" }}>
                    {Array.from({ length: 24 }, (_, h) => {
                      const hour = String(h).padStart(2, "0");
                      const count = stats.queries!.by_hour[hour] ?? 0;
                      const maxH = Math.max(...Object.values(stats.queries!.by_hour), 1);
                      const pct = (count / maxH) * 100;
                      return (
                        <div key={h} title={`${hour}:00 — ${count} queries`} style={{
                          flex: 1, height: `${Math.max(pct, count > 0 ? 8 : 2)}%`,
                          background: count > 0 ? "#34D39988" : C.hairline,
                          borderRadius: "1px", transition: "height 0.4s ease",
                          minHeight: "2px",
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

      {/* Footer marker */}
      <div style={{
        padding: "1rem 1.5rem", borderTop: `1px solid ${C.hairline}`,
        fontFamily: "monospace", fontSize: "9px", color: C.textDim, letterSpacing: "0.14em", textTransform: "uppercase",
        display: "flex", justifyContent: "space-between",
      }}>
        <span>Hermes Memory System · 5-tier</span>
        <span>poll {POLL_MS / 1000}s</span>
      </div>
    </div>
  );
}
