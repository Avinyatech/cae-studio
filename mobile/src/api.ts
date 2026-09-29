// Change this to your PC's LAN IP running the FastAPI backend
// (python -m uvicorn app:app --host 0.0.0.0 --port 8000 in nastran-backend/)
export let API_BASE = "http://192.168.68.100:8000";

export function setApiBase(url: string) {
  API_BASE = url.replace(/\/+$/, "");
}

export type Material = "steel" | "aluminum" | "titanium";
export type BC = "cantilever" | "simply_supported" | "fixed_fixed";
export type Analysis = "modes" | "static";

export interface AnalysisRequest {
  length_mm: number;
  width_mm: number;
  thickness_mm: number;
  nx: number;
  ny: number;
  material: Material;
  bc: BC;
  analysis: Analysis;
  num_modes: number;
  freq_max_hz: number;
  load_n: number;
  load_dir: "x" | "y" | "z";
}

export interface Mesh {
  nodes: Record<string, [number, number, number]>;
  elements: number[][];
  fixed_ids: number[];
  nx: number;
  ny: number;
}

export interface ModeResult {
  mode: number;
  freq_hz: number;
  vectors: Record<string, number[]>;
}

export interface AnalysisResponse {
  mesh: Mesh;
  geometry: { length_mm: number; width_mm: number; thickness_mm: number; bc: string };
  analysis: "modes" | "static";
  modes?: ModeResult[];
  static?: { vectors: Record<string, number[]>; max_deflection_mm: number };
}

export async function runAnalysis(req: AnalysisRequest): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function checkHealth(): Promise<{ ok: boolean }> {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}
