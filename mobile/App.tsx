import React, { useState } from "react";
import {
  SafeAreaView,
  ScrollView,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
  Platform,
} from "react-native";
import { Picker } from "@react-native-picker/picker";
import { WebView } from "react-native-webview";
import { StatusBar } from "expo-status-bar";
import {
  API_BASE,
  setApiBase,
  runAnalysis,
  AnalysisRequest,
  AnalysisResponse,
  Material,
  BC,
  Analysis,
} from "./src/api";
import { buildViewerHtml, ViewerFrame } from "./src/modeViewerHtml";

function Field({
  label,
  value,
  onChangeText,
  keyboardType = "numeric",
}: {
  label: string;
  value: string;
  onChangeText: (t: string) => void;
  keyboardType?: "numeric" | "default";
}) {
  return (
    <View style={styles.fieldRow}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <TextInput
        style={styles.fieldInput}
        value={value}
        onChangeText={onChangeText}
        keyboardType={keyboardType}
      />
    </View>
  );
}

export default function App() {
  const [apiBase, setApiBaseInput] = useState(API_BASE);
  const [lengthMm, setLengthMm] = useState("100");
  const [widthMm, setWidthMm] = useState("20");
  const [thicknessMm, setThicknessMm] = useState("2");
  const [nx, setNx] = useState("6");
  const [ny, setNy] = useState("3");
  const [material, setMaterial] = useState<Material>("steel");
  const [bc, setBc] = useState<BC>("cantilever");
  const [analysis, setAnalysis] = useState<Analysis>("modes");
  const [numModes, setNumModes] = useState("6");
  const [loadN, setLoadN] = useState("50");
  const [loadDir, setLoadDir] = useState<"x" | "y" | "z">("z");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  async function onRun() {
    setError(null);
    setLoading(true);
    setResult(null);
    setApiBase(apiBase);
    const req: AnalysisRequest = {
      length_mm: parseFloat(lengthMm),
      width_mm: parseFloat(widthMm),
      thickness_mm: parseFloat(thicknessMm),
      nx: parseInt(nx, 10),
      ny: parseInt(ny, 10),
      material,
      bc,
      analysis,
      num_modes: parseInt(numModes, 10) || 6,
      freq_max_hz: 200000,
      load_n: parseFloat(loadN) || 0,
      load_dir: loadDir,
    };
    try {
      const res = await runAnalysis(req);
      setResult(res);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  let frames: ViewerFrame[] = [];
  if (result?.analysis === "modes" && result.modes) {
    frames = result.modes.map((m) => ({
      label: "Mode " + m.mode,
      freqHz: m.freq_hz,
      vectors: m.vectors,
    }));
  } else if (result?.analysis === "static" && result.static) {
    frames = [{ label: "Static", vectors: result.static.vectors }];
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="auto" />
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        <Text style={styles.h1}>NASTRAN Design Studio</Text>
        <Text style={styles.sub}>Parametric shell design studies, solved on your NASTRAN-95 server</Text>

        <Text style={styles.sectionTitle}>Server</Text>
        <TextInput
          style={styles.apiInput}
          value={apiBase}
          onChangeText={setApiBaseInput}
          autoCapitalize="none"
          autoCorrect={false}
        />

        <Text style={styles.sectionTitle}>Geometry</Text>
        <Field label="Length (mm)" value={lengthMm} onChangeText={setLengthMm} />
        <Field label="Width (mm)" value={widthMm} onChangeText={setWidthMm} />
        <Field label="Thickness (mm)" value={thicknessMm} onChangeText={setThicknessMm} />
        <Field label="Mesh (along length)" value={nx} onChangeText={setNx} />
        <Field label="Mesh (across width)" value={ny} onChangeText={setNy} />

        <Text style={styles.sectionTitle}>Material</Text>
        <View style={styles.pickerWrap}>
          <Picker selectedValue={material} onValueChange={(v) => setMaterial(v)}>
            <Picker.Item label="Steel" value="steel" />
            <Picker.Item label="Aluminum" value="aluminum" />
            <Picker.Item label="Titanium" value="titanium" />
          </Picker>
        </View>

        <Text style={styles.sectionTitle}>Boundary condition</Text>
        <View style={styles.pickerWrap}>
          <Picker selectedValue={bc} onValueChange={(v) => setBc(v)}>
            <Picker.Item label="Cantilever (fixed one end)" value="cantilever" />
            <Picker.Item label="Simply supported (both ends)" value="simply_supported" />
            <Picker.Item label="Fixed-fixed (both ends)" value="fixed_fixed" />
          </Picker>
        </View>

        <Text style={styles.sectionTitle}>Analysis</Text>
        <View style={styles.pickerWrap}>
          <Picker selectedValue={analysis} onValueChange={(v) => setAnalysis(v)}>
            <Picker.Item label="Normal modes (natural frequencies)" value="modes" />
            <Picker.Item label="Static (deflection under load)" value="static" />
          </Picker>
        </View>

        {analysis === "modes" ? (
          <Field label="Number of modes" value={numModes} onChangeText={setNumModes} />
        ) : (
          <>
            <Field label="Load (N)" value={loadN} onChangeText={setLoadN} />
            <Text style={styles.sectionTitle}>Load direction</Text>
            <View style={styles.pickerWrap}>
              <Picker selectedValue={loadDir} onValueChange={(v) => setLoadDir(v)}>
                <Picker.Item label="Z (out-of-plane / transverse bending)" value="z" />
                <Picker.Item label="Y (in-plane, across width)" value="y" />
                <Picker.Item label="X (in-plane, along length)" value="x" />
              </Picker>
            </View>
          </>
        )}

        <TouchableOpacity style={styles.runBtn} onPress={onRun} disabled={loading}>
          {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.runBtnText}>Run Analysis</Text>}
        </TouchableOpacity>

        {error && (
          <View style={styles.errorBox}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {result && (
          <View style={styles.resultsBox}>
            <Text style={styles.sectionTitle}>
              Results — {result.mesh.elements.length} elements, {Object.keys(result.mesh.nodes).length} nodes
            </Text>

            {result.analysis === "modes" && result.modes && (
              <View style={styles.freqTable}>
                {result.modes.map((m) => (
                  <View key={m.mode} style={styles.freqRow}>
                    <Text style={styles.freqMode}>Mode {m.mode}</Text>
                    <Text style={styles.freqHz}>{m.freq_hz.toFixed(1)} Hz</Text>
                  </View>
                ))}
              </View>
            )}

            {result.analysis === "static" && result.static && (
              <Text style={styles.staticSummary}>
                Max deflection: {result.static.max_deflection_mm.toFixed(3)} mm
              </Text>
            )}

            {frames.length > 0 && (
              <View style={styles.webviewWrap}>
                <WebView
                  originWhitelist={["*"]}
                  source={{ html: buildViewerHtml(result.mesh, frames) }}
                  style={styles.webview}
                  scrollEnabled={false}
                />
              </View>
            )}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#ffffff" },
  scroll: { padding: 16, paddingBottom: 60 },
  h1: { fontSize: 22, fontWeight: "700", color: "#1a1a1a" },
  sub: { fontSize: 13, color: "#6b6b6b", marginTop: 2, marginBottom: 14 },
  sectionTitle: { fontSize: 13, fontWeight: "700", color: "#1a1a1a", marginTop: 14, marginBottom: 6 },
  apiInput: {
    borderWidth: 1, borderColor: "#e5e5e5", borderRadius: 8, padding: 10, fontSize: 13, color: "#1a1a1a",
  },
  fieldRow: { flexDirection: "row", alignItems: "center", marginBottom: 8 },
  fieldLabel: { flex: 1, fontSize: 14, color: "#1a1a1a" },
  fieldInput: {
    width: 110, borderWidth: 1, borderColor: "#e5e5e5", borderRadius: 8,
    padding: 8, fontSize: 14, textAlign: "right", color: "#1a1a1a",
  },
  pickerWrap: {
    borderWidth: 1, borderColor: "#e5e5e5", borderRadius: 8,
    overflow: "hidden", ...(Platform.OS === "ios" ? {} : { justifyContent: "center" }),
  },
  runBtn: {
    backgroundColor: "#3b82f6", borderRadius: 10, paddingVertical: 14,
    alignItems: "center", marginTop: 20,
  },
  runBtnText: { color: "#fff", fontWeight: "700", fontSize: 15 },
  errorBox: { backgroundColor: "#fef2f2", borderRadius: 8, padding: 12, marginTop: 14 },
  errorText: { color: "#b91c1c", fontSize: 12 },
  resultsBox: { marginTop: 20 },
  freqTable: { borderWidth: 1, borderColor: "#e5e5e5", borderRadius: 8, overflow: "hidden" },
  freqRow: {
    flexDirection: "row", justifyContent: "space-between", padding: 10,
    borderBottomWidth: 1, borderBottomColor: "#f0f0f0",
  },
  freqMode: { fontSize: 13, color: "#6b6b6b" },
  freqHz: { fontSize: 13, fontWeight: "700", color: "#1a1a1a" },
  staticSummary: { fontSize: 15, fontWeight: "700", color: "#1a1a1a" },
  webviewWrap: {
    height: 400, marginTop: 14, borderRadius: 10, overflow: "hidden",
    borderWidth: 1, borderColor: "#e5e5e5",
  },
  webview: { flex: 1, backgroundColor: "#ffffff" },
});
