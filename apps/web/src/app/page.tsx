"use client";

import {
  Activity,
  AlertTriangle,
  AudioWaveform,
  BarChart3,
  Cloud,
  Database,
  FileAudio,
  Fingerprint,
  Gauge,
  HeartPulse,
  History,
  KeyRound,
  Languages,
  LoaderCircle,
  MessageSquareText,
  Mic,
  PlayCircle,
  Plus,
  RotateCcw,
  Search,
  Server,
  ShieldCheck,
  SlidersHorizontal,
  Square,
  Trash2,
  Upload
} from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8001";

type View = "analyze" | "dashboard" | "history" | "rules" | "deploy";
type RiskLevel = "normal" | "low" | "medium" | "high";

interface RiskFactor {
  group: string;
  keyword: string;
}

interface TextAnalysisResult {
  risk_score: number;
  risk_level: RiskLevel;
  factors: RiskFactor[];
  suggestion: string;
  model: string;
  model_score: number | null;
  model_prediction: "normal" | "fraud" | null;
  rule_score: number;
  evidence_terms: string[];
}

interface AudioAnalysisResult {
  file_name: string;
  prediction: "normal" | "fraud";
  risk_score: number;
  risk_level: RiskLevel;
  probabilities: {
    normal: number;
    fraud: number;
  };
  model: string;
  feature_window_seconds: number;
  suggestion: string;
  notes: string[];
}

interface EmotionPressureDriver {
  emotion: string;
  probability: number;
  pressure_weight: number;
  contribution: number;
}

interface EmotionAnalysisResult {
  file_name: string;
  primary_emotion: string;
  emotion_confidence: number;
  pressure_score: number;
  pressure_level: RiskLevel;
  probabilities: Record<string, number>;
  pressure_probabilities: Record<string, number>;
  pressure_drivers: EmotionPressureDriver[];
  model: string;
  feature_window_seconds: number;
  suggestion: string;
  notes: string[];
}

interface TranscriptionResult {
  text: string;
  raw_text?: string | null;
  source: string;
  language: string;
  language_probability: number;
  duration_seconds: number;
  model: string;
  segments: Array<{
    start: number;
    end: number;
    text: string;
  }>;
}

interface RiskTimelineSegment {
  index: number;
  start: number;
  end: number;
  risk_score: number;
  risk_level: RiskLevel;
  audio_score: number | null;
  text_score: number | null;
  pressure_score: number | null;
  transcript: string;
  factors: RiskFactor[];
}

interface RiskTimelineSummary {
  duration_seconds: number;
  window_seconds: number;
  hop_seconds: number;
  segment_count: number;
  medium_or_high_segments: number;
  high_risk_segments: number;
  peak_risk_score: number;
  peak_start: number | null;
  peak_end: number | null;
}

interface CallAnalysisResult {
  record_id: number | null;
  prediction: "normal" | "fraud";
  risk_score: number;
  risk_level: RiskLevel;
  transcript: string;
  transcript_source: "manual" | "asr" | "none";
  asr: TranscriptionResult | null;
  audio: AudioAnalysisResult | null;
  emotion: EmotionAnalysisResult | null;
  text: TextAnalysisResult | null;
  fusion_weights: {
    audio: number;
    text: number;
  };
  fusion_method: string;
  decision_threshold: number;
  fusion_diagnostics: {
    audio_confidence: number;
    text_confidence: number;
    agreement: number;
    rule_adjustment: number;
  } | null;
  timeline: RiskTimelineSegment[];
  timeline_summary: RiskTimelineSummary | null;
  suggestion: string;
  notes: string[];
}

interface DemoSample {
  id: string;
  title: string;
  scenario: string;
  label: string;
  description: string;
  transcript: string;
  has_audio: boolean;
  audio_file_name: string | null;
}

interface CallRecordSummary {
  id: number;
  created_at: string;
  input_type: string;
  file_name: string | null;
  prediction: "normal" | "fraud";
  risk_score: number;
  risk_level: RiskLevel;
  pressure_score: number | null;
  pressure_level: RiskLevel | null;
  transcript_preview: string;
  risk_factors: RiskFactor[];
  model_summary: Record<string, string | number | null>;
}

interface CallRecordDetail extends CallRecordSummary {
  transcript: string;
  analysis_result: CallAnalysisResult;
}

interface CallListResponse {
  records: CallRecordSummary[];
  total: number;
  limit: number;
  offset: number;
}

interface AnalyticsSummary {
  total_calls: number;
  high_risk_calls: number;
  average_risk_score: number;
  risk_level_counts: Record<string, number>;
  prediction_counts: Record<string, number>;
  top_keywords: Array<{ keyword: string; count: number }>;
  recent_calls: CallRecordSummary[];
}

interface FeatureStatus {
  id: string;
  name: string;
  enabled: boolean;
  status: string;
  detail: string;
}

interface DeploymentStatus {
  mode: string;
  demo_mode: boolean;
  api_version: string;
  database_path: string;
  features: FeatureStatus[];
  limitations: string[];
  next_steps: string[];
}

interface RuleRecord {
  id: number;
  group: string;
  keyword: string;
  weight: number;
  enabled: boolean;
  source: "default" | "custom" | string;
  created_at: string;
  updated_at: string;
}

const levelCopy: Record<RiskLevel, string> = {
  normal: "正常",
  low: "低风险",
  medium: "中风险",
  high: "高风险"
};

const levelTone: Record<RiskLevel, string> = {
  normal: "border-emerald-200 bg-emerald-50 text-emerald-700",
  low: "border-sky-200 bg-sky-50 text-sky-700",
  medium: "border-amber-200 bg-amber-50 text-amber-800",
  high: "border-rose-200 bg-rose-50 text-rose-700"
};

const demoLabelCopy: Record<string, string> = {
  normal: "正常",
  fraud: "诈骗",
  high_pressure: "高压",
  normal_pressure: "平稳"
};

const demoLabelTone: Record<string, string> = {
  normal: "border-emerald-200 bg-emerald-50 text-emerald-700",
  fraud: "border-rose-200 bg-rose-50 text-rose-700",
  high_pressure: "border-amber-200 bg-amber-50 text-amber-800",
  normal_pressure: "border-sky-200 bg-sky-50 text-sky-700"
};

const navItems: Array<{ id: View; label: string; icon: ReactNode }> = [
  { id: "analyze", label: "分析台", icon: <ShieldCheck size={16} /> },
  { id: "dashboard", label: "数据看板", icon: <BarChart3 size={16} /> },
  { id: "history", label: "历史记录", icon: <History size={16} /> },
  { id: "rules", label: "规则管理", icon: <SlidersHorizontal size={16} /> },
  { id: "deploy", label: "部署拓展", icon: <Cloud size={16} /> }
];

export default function Home() {
  const [activeView, setActiveView] = useState<View>("analyze");
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [transcript, setTranscript] = useState("");
  const [result, setResult] = useState<CallAnalysisResult | null>(null);
  const [demoSamples, setDemoSamples] = useState<DemoSample[]>([]);
  const [loadingDemoId, setLoadingDemoId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [savedNotice, setSavedNotice] = useState<string | null>(null);
  const [records, setRecords] = useState<CallRecordSummary[]>([]);
  const [recordsTotal, setRecordsTotal] = useState(0);
  const [selectedRecord, setSelectedRecord] = useState<CallRecordDetail | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [deploymentStatus, setDeploymentStatus] = useState<DeploymentStatus | null>(null);
  const [rules, setRules] = useState<RuleRecord[]>([]);
  const [ruleFilter, setRuleFilter] = useState("");
  const [ruleForm, setRuleForm] = useState({ group: "money_transfer", keyword: "", weight: "1.0" });
  const [panelError, setPanelError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const recordingChunksRef = useRef<Float32Array[]>([]);
  const recordingSampleRateRef = useRef(44100);
  const recordingTimerRef = useRef<number | null>(null);

  const riskPercent = Math.round((result?.risk_score ?? 0) * 100);
  const audioPercent = Math.round((result?.audio?.risk_score ?? 0) * 100);
  const textPercent = Math.round((result?.text?.risk_score ?? 0) * 100);
  const stressPercent = Math.round((result?.emotion?.pressure_score ?? 0) * 100);
  const selectedName = file?.name ?? "未选择文件";

  const signalBars = useMemo(() => {
    const base = result ? riskPercent : 30;
    return Array.from({ length: 44 }, (_, index) => {
      const wave = Math.abs(Math.sin(index * 0.6)) * 36;
      const pulse = ((index * 19 + base) % 31) + 14;
      return Math.min(84, Math.round(wave + pulse));
    });
  }, [result, riskPercent]);

  const filteredRules = useMemo(() => {
    const query = ruleFilter.trim().toLowerCase();
    if (!query) {
      return rules;
    }
    return rules.filter(
      (rule) =>
        rule.group.toLowerCase().includes(query) ||
        rule.keyword.toLowerCase().includes(query) ||
        rule.source.toLowerCase().includes(query)
    );
  }, [ruleFilter, rules]);

  useEffect(() => {
    void loadDemoSamples();
    void loadDeploymentStatus();
    void refreshOverview();
  }, []);

  useEffect(() => {
    if (activeView === "dashboard") {
      void loadAnalytics();
    }
    if (activeView === "history") {
      void loadRecords();
    }
    if (activeView === "rules") {
      void loadRules();
    }
  }, [activeView]);

  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
      closeRecordingGraph();
    };
  }, [audioUrl]);

  async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${path}`, init);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail ?? "请求失败");
    }
    return payload as T;
  }

  async function loadDemoSamples() {
    try {
      const payload = await apiJson<DemoSample[]>("/api/demo/samples");
      setDemoSamples(Array.isArray(payload) ? payload : []);
    } catch {
      setDemoSamples([]);
    }
  }

  async function loadDeploymentStatus() {
    try {
      setDeploymentStatus(await apiJson<DeploymentStatus>("/api/deployment/status"));
    } catch {
      setDeploymentStatus(null);
    }
  }

  async function refreshOverview() {
    await Promise.allSettled([loadAnalytics(), loadRecords()]);
  }

  async function loadAnalytics() {
    try {
      setAnalytics(await apiJson<AnalyticsSummary>("/api/analytics/summary"));
      setPanelError(null);
    } catch (err) {
      setPanelError(err instanceof Error ? err.message : "数据看板加载失败");
    }
  }

  async function loadRecords() {
    try {
      const payload = await apiJson<CallListResponse>("/api/calls?limit=30&offset=0");
      setRecords(payload.records);
      setRecordsTotal(payload.total);
      setPanelError(null);
    } catch (err) {
      setPanelError(err instanceof Error ? err.message : "历史记录加载失败");
    }
  }

  async function loadRecordDetail(recordId: number) {
    try {
      setSelectedRecord(await apiJson<CallRecordDetail>(`/api/calls/${recordId}`));
      setPanelError(null);
    } catch (err) {
      setPanelError(err instanceof Error ? err.message : "记录详情加载失败");
    }
  }

  async function deleteRecord(recordId: number) {
    try {
      await apiJson(`/api/calls/${recordId}`, { method: "DELETE" });
      if (selectedRecord?.id === recordId) {
        setSelectedRecord(null);
      }
      await refreshOverview();
    } catch (err) {
      setPanelError(err instanceof Error ? err.message : "删除失败");
    }
  }

  async function loadRules() {
    try {
      setRules(await apiJson<RuleRecord[]>("/api/rules"));
      setPanelError(null);
    } catch (err) {
      setPanelError(err instanceof Error ? err.message : "规则加载失败");
    }
  }

  async function createRule() {
    if (!ruleForm.keyword.trim()) {
      setPanelError("请先输入关键词");
      return;
    }
    try {
      await apiJson<RuleRecord>("/api/rules", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          group: ruleForm.group.trim(),
          keyword: ruleForm.keyword.trim(),
          weight: Number(ruleForm.weight) || 1,
          enabled: true
        })
      });
      setRuleForm((value) => ({ ...value, keyword: "" }));
      await loadRules();
    } catch (err) {
      setPanelError(err instanceof Error ? err.message : "新增规则失败");
    }
  }

  async function toggleRule(ruleId: number) {
    try {
      await apiJson<RuleRecord>(`/api/rules/${ruleId}/toggle`, { method: "PATCH" });
      await loadRules();
    } catch (err) {
      setPanelError(err instanceof Error ? err.message : "切换规则失败");
    }
  }

  async function deleteRule(ruleId: number) {
    try {
      await apiJson(`/api/rules/${ruleId}`, { method: "DELETE" });
      await loadRules();
    } catch (err) {
      setPanelError(err instanceof Error ? err.message : "删除规则失败");
    }
  }

  async function resetDefaultRules() {
    try {
      const payload = await apiJson<RuleRecord[]>("/api/rules/reset-defaults", { method: "POST" });
      setRules(payload);
      setPanelError(null);
    } catch (err) {
      setPanelError(err instanceof Error ? err.message : "恢复默认规则失败");
    }
  }

  function selectFile(nextFile: File | null) {
    if (nextFile) {
      setAudioFile(nextFile);
    }
  }

  function setAudioFile(nextFile: File) {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setFile(nextFile);
    setAudioUrl(URL.createObjectURL(nextFile));
    setResult(null);
    setSavedNotice(null);
    setError(null);
  }

  function clearAudioFile() {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setFile(null);
    setAudioUrl(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  function reset() {
    setIsRecording(false);
    closeRecordingGraph();
    clearAudioFile();
    setTranscript("");
    setResult(null);
    setError(null);
    setSavedNotice(null);
    setRecordingSeconds(0);
  }

  async function startRecording() {
    const AudioContextConstructor = getAudioContextConstructor();
    if (!navigator.mediaDevices?.getUserMedia || !AudioContextConstructor) {
      setError("当前浏览器不支持录音");
      return;
    }

    try {
      closeRecordingGraph();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new AudioContextConstructor();
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);

      recordingChunksRef.current = [];
      recordingSampleRateRef.current = audioContext.sampleRate;
      processor.onaudioprocess = (event) => {
        recordingChunksRef.current.push(new Float32Array(event.inputBuffer.getChannelData(0)));
        event.outputBuffer.getChannelData(0).fill(0);
      };

      source.connect(processor);
      processor.connect(audioContext.destination);
      audioContextRef.current = audioContext;
      mediaStreamRef.current = stream;
      mediaSourceRef.current = source;
      processorRef.current = processor;
      setIsRecording(true);
      setRecordingSeconds(0);
      setResult(null);
      setSavedNotice(null);
      setError(null);
      recordingTimerRef.current = window.setInterval(
        () => setRecordingSeconds((value) => value + 1),
        1000
      );
    } catch {
      closeRecordingGraph();
      setIsRecording(false);
      setError("无法获取麦克风权限");
    }
  }

  function stopRecording() {
    if (!isRecording) {
      return;
    }
    setIsRecording(false);
    const chunks = recordingChunksRef.current;
    const sampleRate = recordingSampleRateRef.current;
    closeRecordingGraph();

    if (chunks.length < 2) {
      setError("录音时间太短，请重新录制");
      return;
    }

    const samples = mergeAudioChunks(chunks);
    const wavBlob = encodeWav(samples, sampleRate);
    const nextFile = new File([wavBlob], `callguard-recording-${Date.now()}.wav`, {
      type: "audio/wav"
    });
    setAudioFile(nextFile);
  }

  function closeRecordingGraph() {
    if (recordingTimerRef.current !== null) {
      window.clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    processorRef.current?.disconnect();
    mediaSourceRef.current?.disconnect();
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      void audioContextRef.current.close();
    }
    processorRef.current = null;
    mediaSourceRef.current = null;
    mediaStreamRef.current = null;
    audioContextRef.current = null;
  }

  async function loadDemoSample(sample: DemoSample) {
    setLoadingDemoId(sample.id);
    setResult(null);
    setSavedNotice(null);
    setError(null);
    setTranscript(sample.transcript);

    try {
      if (sample.has_audio) {
        const response = await fetch(`${API_BASE_URL}/api/demo/samples/${sample.id}/audio`);
        if (!response.ok) {
          throw new Error("样例加载失败");
        }
        const blob = await response.blob();
        setAudioFile(
          new File([blob], sample.audio_file_name ?? `${sample.id}.wav`, {
            type: blob.type || "audio/wav"
          })
        );
      } else {
        clearAudioFile();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "样例加载失败");
    } finally {
      setLoadingDemoId(null);
    }
  }

  async function analyzeCall() {
    if (!file && !transcript.trim()) {
      setError("请先提供音频或通话文本");
      return;
    }

    setIsLoading(true);
    setError(null);
    setSavedNotice(null);

    const formData = new FormData();
    if (file) {
      formData.append("file", file);
    }
    formData.append("transcript", transcript);

    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze/call`, {
        method: "POST",
        body: formData
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "分析失败");
      }
      const nextResult = payload as CallAnalysisResult;
      setResult(nextResult);
      setSavedNotice(
        nextResult.record_id ? `已保存到历史记录 #${nextResult.record_id}` : "本次结果未写入历史记录"
      );
      await refreshOverview();
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "分析失败");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#f7f8fa] text-[#171b22]">
      <header className="border-b border-[#dde2ea] bg-white">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-md bg-[#0f766e] text-white shadow-sm">
                <ShieldCheck size={22} />
              </div>
              <div>
                <p className="text-sm text-[#667085]">Call risk awareness</p>
                <h1 className="text-xl font-semibold tracking-normal">CallGuard</h1>
              </div>
            </div>
            <nav className="flex flex-wrap items-center gap-2">
              {navItems.map((item) => (
                <button
                  className={`inline-flex h-10 items-center gap-2 rounded-md border px-3 text-sm font-medium transition ${
                    activeView === item.id
                      ? "border-[#0f766e] bg-[#e6f4f1] text-[#0f766e]"
                      : "border-[#cfd7e3] bg-white text-[#344054] hover:bg-[#f2f5f8]"
                  }`}
                  key={item.id}
                  onClick={() => setActiveView(item.id)}
                  type="button"
                >
                  {item.icon}
                  {item.label}
                </button>
              ))}
              <button
                className="inline-flex h-10 items-center gap-2 rounded-md border border-[#cfd7e3] bg-white px-3 text-sm font-medium text-[#344054] transition hover:bg-[#f2f5f8]"
                onClick={reset}
                type="button"
              >
                <RotateCcw size={16} />
                重置
              </button>
            </nav>
          </div>
        </div>
      </header>

      {activeView === "analyze" ? renderAnalyzeView() : null}
      {activeView === "dashboard" ? renderDashboardView() : null}
      {activeView === "history" ? renderHistoryView() : null}
      {activeView === "rules" ? renderRulesView() : null}
      {activeView === "deploy" ? renderDeployView() : null}
    </main>
  );

  function renderAnalyzeView() {
    const factors = result?.text?.factors ?? [];
    const evidenceTerms = result?.text?.evidence_terms ?? [];
    const emotionDrivers = result?.emotion?.pressure_drivers ?? [];
    const pressureProbabilities = result?.emotion?.pressure_probabilities ?? {};

    return (
      <section className="mx-auto grid max-w-7xl gap-6 px-6 py-8 lg:grid-cols-[410px_1fr]">
        <aside className="space-y-5">
          <Panel icon={<FileAudio size={20} />} title="音频">
            <input
              id="callguard-audio-upload"
              ref={inputRef}
              accept="audio/*"
              className="sr-only"
              onChange={(event) => selectFile(event.target.files?.[0] ?? null)}
              type="file"
            />
            <label
              className="grid min-h-44 w-full cursor-pointer place-items-center rounded-md border border-dashed border-[#98a2b3] bg-[#fbfcfe] p-6 text-center transition hover:border-[#0f766e] hover:bg-[#f3fbf9]"
              htmlFor="callguard-audio-upload"
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                selectFile(event.dataTransfer.files?.[0] ?? null);
              }}
            >
              <div>
                <AudioWaveform className="mx-auto mb-3 text-[#0f766e]" size={36} />
                <p className="font-medium">{selectedName}</p>
                <p className="mt-1 text-sm text-[#667085]">mp3 / wav / m4a</p>
              </div>
            </label>

            {audioUrl ? <audio className="mt-4 w-full" controls src={audioUrl} /> : null}

            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <button
                className="inline-flex items-center justify-center gap-2 rounded-md border border-[#cfd7e3] bg-white px-3 py-2.5 text-sm font-medium text-[#344054] transition hover:bg-[#f2f5f8] disabled:cursor-not-allowed disabled:bg-[#f2f5f8]"
                disabled={isRecording || isLoading}
                onClick={startRecording}
                type="button"
              >
                <Mic size={16} />
                开始录音
              </button>
              <button
                className="inline-flex items-center justify-center gap-2 rounded-md border border-[#fecdd3] bg-[#fff1f2] px-3 py-2.5 text-sm font-medium text-[#be123c] transition hover:bg-[#ffe4e6] disabled:cursor-not-allowed disabled:border-[#dde2ea] disabled:bg-[#f2f5f8] disabled:text-[#98a2b3]"
                disabled={!isRecording}
                onClick={stopRecording}
                type="button"
              >
                <Square size={15} />
                停止录音
              </button>
            </div>

            {isRecording ? (
              <div className="mt-3 flex items-center justify-between rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
                <span>正在录音</span>
                <span className="font-semibold">{formatDuration(recordingSeconds)}</span>
              </div>
            ) : null}
          </Panel>

          <Panel icon={<MessageSquareText size={20} />} title="通话文本">
            <textarea
              className="min-h-36 w-full resize-y rounded-md border border-[#cfd7e3] bg-[#fbfcfe] p-3 text-sm leading-6 outline-none transition placeholder:text-[#98a2b3] focus:border-[#0f766e] focus:bg-white"
              onChange={(event) => {
                setTranscript(event.target.value);
                setResult(null);
                setSavedNotice(null);
                setError(null);
              }}
              placeholder="可选：输入或修正通话文本。留空时系统会尝试自动转写音频。"
              value={transcript}
            />
            <button
              className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-md bg-[#0f766e] px-4 py-3 text-sm font-semibold text-white transition hover:bg-[#0b615a] disabled:cursor-not-allowed disabled:bg-[#98a2b3]"
              disabled={(!file && !transcript.trim()) || isLoading}
              onClick={analyzeCall}
              type="button"
            >
              {isLoading ? <LoaderCircle className="animate-spin" size={17} /> : <Upload size={17} />}
              分析通话
            </button>
            {savedNotice ? (
              <div className="mt-4 rounded-md border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-700">
                {savedNotice}
              </div>
            ) : null}
            {error ? (
              <div className="mt-4 rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
                {error}
              </div>
            ) : null}
          </Panel>

          <Panel icon={<PlayCircle size={20} />} title="演示样例">
            <div className="space-y-2">
              {demoSamples.map((sample) => (
                <button
                  className="w-full rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-3 text-left transition hover:border-[#0f766e] hover:bg-[#f3fbf9] disabled:cursor-wait disabled:opacity-70"
                  disabled={loadingDemoId !== null || isLoading || isRecording}
                  key={sample.id}
                  onClick={() => void loadDemoSample(sample)}
                  type="button"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="font-medium text-[#171b22]">{sample.title}</p>
                      <p className="mt-1 text-xs leading-5 text-[#667085]">{sample.description}</p>
                    </div>
                    <span
                      className={`shrink-0 rounded-md border px-2 py-0.5 text-xs font-medium ${
                        demoLabelTone[sample.label] ?? "border-[#dde2ea] bg-[#f2f5f8] text-[#667085]"
                      }`}
                    >
                      {demoLabelCopy[sample.label] ?? sample.label}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center gap-2 text-xs text-[#667085]">
                    <span>{sample.has_audio ? sample.audio_file_name : "纯文本"}</span>
                    {loadingDemoId === sample.id ? <span>载入中</span> : null}
                  </div>
                </button>
              ))}
            </div>
          </Panel>

          <Panel icon={<Activity size={20} />} title="信号概览">
            <div className="flex h-24 items-end gap-1 rounded-md bg-[#f2f5f8] px-3 py-4">
              {signalBars.map((height, index) => (
                <div
                  className="flex-1 rounded-t bg-[#0f766e]"
                  key={`${height}-${index}`}
                  style={{ height: `${height}%`, opacity: 0.28 + index / 88 }}
                />
              ))}
            </div>
          </Panel>
        </aside>

        <div className="space-y-6">
          <section className="grid gap-4 md:grid-cols-4">
            <Metric icon={<Gauge size={18} />} label="融合风险" unit="/100" value={result ? String(riskPercent) : "--"} />
            <Metric icon={<AudioWaveform size={18} />} label="音频风险" unit="/100" value={result?.audio ? String(audioPercent) : "--"} />
            <Metric icon={<MessageSquareText size={18} />} label="文本风险" unit="/100" value={result?.text ? String(textPercent) : "--"} />
            <Metric icon={<HeartPulse size={18} />} label="压力信号" unit="/100" value={result?.emotion ? String(stressPercent) : "--"} />
          </section>

          <Panel title="综合判断">
            <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-sm text-[#667085]">Fusion risk</p>
                <h2 className="text-lg font-semibold">综合判断</h2>
              </div>
              <RiskBadge level={result?.risk_level} fallback="等待分析" />
            </div>
            <div className="grid gap-6 lg:grid-cols-[1fr_290px]">
              <div>
                <ProgressBar value={result ? riskPercent : 0} level={result?.risk_level ?? "normal"} />
                <div className="mt-5 rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4">
                  <p className="text-sm text-[#667085]">建议</p>
                  <p className="mt-2 font-medium leading-7">
                    {result?.suggestion ?? "上传音频或输入通话文本后开始分析。"}
                  </p>
                </div>
              </div>
              <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4">
                <p className="text-sm text-[#667085]">Prediction</p>
                <p className="mt-2 text-2xl font-semibold">
                  {result ? (result.prediction === "fraud" ? "风险通话" : "正常通话") : "--"}
                </p>
                <div className="mt-5 space-y-3 text-sm text-[#475467]">
                  <InfoRow label="Transcript" value={result?.transcript_source ?? "--"} />
                  <InfoRow label="Audio weight" value={result ? `${Math.round(result.fusion_weights.audio * 100)}%` : "--"} />
                  <InfoRow label="Text weight" value={result ? `${Math.round(result.fusion_weights.text * 100)}%` : "--"} />
                  <InfoRow label="Pressure" value={result?.emotion ? `${stressPercent}/100` : "--"} />
                  <InfoRow label="Text model" value={result?.text?.model ?? "--"} />
                  <InfoRow label="Fusion" value={result?.fusion_method ?? "--"} />
                </div>
              </div>
            </div>
          </Panel>

          <TimelinePanel result={result} />

          <Panel icon={<MessageSquareText size={20} />} title="自动转写">
            <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4">
              <p className="min-h-16 text-sm leading-7 text-[#344054]">
                {result?.transcript || "上传音频或输入通话文本后开始分析。"}
              </p>
              <div className="mt-4 grid gap-3 text-sm text-[#475467] md:grid-cols-3">
                <InfoRow label="Source" value={result?.transcript_source ?? "--"} />
                <InfoRow label="ASR model" value={result?.asr?.model ?? "--"} />
                <InfoRow
                  label="Language"
                  value={result?.asr ? `${result.asr.language} ${Math.round(result.asr.language_probability * 100)}%` : "--"}
                />
              </div>
            </div>
          </Panel>

          <Panel icon={<HeartPulse size={20} />} title="语音压力/情绪辅助">
            <div className="grid gap-6 lg:grid-cols-[230px_1fr]">
              <div>
                <p className="text-sm text-[#667085]">Pressure score</p>
                <div className="mt-3 flex items-end gap-1">
                  <span className="text-5xl font-semibold">{result?.emotion ? stressPercent : "--"}</span>
                  <span className="pb-1 text-sm text-[#667085]">/100</span>
                </div>
                <div className="mt-4">
                  <ProgressBar value={result?.emotion ? stressPercent : 0} level={result?.emotion?.pressure_level ?? "normal"} compact />
                </div>
              </div>
              <div className="space-y-5">
                <div className="grid gap-3 text-sm text-[#475467] md:grid-cols-3">
                  <InfoRow label="主要情绪" value={result?.emotion?.primary_emotion ?? "--"} />
                  <InfoRow label="情绪置信度" value={result?.emotion ? `${Math.round(result.emotion.emotion_confidence * 100)}%` : "--"} />
                  <InfoRow label="压力模型" value={result?.emotion?.model ?? "--"} />
                </div>
                <div className="grid gap-3 md:grid-cols-3">
                  {emotionDrivers.length > 0 ? (
                    emotionDrivers.map((driver) => (
                      <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-3" key={driver.emotion}>
                        <p className="font-medium">{driver.emotion}</p>
                        <p className="mt-1 text-sm text-[#667085]">
                          {Math.round(driver.probability * 100)}% / {Math.round(driver.contribution * 100)}/100
                        </p>
                      </div>
                    ))
                  ) : (
                    <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-3 text-sm text-[#667085]">
                      暂无压力来源。
                    </div>
                  )}
                </div>
                <div className="grid gap-3 md:grid-cols-3">
                  {Object.entries(pressureProbabilities).map(([label, value]) => (
                    <div className="rounded-md bg-[#f2f5f8] p-3 text-sm text-[#475467]" key={label}>
                      <p>{levelCopy[label as RiskLevel] ?? label}</p>
                      <p className="mt-1 font-semibold text-[#171b22]">{Math.round(value * 100)}%</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Panel>

          <section className="grid gap-6 lg:grid-cols-[1fr_340px]">
            <Panel icon={<AlertTriangle size={20} />} title="风险因素">
              <div className="grid gap-3 md:grid-cols-2">
                {factors.length > 0 ? (
                  factors.map((factor) => (
                    <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-3" key={`${factor.group}-${factor.keyword}`}>
                      <p className="font-medium">{factor.keyword}</p>
                      <p className="mt-1 text-sm text-[#667085]">{factor.group}</p>
                    </div>
                  ))
                ) : (
                  <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-3 text-sm text-[#667085]">
                    暂无文本风险因素。
                  </div>
                )}
              </div>
              {evidenceTerms.length > 0 ? (
                <div className="mt-5 rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4">
                  <p className="text-sm font-medium text-[#344054]">模型证据</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {evidenceTerms.map((term) => (
                      <span className="rounded-md bg-[#e6f4f1] px-2.5 py-1 text-sm font-medium text-[#0f766e]" key={term}>
                        {term}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
            </Panel>

            <Panel icon={<Database size={20} />} title="模型状态">
              <div className="space-y-3 text-sm text-[#475467]">
                <InfoRow label="Audio" value={result?.audio ? `${audioPercent}/100` : "--"} />
                <InfoRow label="Pressure" value={result?.emotion ? `${stressPercent}/100` : "--"} />
                <InfoRow label="Text" value={result?.text ? `${textPercent}/100` : "--"} />
                <InfoRow label="Rule" value={result?.text ? `${Math.round(result.text.rule_score * 100)}/100` : "--"} />
                <InfoRow label="Label" value={result ? levelCopy[result.risk_level] : "--"} />
                <InfoRow label="Record" value={result?.record_id ? `#${result.record_id}` : "--"} />
              </div>
              <div className="mt-5 space-y-3">
                {(result?.notes ?? ["Audio baseline ready.", "ASR and text ML ready."]).map((item) => (
                  <div className="rounded-md bg-[#f2f5f8] p-3 text-sm leading-6 text-[#475467]" key={item}>
                    {item}
                  </div>
                ))}
              </div>
            </Panel>
          </section>
        </div>
      </section>
    );
  }

  function renderDashboardView() {
    const data = analytics;
    const distribution: Array<{ key: RiskLevel; label: string; count: number }> = ([
      "high",
      "medium",
      "low",
      "normal"
    ] as RiskLevel[]).map((key) => ({
      key,
      label: levelCopy[key],
      count: data?.risk_level_counts?.[key] ?? 0
    }));
    const maxCount = Math.max(1, ...distribution.map((item) => item.count));

    return (
      <section className="mx-auto max-w-7xl px-6 py-8">
        <ViewHeader title="数据看板" description="从本地历史记录聚合风险分布、近期分析和高频风险关键词。" />
        {panelError ? <ErrorBox message={panelError} /> : null}
        <section className="grid gap-4 md:grid-cols-4">
          <Metric icon={<Database size={18} />} label="总分析次数" unit="次" value={String(data?.total_calls ?? 0)} />
          <Metric icon={<AlertTriangle size={18} />} label="高风险次数" unit="次" value={String(data?.high_risk_calls ?? 0)} />
          <Metric icon={<Gauge size={18} />} label="平均风险分" unit="/100" value={String(Math.round((data?.average_risk_score ?? 0) * 100))} />
          <Metric icon={<ShieldCheck size={18} />} label="最近记录" unit="条" value={String(data?.recent_calls.length ?? 0)} />
        </section>

        {(data?.total_calls ?? 0) === 0 ? (
          <EmptyState title="还没有历史数据" description="先回到分析台，点击 demo 或上传音频生成一次分析记录。" />
        ) : (
          <section className="mt-6 grid gap-6 lg:grid-cols-[1fr_380px]">
            <Panel icon={<BarChart3 size={20} />} title="风险等级分布">
              <div className="space-y-4">
                {distribution.map((item) => (
                  <div key={item.key}>
                    <div className="mb-2 flex items-center justify-between text-sm">
                      <span className="font-medium">{item.label}</span>
                      <span className="text-[#667085]">{item.count} 次</span>
                    </div>
                    <div className="h-3 overflow-hidden rounded-full bg-[#e7ecf2]">
                      <div
                        className={`h-full rounded-full ${
                          item.key === "high"
                            ? "bg-[#e11d48]"
                            : item.key === "medium"
                              ? "bg-[#d97706]"
                              : item.key === "low"
                                ? "bg-[#0284c7]"
                                : "bg-[#0f766e]"
                        }`}
                        style={{ width: `${(item.count / maxCount) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel icon={<Search size={20} />} title="Top 风险关键词">
              <div className="space-y-2">
                {(data?.top_keywords ?? []).length > 0 ? (
                  data?.top_keywords.map((item) => (
                    <div className="flex items-center justify-between rounded-md bg-[#f2f5f8] px-3 py-2 text-sm" key={item.keyword}>
                      <span className="font-medium">{item.keyword}</span>
                      <span className="text-[#667085]">{item.count} 次</span>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-[#667085]">暂无命中关键词。</p>
                )}
              </div>
            </Panel>

            <Panel icon={<History size={20} />} title="最近分析">
              <RecordTable records={data?.recent_calls ?? []} onOpen={(id) => {
                setActiveView("history");
                void loadRecordDetail(id);
              }} onDelete={(id) => void deleteRecord(id)} />
            </Panel>
          </section>
        )}
      </section>
    );
  }

  function renderHistoryView() {
    return (
      <section className="mx-auto max-w-7xl px-6 py-8">
        <ViewHeader title="历史记录" description={`本地共保存 ${recordsTotal} 条分析记录，可回看详情或删除。`} />
        {panelError ? <ErrorBox message={panelError} /> : null}
        <div className="grid gap-6 lg:grid-cols-[1fr_420px]">
          <Panel icon={<History size={20} />} title="记录列表">
            {records.length === 0 ? (
              <EmptyState title="暂无历史记录" description="完成一次文本或音频分析后，记录会自动出现在这里。" />
            ) : (
              <RecordTable records={records} onOpen={(id) => void loadRecordDetail(id)} onDelete={(id) => void deleteRecord(id)} />
            )}
          </Panel>

          <Panel icon={<MessageSquareText size={20} />} title="记录详情">
            {selectedRecord ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <RiskBadge level={selectedRecord.risk_level} />
                  <span className="text-sm text-[#667085]">{formatDate(selectedRecord.created_at)}</span>
                </div>
                <div className="grid gap-3 text-sm text-[#475467]">
                  <InfoRow label="编号" value={`#${selectedRecord.id}`} />
                  <InfoRow label="输入来源" value={inputTypeCopy(selectedRecord.input_type)} />
                  <InfoRow label="预测" value={selectedRecord.prediction === "fraud" ? "风险通话" : "正常通话"} />
                  <InfoRow label="风险分" value={`${Math.round(selectedRecord.risk_score * 100)}/100`} />
                  <InfoRow label="压力分" value={selectedRecord.pressure_score === null ? "--" : `${Math.round(selectedRecord.pressure_score * 100)}/100`} />
                </div>
                <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4">
                  <p className="mb-2 text-sm font-medium text-[#344054]">通话文本</p>
                  <p className="max-h-56 overflow-auto text-sm leading-7 text-[#475467]">{selectedRecord.transcript || "无文本"}</p>
                </div>
                {selectedRecord.analysis_result.timeline_summary ? (
                  <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4">
                    <p className="mb-3 text-sm font-medium text-[#344054]">风险时间线摘要</p>
                    <div className="grid gap-3 text-sm text-[#475467]">
                      <InfoRow label="片段数" value={`${selectedRecord.analysis_result.timeline_summary.segment_count}`} />
                      <InfoRow label="峰值风险" value={`${Math.round(selectedRecord.analysis_result.timeline_summary.peak_risk_score * 100)}/100`} />
                      <InfoRow label="峰值位置" value={`${formatTimelineTime(selectedRecord.analysis_result.timeline_summary.peak_start ?? 0)} - ${formatTimelineTime(selectedRecord.analysis_result.timeline_summary.peak_end ?? 0)}`} />
                    </div>
                  </div>
                ) : null}
                <div className="flex flex-wrap gap-2">
                  {selectedRecord.risk_factors.length > 0 ? selectedRecord.risk_factors.map((factor) => (
                    <span className="rounded-md bg-[#e6f4f1] px-2.5 py-1 text-sm font-medium text-[#0f766e]" key={`${factor.group}-${factor.keyword}`}>
                      {factor.keyword}
                    </span>
                  )) : <span className="text-sm text-[#667085]">暂无风险关键词。</span>}
                </div>
              </div>
            ) : (
              <EmptyState title="选择一条记录" description="点击左侧记录的“详情”，这里会展示完整分析摘要。" />
            )}
          </Panel>
        </div>
      </section>
    );
  }

  function renderRulesView() {
    return (
      <section className="mx-auto max-w-7xl px-6 py-8">
        <ViewHeader title="规则管理" description="管理文本风险关键词。默认规则可停用，自定义规则可删除。" />
        {panelError ? <ErrorBox message={panelError} /> : null}
        <div className="grid gap-6 lg:grid-cols-[390px_1fr]">
          <Panel icon={<Plus size={20} />} title="新增关键词规则">
            <div className="space-y-3">
              <label className="block text-sm font-medium text-[#344054]">
                规则组
                <input
                  className="mt-2 h-10 w-full rounded-md border border-[#cfd7e3] bg-[#fbfcfe] px-3 outline-none focus:border-[#0f766e]"
                  onChange={(event) => setRuleForm((value) => ({ ...value, group: event.target.value }))}
                  value={ruleForm.group}
                />
              </label>
              <label className="block text-sm font-medium text-[#344054]">
                关键词
                <input
                  className="mt-2 h-10 w-full rounded-md border border-[#cfd7e3] bg-[#fbfcfe] px-3 outline-none focus:border-[#0f766e]"
                  onChange={(event) => setRuleForm((value) => ({ ...value, keyword: event.target.value }))}
                  placeholder="例如：安全账户"
                  value={ruleForm.keyword}
                />
              </label>
              <label className="block text-sm font-medium text-[#344054]">
                权重
                <input
                  className="mt-2 h-10 w-full rounded-md border border-[#cfd7e3] bg-[#fbfcfe] px-3 outline-none focus:border-[#0f766e]"
                  onChange={(event) => setRuleForm((value) => ({ ...value, weight: event.target.value }))}
                  step="0.1"
                  type="number"
                  value={ruleForm.weight}
                />
              </label>
              <button
                className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-[#0f766e] px-4 py-3 text-sm font-semibold text-white transition hover:bg-[#0b615a]"
                onClick={createRule}
                type="button"
              >
                <Plus size={16} />
                新增规则
              </button>
              <button
                className="inline-flex w-full items-center justify-center gap-2 rounded-md border border-[#cfd7e3] bg-white px-4 py-3 text-sm font-semibold text-[#344054] transition hover:bg-[#f2f5f8]"
                onClick={resetDefaultRules}
                type="button"
              >
                <RotateCcw size={16} />
                恢复默认规则
              </button>
            </div>
          </Panel>

          <Panel icon={<SlidersHorizontal size={20} />} title="关键词规则">
            <div className="mb-4 flex items-center gap-2 rounded-md border border-[#cfd7e3] bg-[#fbfcfe] px-3">
              <Search className="text-[#667085]" size={16} />
              <input
                className="h-10 flex-1 bg-transparent text-sm outline-none placeholder:text-[#98a2b3]"
                onChange={(event) => setRuleFilter(event.target.value)}
                placeholder="搜索规则组、关键词或来源"
                value={ruleFilter}
              />
            </div>
            <div className="max-h-[620px] overflow-auto rounded-md border border-[#dde2ea]">
              <table className="w-full min-w-[720px] border-collapse text-sm">
                <thead className="sticky top-0 bg-[#f2f5f8] text-left text-[#667085]">
                  <tr>
                    <th className="px-3 py-2 font-medium">规则组</th>
                    <th className="px-3 py-2 font-medium">关键词</th>
                    <th className="px-3 py-2 font-medium">来源</th>
                    <th className="px-3 py-2 font-medium">状态</th>
                    <th className="px-3 py-2 text-right font-medium">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRules.map((rule) => (
                    <tr className="border-t border-[#dde2ea]" key={rule.id}>
                      <td className="px-3 py-2 text-[#475467]">{rule.group}</td>
                      <td className="px-3 py-2 font-medium">{rule.keyword}</td>
                      <td className="px-3 py-2 text-[#667085]">{rule.source === "default" ? "默认" : "自定义"}</td>
                      <td className="px-3 py-2">
                        <span className={`rounded-md px-2 py-1 text-xs font-medium ${rule.enabled ? "bg-emerald-50 text-emerald-700" : "bg-[#f2f5f8] text-[#667085]"}`}>
                          {rule.enabled ? "启用" : "停用"}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right">
                        <button
                          className="mr-2 rounded-md border border-[#cfd7e3] px-2.5 py-1.5 text-xs font-medium text-[#344054] hover:bg-[#f2f5f8]"
                          onClick={() => void toggleRule(rule.id)}
                          type="button"
                        >
                          {rule.enabled ? "停用" : "启用"}
                        </button>
                        <button
                          className="rounded-md border border-rose-200 px-2.5 py-1.5 text-xs font-medium text-rose-700 hover:bg-rose-50 disabled:cursor-not-allowed disabled:border-[#dde2ea] disabled:text-[#98a2b3]"
                          disabled={rule.source !== "custom"}
                          onClick={() => void deleteRule(rule.id)}
                          type="button"
                        >
                          删除
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
      </section>
    );
  }

  function renderDeployView() {
    const status = deploymentStatus;
    const featureList = status?.features ?? [
      {
        id: "offline",
        name: "部署状态",
        enabled: false,
        status: "未连接",
        detail: "请先启动后端服务，或检查 NEXT_PUBLIC_API_URL 是否指向正确的 API 地址。"
      }
    ];

    return (
      <section className="mx-auto max-w-7xl px-6 py-8">
        <ViewHeader
          title="部署与拓展"
          description="把本地机器学习系统整理成可展示、可部署、可解释的产品原型；云端演示版保留轻量能力，本地完整版保留完整音频模型链路。"
        />

        <section className="grid gap-4 md:grid-cols-4">
          <Metric icon={<Cloud size={18} />} label="部署模式" unit="" value={status?.demo_mode ? "Demo" : "Full"} />
          <Metric icon={<Server size={18} />} label="API 版本" unit="" value={status?.api_version ?? "--"} />
          <Metric icon={<ShieldCheck size={18} />} label="在线能力" unit="项" value={String(featureList.filter((item) => item.enabled).length)} />
          <Metric icon={<Database size={18} />} label="本地记录" unit="条" value={String(analytics?.total_calls ?? 0)} />
        </section>

        <section className="mt-6 grid gap-6 lg:grid-cols-[1fr_380px]">
          <Panel icon={<Server size={20} />} title="运行能力">
            <div className="grid gap-3 md:grid-cols-2">
              {featureList.map((feature) => (
                <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4" key={feature.id}>
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="font-semibold">{feature.name}</p>
                      <p className="mt-1 text-sm text-[#667085]">{feature.status}</p>
                    </div>
                    <span
                      className={`rounded-md px-2.5 py-1 text-xs font-medium ${
                        feature.enabled ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-800"
                      }`}
                    >
                      {feature.enabled ? "可用" : "受限"}
                    </span>
                  </div>
                  <p className="mt-3 text-sm leading-6 text-[#475467]">{feature.detail}</p>
                </div>
              ))}
            </div>
          </Panel>

          <Panel icon={<Cloud size={20} />} title="云端演示版">
            <div className="space-y-4 text-sm leading-6 text-[#475467]">
              <p>
                云端部署建议采用“前端 Vercel + 后端 Render/Railway”的轻量组合，公开演示时使用文本风险、规则管理、历史记录和看板。
              </p>
              <div className="rounded-md bg-[#f2f5f8] p-3">
                <p className="font-medium text-[#344054]">推荐环境变量</p>
                <p className="mt-2 font-mono text-xs">CALLGUARD_DEMO_MODE=1</p>
                <p className="font-mono text-xs">NEXT_PUBLIC_API_URL=https://你的后端地址</p>
              </div>
              <div className="space-y-2">
                {(status?.limitations ?? ["当前未读取到部署状态，请检查后端服务。"]).map((item) => (
                  <p className="rounded-md border border-[#dde2ea] bg-white p-3" key={item}>{item}</p>
                ))}
              </div>
            </div>
          </Panel>

          <Panel icon={<Fingerprint size={20} />} title="AI 伪造语音检测">
            <div className="grid gap-4 md:grid-cols-[260px_1fr]">
              <div className="rounded-md border border-amber-200 bg-amber-50 p-4 text-amber-900">
                <p className="font-semibold">未来拓展，不作为当前主线</p>
                <p className="mt-2 text-sm leading-6">
                  它可以增强产品完整度，但不要替代本项目的中文通话风险、压力感知和 CAEF 融合主题。
                </p>
              </div>
              <div className="grid gap-3 md:grid-cols-3">
                <RoadmapCard title="数据准备" description="准备真实/合成语音、TTS、VC 和 replay 样本，避免只用单一来源。" />
                <RoadmapCard title="声学检测" description="提取频谱、相位、伪影和说话人一致性特征，训练二分类模型。" />
                <RoadmapCard title="融合接入" description="将伪造概率作为独立安全信号接入 CAEF，而不是直接判定诈骗。" />
              </div>
            </div>
          </Panel>

          <Panel icon={<KeyRound size={20} />} title="登录注册取舍">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4">
                <p className="font-semibold">当前版本不强制登录</p>
                <p className="mt-2 text-sm leading-6 text-[#475467]">
                  更适合课堂展示：打开即可体验，避免账号系统抢走机器学习主线。
                </p>
              </div>
              <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4">
                <p className="font-semibold">未来接入方式</p>
                <p className="mt-2 text-sm leading-6 text-[#475467]">
                  若做多人使用，可以接入 Clerk/Auth.js，并在历史记录表增加 user_id 字段实现数据隔离。
                </p>
              </div>
            </div>
          </Panel>

          <Panel icon={<Languages size={20} />} title="多语言策略">
            <div className="space-y-3 text-sm leading-6 text-[#475467]">
              <p>
                当前不急于做多语言，因为项目优势在中文诈骗话术、中文 ASR 和中文数据集。多语言适合作为部署后的国际化包装，而不是课程报告主线。
              </p>
              <div className="grid gap-3 md:grid-cols-3">
                <RoadmapCard title="中文优先" description="保持 TeleAntiFraud 和中文规则解释为核心。" />
                <RoadmapCard title="英文界面" description="未来可只翻译 UI 文案，不改变模型主线。" />
                <RoadmapCard title="跨语种模型" description="需要另行补充英文通话欺诈语料和 ASR 评测。" />
              </div>
            </div>
          </Panel>

          <Panel icon={<ShieldCheck size={20} />} title="下一步">
            <div className="space-y-2">
              {(status?.next_steps ?? [
                "启动后端后刷新本页。",
                "部署时把前端环境变量 NEXT_PUBLIC_API_URL 指向公开后端地址。"
              ]).map((item) => (
                <div className="rounded-md bg-[#f2f5f8] p-3 text-sm leading-6 text-[#475467]" key={item}>
                  {item}
                </div>
              ))}
            </div>
          </Panel>
        </section>
      </section>
    );
  }
}

function Panel({ icon, title, children }: { icon?: ReactNode; title?: string; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-[#dde2ea] bg-white p-5 shadow-sm">
      {title ? (
        <div className="mb-4 flex items-center gap-3">
          {icon}
          <h2 className="text-lg font-semibold">{title}</h2>
        </div>
      ) : null}
      {children}
    </section>
  );
}

function ViewHeader({ title, description }: { title: string; description: string }) {
  return (
    <div className="mb-6">
      <h2 className="text-2xl font-semibold">{title}</h2>
      <p className="mt-2 text-sm leading-6 text-[#667085]">{description}</p>
    </div>
  );
}

function Metric({
  icon,
  label,
  value,
  unit
}: {
  icon: ReactNode;
  label: string;
  value: string;
  unit: string;
}) {
  return (
    <div className="rounded-lg border border-[#dde2ea] bg-white p-5 shadow-sm">
      <div className="flex items-center gap-2 text-sm text-[#667085]">
        {icon}
        <span>{label}</span>
      </div>
      <div className="mt-3 flex items-end gap-1">
        <span className="text-4xl font-semibold">{value}</span>
        <span className="pb-1 text-sm text-[#667085]">{unit}</span>
      </div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <span className="text-[#667085]">{label}</span>
      <span className="max-w-44 break-words text-right font-medium text-[#171b22]">{value}</span>
    </div>
  );
}

function RiskBadge({ level, fallback }: { level?: RiskLevel | null; fallback?: string }) {
  return (
    <span
      className={`rounded-md border px-3 py-1 text-sm font-medium ${
        level ? levelTone[level] : "border-[#dde2ea] bg-[#f2f5f8] text-[#667085]"
      }`}
    >
      {level ? levelCopy[level] : fallback ?? "--"}
    </span>
  );
}

function ProgressBar({ value, level, compact = false }: { value: number; level: RiskLevel; compact?: boolean }) {
  return (
    <div>
      {!compact ? (
        <div className="mb-3 flex items-center justify-between text-sm">
          <span className="text-[#667085]">Risk score</span>
          <span className="font-semibold">{value}%</span>
        </div>
      ) : null}
      <div className={`${compact ? "h-3" : "h-4"} overflow-hidden rounded-full bg-[#e7ecf2]`}>
        <div
          className={`h-full rounded-full transition-all ${
            level === "high"
              ? "bg-[#e11d48]"
              : level === "medium"
                ? "bg-[#d97706]"
                : level === "low"
                  ? "bg-[#0284c7]"
                  : "bg-[#0f766e]"
          }`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

function TimelinePanel({ result }: { result: CallAnalysisResult | null }) {
  const timeline = result?.timeline ?? [];
  const summary = result?.timeline_summary ?? null;
  const peakPercent = Math.round((summary?.peak_risk_score ?? 0) * 100);

  return (
    <Panel icon={<Activity size={20} />} title="风险时间线">
      {timeline.length === 0 ? (
        <div className="rounded-md border border-dashed border-[#cfd7e3] bg-[#fbfcfe] p-5 text-sm leading-6 text-[#667085]">
          上传或录制音频后，系统会按 5 秒窗口生成分段风险时间线，定位通话中风险升高的片段。
        </div>
      ) : (
        <div className="space-y-5">
          <div className="grid gap-3 md:grid-cols-4">
            <TimelineStat label="音频时长" value={formatTimelineTime(summary?.duration_seconds ?? 0)} />
            <TimelineStat label="片段数" value={`${summary?.segment_count ?? timeline.length}`} />
            <TimelineStat label="中高风险片段" value={`${summary?.medium_or_high_segments ?? 0}`} />
            <TimelineStat label="峰值风险" value={`${peakPercent}/100`} />
          </div>

          <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4">
            <div className="mb-3 flex items-center justify-between text-sm">
              <span className="text-[#667085]">Timeline overview</span>
              <span className="font-medium">
                峰值 {summary?.peak_start !== null && summary?.peak_start !== undefined
                  ? `${formatTimelineTime(summary.peak_start)} - ${formatTimelineTime(summary.peak_end ?? summary.peak_start)}`
                  : "--"}
              </span>
            </div>
            <div className="flex h-14 items-end gap-1">
              {timeline.map((segment) => (
                <div
                  className={`min-w-2 flex-1 rounded-t ${
                    segment.risk_level === "high"
                      ? "bg-[#e11d48]"
                      : segment.risk_level === "medium"
                        ? "bg-[#d97706]"
                        : segment.risk_level === "low"
                          ? "bg-[#0284c7]"
                          : "bg-[#0f766e]"
                  }`}
                  key={segment.index}
                  style={{ height: `${Math.max(8, Math.round(segment.risk_score * 100))}%` }}
                  title={`${formatTimelineTime(segment.start)}-${formatTimelineTime(segment.end)} ${Math.round(segment.risk_score * 100)}/100`}
                />
              ))}
            </div>
          </div>

          <div className="max-h-[420px] space-y-3 overflow-auto pr-1">
            {timeline.map((segment) => (
              <div className="rounded-md border border-[#dde2ea] bg-white p-4" key={segment.index}>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="font-semibold">
                      {formatTimelineTime(segment.start)} - {formatTimelineTime(segment.end)}
                    </p>
                    <p className="mt-1 text-sm text-[#667085]">
                      音频 {scoreCopy(segment.audio_score)} · 文本 {scoreCopy(segment.text_score)} · 压力 {scoreCopy(segment.pressure_score)}
                    </p>
                  </div>
                  <RiskBadge level={segment.risk_level} />
                </div>

                <ProgressBar
                  compact
                  level={segment.risk_level}
                  value={Math.round(segment.risk_score * 100)}
                />

                {segment.transcript ? (
                  <p className="mt-3 rounded-md bg-[#f2f5f8] p-3 text-sm leading-6 text-[#475467]">
                    {segment.transcript}
                  </p>
                ) : (
                  <p className="mt-3 text-sm text-[#98a2b3]">该片段暂无可用转写文本，仅使用音频信号评估。</p>
                )}

                {segment.factors.length > 0 ? (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {segment.factors.map((factor) => (
                      <span className="rounded-md bg-[#e6f4f1] px-2.5 py-1 text-xs font-medium text-[#0f766e]" key={`${segment.index}-${factor.group}-${factor.keyword}`}>
                        {factor.keyword}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </div>
      )}
    </Panel>
  );
}

function TimelineStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-3">
      <p className="text-xs text-[#667085]">{label}</p>
      <p className="mt-1 text-lg font-semibold">{value}</p>
    </div>
  );
}

function RoadmapCard({ title, description }: { title: string; description: string }) {
  return (
    <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4">
      <p className="font-semibold">{title}</p>
      <p className="mt-2 text-sm leading-6 text-[#475467]">{description}</p>
    </div>
  );
}

function scoreCopy(value: number | null) {
  return value === null ? "--" : `${Math.round(value * 100)}/100`;
}

function RecordTable({
  records,
  onOpen,
  onDelete
}: {
  records: CallRecordSummary[];
  onOpen: (id: number) => void;
  onDelete: (id: number) => void;
}) {
  return (
    <div className="overflow-auto rounded-md border border-[#dde2ea]">
      <table className="w-full min-w-[760px] border-collapse text-sm">
        <thead className="bg-[#f2f5f8] text-left text-[#667085]">
          <tr>
            <th className="px-3 py-2 font-medium">时间</th>
            <th className="px-3 py-2 font-medium">风险</th>
            <th className="px-3 py-2 font-medium">来源</th>
            <th className="px-3 py-2 font-medium">文本摘要</th>
            <th className="px-3 py-2 text-right font-medium">操作</th>
          </tr>
        </thead>
        <tbody>
          {records.map((record) => (
            <tr className="border-t border-[#dde2ea]" key={record.id}>
              <td className="whitespace-nowrap px-3 py-2 text-[#667085]">{formatDate(record.created_at)}</td>
              <td className="px-3 py-2">
                <RiskBadge level={record.risk_level} />
              </td>
              <td className="px-3 py-2 text-[#475467]">{inputTypeCopy(record.input_type)}</td>
              <td className="max-w-[300px] truncate px-3 py-2 text-[#344054]">
                {record.transcript_preview || record.file_name || "无文本"}
              </td>
              <td className="whitespace-nowrap px-3 py-2 text-right">
                <button
                  className="mr-2 rounded-md border border-[#cfd7e3] px-2.5 py-1.5 text-xs font-medium text-[#344054] hover:bg-[#f2f5f8]"
                  onClick={() => onOpen(record.id)}
                  type="button"
                >
                  详情
                </button>
                <button
                  className="rounded-md border border-rose-200 px-2.5 py-1.5 text-xs font-medium text-rose-700 hover:bg-rose-50"
                  onClick={() => onDelete(record.id)}
                  type="button"
                >
                  <Trash2 size={13} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EmptyState({ title, description }: { title: string; description: string }) {
  return (
    <div className="rounded-lg border border-dashed border-[#cfd7e3] bg-white p-8 text-center">
      <p className="font-semibold">{title}</p>
      <p className="mt-2 text-sm text-[#667085]">{description}</p>
    </div>
  );
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="mb-5 rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
      {message}
    </div>
  );
}

function inputTypeCopy(value: string) {
  if (value === "audio_text") {
    return "音频+文本";
  }
  if (value === "audio") {
    return "音频";
  }
  return "文本";
}

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function formatTimelineTime(seconds: number) {
  const safeSeconds = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(safeSeconds / 60)
    .toString()
    .padStart(2, "0");
  const rest = (safeSeconds % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

function formatDuration(seconds: number) {
  const minutes = Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0");
  const rest = (seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

function getAudioContextConstructor() {
  const browserWindow = window as Window &
    typeof globalThis & {
      webkitAudioContext?: typeof AudioContext;
    };
  return browserWindow.AudioContext ?? browserWindow.webkitAudioContext;
}

function mergeAudioChunks(chunks: Float32Array[]) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const result = new Float32Array(totalLength);
  let offset = 0;
  chunks.forEach((chunk) => {
    result.set(chunk, offset);
    offset += chunk.length;
  });
  return result;
}

function encodeWav(samples: Float32Array, sampleRate: number) {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);
  floatTo16BitPcm(view, 44, samples);

  return new Blob([view], { type: "audio/wav" });
}

function writeAscii(view: DataView, offset: number, text: string) {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

function floatTo16BitPcm(view: DataView, offset: number, samples: Float32Array) {
  for (let index = 0; index < samples.length; index += 1, offset += 2) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
}
