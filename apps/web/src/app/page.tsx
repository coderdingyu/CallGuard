"use client";

import {
  Activity,
  AlertTriangle,
  AudioWaveform,
  FileAudio,
  Gauge,
  HeartPulse,
  LoaderCircle,
  MessageSquareText,
  Mic,
  PlayCircle,
  RotateCcw,
  ShieldCheck,
  Square,
  Upload
} from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8001";

const copy = {
  audio: "\u97f3\u9891",
  transcript: "\u901a\u8bdd\u6587\u672c",
  autoTranscript: "\u81ea\u52a8\u8f6c\u5199",
  noFile: "\u672a\u9009\u62e9\u6587\u4ef6",
  analyze: "\u5206\u6790\u901a\u8bdd",
  reset: "\u91cd\u7f6e",
  recordStart: "\u5f00\u59cb\u5f55\u97f3",
  recordStop: "\u505c\u6b62\u5f55\u97f3",
  recording: "\u6b63\u5728\u5f55\u97f3",
  demoSamples: "\u6f14\u793a\u6837\u4f8b",
  loadingSample: "\u8f7d\u5165\u6837\u4f8b",
  textOnly: "\u7eaf\u6587\u672c",
  signal: "\u4fe1\u53f7\u6982\u89c8",
  fusionRisk: "\u878d\u5408\u98ce\u9669",
  audioRisk: "\u97f3\u9891\u98ce\u9669",
  textRisk: "\u6587\u672c\u98ce\u9669",
  stressRisk: "\u538b\u529b\u4fe1\u53f7",
  overall: "\u7efc\u5408\u5224\u65ad",
  waiting: "\u7b49\u5f85\u5206\u6790",
  suggestion: "\u5efa\u8bae",
  modelEvidence: "\u6a21\u578b\u8bc1\u636e",
  ruleSignal: "\u89c4\u5219\u4fe1\u53f7",
  chooseToStart: "\u4e0a\u4f20\u97f3\u9891\u6216\u8f93\u5165\u901a\u8bdd\u6587\u672c\u540e\u5f00\u59cb\u5206\u6790\u3002",
  riskCall: "\u98ce\u9669\u901a\u8bdd",
  normalCall: "\u6b63\u5e38\u901a\u8bdd",
  factors: "\u98ce\u9669\u56e0\u7d20",
  noFactors: "\u6682\u65e0\u6587\u672c\u98ce\u9669\u56e0\u7d20\u3002",
  modelState: "\u6a21\u578b\u72b6\u6001",
  emotionAssist: "\u8bed\u97f3\u538b\u529b/\u60c5\u7eea\u8f85\u52a9",
  primaryEmotion: "\u4e3b\u8981\u60c5\u7eea",
  emotionConfidence: "\u60c5\u7eea\u7f6e\u4fe1\u5ea6",
  pressureDrivers: "\u538b\u529b\u6765\u6e90",
  pressureModel: "\u538b\u529b\u6a21\u578b",
  pressureAuxiliary: "\u8f85\u52a9\u6307\u6807",
  adaptiveFusion: "\u81ea\u9002\u5e94\u878d\u5408",
  modalityAgreement: "\u6a21\u6001\u4e00\u81f4\u5ea6",
  provideInput: "\u8bf7\u5148\u63d0\u4f9b\u97f3\u9891\u6216\u901a\u8bdd\u6587\u672c\u3002",
  failed: "\u5206\u6790\u5931\u8d25\u3002",
  demoFailed: "\u6837\u4f8b\u52a0\u8f7d\u5931\u8d25\u3002",
  micUnsupported: "\u5f53\u524d\u6d4f\u89c8\u5668\u4e0d\u652f\u6301\u5f55\u97f3\u3002",
  micDenied: "\u65e0\u6cd5\u83b7\u53d6\u9ea6\u514b\u98ce\u6743\u9650\u3002",
  emptyRecording: "\u5f55\u97f3\u65f6\u95f4\u592a\u77ed\uff0c\u8bf7\u91cd\u65b0\u5f55\u5236\u3002",
  placeholder:
    "\u53ef\u9009\uff1a\u8f93\u5165\u6216\u4fee\u6b63\u901a\u8bdd\u6587\u672c\u3002\u7559\u7a7a\u65f6\u7cfb\u7edf\u4f1a\u5c1d\u8bd5\u81ea\u52a8\u8f6c\u5199\u97f3\u9891\u3002"
};

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

interface CallAnalysisResult {
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

const levelCopy: Record<RiskLevel, string> = {
  normal: "\u6b63\u5e38",
  low: "\u4f4e\u98ce\u9669",
  medium: "\u4e2d\u98ce\u9669",
  high: "\u9ad8\u98ce\u9669"
};

const levelTone: Record<RiskLevel, string> = {
  normal: "border-emerald-200 bg-emerald-50 text-emerald-700",
  low: "border-sky-200 bg-sky-50 text-sky-700",
  medium: "border-amber-200 bg-amber-50 text-amber-800",
  high: "border-rose-200 bg-rose-50 text-rose-700"
};

const demoLabelCopy: Record<string, string> = {
  normal: "\u6b63\u5e38",
  fraud: "\u8bc8\u9a97",
  high_pressure: "\u9ad8\u538b",
  normal_pressure: "\u5e73\u7a33"
};

const demoLabelTone: Record<string, string> = {
  normal: "border-emerald-200 bg-emerald-50 text-emerald-700",
  fraud: "border-rose-200 bg-rose-50 text-rose-700",
  high_pressure: "border-amber-200 bg-amber-50 text-amber-800",
  normal_pressure: "border-sky-200 bg-sky-50 text-sky-700"
};

export default function Home() {
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
  const emotionConfidencePercent = Math.round((result?.emotion?.emotion_confidence ?? 0) * 100);
  const textModelPercent =
    result?.text?.model_score !== null && result?.text?.model_score !== undefined
      ? Math.round(result.text.model_score * 100)
      : null;
  const rulePercent = result?.text ? Math.round(result.text.rule_score * 100) : null;
  const selectedName = file?.name ?? copy.noFile;
  const factors = result?.text?.factors ?? [];
  const evidenceTerms = result?.text?.evidence_terms ?? [];
  const emotionDrivers = result?.emotion?.pressure_drivers ?? [];
  const pressureProbabilities = result?.emotion?.pressure_probabilities ?? {};

  const signalBars = useMemo(() => {
    const base = result ? riskPercent : 30;
    return Array.from({ length: 44 }, (_, index) => {
      const wave = Math.abs(Math.sin(index * 0.6)) * 36;
      const pulse = ((index * 19 + base) % 31) + 14;
      return Math.min(84, Math.round(wave + pulse));
    });
  }, [result, riskPercent]);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/demo/samples`)
      .then((response) => (response.ok ? response.json() : []))
      .then((payload) => setDemoSamples(Array.isArray(payload) ? payload : []))
      .catch(() => setDemoSamples([]));
  }, []);

  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
      closeRecordingGraph();
    };
  }, [audioUrl]);

  function selectFile(nextFile: File | null) {
    if (!nextFile) {
      return;
    }
    setAudioFile(nextFile);
  }

  function setAudioFile(nextFile: File) {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setFile(nextFile);
    setAudioUrl(URL.createObjectURL(nextFile));
    setResult(null);
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
    setRecordingSeconds(0);
  }

  async function startRecording() {
    const AudioContextConstructor = getAudioContextConstructor();
    if (!navigator.mediaDevices?.getUserMedia || !AudioContextConstructor) {
      setError(copy.micUnsupported);
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
      setError(null);
      recordingTimerRef.current = window.setInterval(
        () => setRecordingSeconds((value) => value + 1),
        1000
      );
    } catch {
      closeRecordingGraph();
      setIsRecording(false);
      setError(copy.micDenied);
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
      setError(copy.emptyRecording);
      return;
    }

    const samples = mergeAudioChunks(chunks);
    const wavBlob = encodeWav(samples, sampleRate);
    const nextFile = new File([wavBlob], `callguard-recording-${Date.now()}.wav`, {
      type: "audio/wav"
    });
    setAudioFile(nextFile);
    setError(null);
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
    setError(null);
    setTranscript(sample.transcript);

    try {
      if (sample.has_audio) {
        const response = await fetch(`${API_BASE_URL}/api/demo/samples/${sample.id}/audio`);
        if (!response.ok) {
          throw new Error(copy.demoFailed);
        }
        const blob = await response.blob();
        const nextFile = new File(
          [blob],
          sample.audio_file_name ?? `${sample.id}.wav`,
          { type: blob.type || "audio/wav" }
        );
        setAudioFile(nextFile);
      } else {
        clearAudioFile();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : copy.demoFailed);
    } finally {
      setLoadingDemoId(null);
    }
  }

  async function analyzeCall() {
    if (!file && !transcript.trim()) {
      setError(copy.provideInput);
      return;
    }

    setIsLoading(true);
    setError(null);

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
        throw new Error(payload.detail ?? copy.failed);
      }
      setResult(payload as CallAnalysisResult);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : copy.failed);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#f7f8fa] text-[#171b22]">
      <header className="border-b border-[#dde2ea] bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-md bg-[#0f766e] text-white shadow-sm">
              <ShieldCheck size={22} />
            </div>
            <div>
              <p className="text-sm text-[#667085]">Call risk awareness</p>
              <h1 className="text-xl font-semibold tracking-normal">CallGuard</h1>
            </div>
          </div>
          <button
            className="inline-flex items-center gap-2 rounded-md border border-[#cfd7e3] bg-white px-4 py-2 text-sm font-medium text-[#344054] transition hover:bg-[#f2f5f8]"
            onClick={reset}
            type="button"
          >
            <RotateCcw size={16} />
            {copy.reset}
          </button>
        </div>
      </header>

      <section className="mx-auto grid max-w-7xl gap-6 px-6 py-8 lg:grid-cols-[410px_1fr]">
        <aside className="space-y-5">
          <section className="rounded-lg border border-[#dde2ea] bg-white p-5 shadow-sm">
            <div className="mb-5 flex items-center gap-3">
              <FileAudio size={20} />
              <h2 className="text-lg font-semibold">{copy.audio}</h2>
            </div>

            <input
              id="callguard-audio-upload"
              ref={inputRef}
              accept="audio/*"
              className="sr-only"
              onChange={(event) => selectFile(event.target.files?.[0] ?? null)}
              type="file"
            />

            <label
              className="grid min-h-44 w-full cursor-pointer place-items-center rounded-md border border-dashed border-[#98a2b3] bg-[#fbfcfe] p-6 text-center transition hover:border-[#0f766e] hover:bg-[#f3fbf9] focus-within:border-[#0f766e] focus-within:bg-white"
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
                className="inline-flex items-center justify-center gap-2 rounded-md border border-[#cfd7e3] bg-white px-3 py-2.5 text-sm font-medium text-[#344054] transition hover:bg-[#f2f5f8] disabled:cursor-not-allowed disabled:bg-[#f2f5f8] disabled:text-[#98a2b3]"
                disabled={isRecording || isLoading}
                onClick={startRecording}
                type="button"
              >
                <Mic size={16} />
                {copy.recordStart}
              </button>
              <button
                className="inline-flex items-center justify-center gap-2 rounded-md border border-[#fecdd3] bg-[#fff1f2] px-3 py-2.5 text-sm font-medium text-[#be123c] transition hover:bg-[#ffe4e6] disabled:cursor-not-allowed disabled:border-[#dde2ea] disabled:bg-[#f2f5f8] disabled:text-[#98a2b3]"
                disabled={!isRecording}
                onClick={stopRecording}
                type="button"
              >
                <Square size={15} />
                {copy.recordStop}
              </button>
            </div>

            {isRecording ? (
              <div className="mt-3 flex items-center justify-between rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
                <span>{copy.recording}</span>
                <span className="font-semibold">{formatDuration(recordingSeconds)}</span>
              </div>
            ) : null}
          </section>

          <section className="rounded-lg border border-[#dde2ea] bg-white p-5 shadow-sm">
            <div className="mb-4 flex items-center gap-3">
              <PlayCircle size={20} />
              <h2 className="text-lg font-semibold">{copy.demoSamples}</h2>
            </div>
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
                    <span>{sample.has_audio ? sample.audio_file_name : copy.textOnly}</span>
                    {loadingDemoId === sample.id ? <span>{copy.loadingSample}</span> : null}
                  </div>
                </button>
              ))}
            </div>
          </section>

          <section className="rounded-lg border border-[#dde2ea] bg-white p-5 shadow-sm">
            <div className="mb-4 flex items-center gap-3">
              <MessageSquareText size={20} />
              <h2 className="text-lg font-semibold">{copy.transcript}</h2>
            </div>
            <textarea
              className="min-h-36 w-full resize-y rounded-md border border-[#cfd7e3] bg-[#fbfcfe] p-3 text-sm leading-6 outline-none transition placeholder:text-[#98a2b3] focus:border-[#0f766e] focus:bg-white"
              onChange={(event) => {
                setTranscript(event.target.value);
                setResult(null);
                setError(null);
              }}
              placeholder={copy.placeholder}
              value={transcript}
            />

            <button
              className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-md bg-[#0f766e] px-4 py-3 text-sm font-semibold text-white transition hover:bg-[#0b615a] disabled:cursor-not-allowed disabled:bg-[#98a2b3]"
              disabled={(!file && !transcript.trim()) || isLoading}
              onClick={analyzeCall}
              type="button"
            >
              {isLoading ? <LoaderCircle className="animate-spin" size={17} /> : <Upload size={17} />}
              {copy.analyze}
            </button>

            {error ? (
              <div className="mt-4 rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
                {error}
              </div>
            ) : null}
          </section>

          <section className="rounded-lg border border-[#dde2ea] bg-white p-5 shadow-sm">
            <div className="mb-4 flex items-center gap-3">
              <Activity size={20} />
              <h2 className="text-lg font-semibold">{copy.signal}</h2>
            </div>
            <div className="flex h-24 items-end gap-1 rounded-md bg-[#f2f5f8] px-3 py-4">
              {signalBars.map((height, index) => (
                <div
                  className="flex-1 rounded-t bg-[#0f766e]"
                  key={`${height}-${index}`}
                  style={{ height: `${height}%`, opacity: 0.28 + index / 88 }}
                />
              ))}
            </div>
          </section>
        </aside>

        <div className="space-y-6">
          <section className="grid gap-4 md:grid-cols-4">
            <Metric icon={<Gauge size={18} />} label={copy.fusionRisk} unit="/100" value={result ? String(riskPercent) : "--"} />
            <Metric icon={<AudioWaveform size={18} />} label={copy.audioRisk} unit="/100" value={result?.audio ? String(audioPercent) : "--"} />
            <Metric icon={<MessageSquareText size={18} />} label={copy.textRisk} unit="/100" value={result?.text ? String(textPercent) : "--"} />
            <Metric icon={<HeartPulse size={18} />} label={copy.stressRisk} unit="/100" value={result?.emotion ? String(stressPercent) : "--"} />
          </section>

          <section className="rounded-lg border border-[#dde2ea] bg-white p-5 shadow-sm">
            <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-sm text-[#667085]">Fusion risk</p>
                <h2 className="text-lg font-semibold">{copy.overall}</h2>
              </div>
              <span
                className={`rounded-md border px-3 py-1 text-sm font-medium ${
                  result ? levelTone[result.risk_level] : "border-[#dde2ea] bg-[#f2f5f8] text-[#667085]"
                }`}
              >
                {result ? levelCopy[result.risk_level] : copy.waiting}
              </span>
            </div>

            <div className="grid gap-6 lg:grid-cols-[1fr_290px]">
              <div>
                <div className="mb-3 flex items-center justify-between text-sm">
                  <span className="text-[#667085]">Risk score</span>
                  <span className="font-semibold">{result ? `${riskPercent}%` : "--"}</span>
                </div>
                <div className="h-4 overflow-hidden rounded-full bg-[#e7ecf2]">
                  <div
                    className={`h-full rounded-full transition-all ${
                      result?.risk_level === "high"
                        ? "bg-[#e11d48]"
                        : result?.risk_level === "medium"
                          ? "bg-[#d97706]"
                          : result?.risk_level === "low"
                            ? "bg-[#0284c7]"
                            : "bg-[#0f766e]"
                    }`}
                    style={{ width: `${result ? riskPercent : 0}%` }}
                  />
                </div>

                <div className="mt-5 rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4">
                  <p className="text-sm text-[#667085]">{copy.suggestion}</p>
                  <p className="mt-2 font-medium leading-7">
                    {result?.suggestion ?? copy.chooseToStart}
                  </p>
                </div>
              </div>

              <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4">
                <p className="text-sm text-[#667085]">Prediction</p>
                <p className="mt-2 text-2xl font-semibold">
                  {result ? (result.prediction === "fraud" ? copy.riskCall : copy.normalCall) : "--"}
                </p>
                <div className="mt-5 space-y-3 text-sm text-[#475467]">
                  <InfoRow label="Transcript" value={result?.transcript_source ?? "--"} />
                <InfoRow label="Audio weight" value={result ? `${Math.round(result.fusion_weights.audio * 100)}%` : "--"} />
                <InfoRow label="Text weight" value={result ? `${Math.round(result.fusion_weights.text * 100)}%` : "--"} />
                <InfoRow label={copy.pressureAuxiliary} value={result?.emotion ? `${stressPercent}/100` : "--"} />
                <InfoRow label="Text model" value={result?.text?.model ?? "--"} />
                <InfoRow label={copy.adaptiveFusion} value={result?.fusion_method ?? "--"} />
                </div>
              </div>
            </div>
          </section>

          <section className="rounded-lg border border-[#dde2ea] bg-white p-5 shadow-sm">
            <div className="mb-4 flex items-center gap-3">
              <MessageSquareText size={20} />
              <h2 className="text-lg font-semibold">{copy.autoTranscript}</h2>
            </div>
            <div className="rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4">
              <p className="min-h-16 text-sm leading-7 text-[#344054]">
                {result?.transcript || copy.chooseToStart}
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
          </section>

          <section className="rounded-lg border border-[#dde2ea] bg-white p-5 shadow-sm">
            <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <HeartPulse size={20} />
                <h2 className="text-lg font-semibold">{copy.emotionAssist}</h2>
              </div>
              <span
                className={`rounded-md border px-3 py-1 text-sm font-medium ${
                  result?.emotion ? levelTone[result.emotion.pressure_level] : "border-[#dde2ea] bg-[#f2f5f8] text-[#667085]"
                }`}
              >
                {result?.emotion ? levelCopy[result.emotion.pressure_level] : copy.waiting}
              </span>
            </div>

            <div className="grid gap-6 lg:grid-cols-[230px_1fr]">
              <div>
                <p className="text-sm text-[#667085]">Pressure score</p>
                <div className="mt-3 flex items-end gap-1">
                  <span className="text-5xl font-semibold">{result?.emotion ? stressPercent : "--"}</span>
                  <span className="pb-1 text-sm text-[#667085]">/100</span>
                </div>
                <div className="mt-4 h-3 overflow-hidden rounded-full bg-[#e7ecf2]">
                  <div
                    className={`h-full rounded-full transition-all ${
                      result?.emotion?.pressure_level === "high"
                        ? "bg-[#e11d48]"
                        : result?.emotion?.pressure_level === "medium"
                          ? "bg-[#d97706]"
                          : result?.emotion?.pressure_level === "low"
                            ? "bg-[#0284c7]"
                            : "bg-[#0f766e]"
                    }`}
                    style={{ width: `${result?.emotion ? stressPercent : 0}%` }}
                  />
                </div>
              </div>

              <div className="space-y-5">
                <div className="grid gap-3 text-sm text-[#475467] md:grid-cols-3">
                  <InfoRow label={copy.primaryEmotion} value={result?.emotion?.primary_emotion ?? "--"} />
                  <InfoRow label={copy.emotionConfidence} value={result?.emotion ? `${emotionConfidencePercent}%` : "--"} />
                  <InfoRow label={copy.pressureModel} value={result?.emotion?.model ?? "--"} />
                </div>

                <div>
                  <p className="mb-3 text-sm font-medium text-[#344054]">{copy.pressureDrivers}</p>
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
                        {copy.chooseToStart}
                      </div>
                    )}
                  </div>
                </div>

                <div className="grid gap-3 md:grid-cols-3">
                  {Object.entries(pressureProbabilities).map(([label, value]) => (
                    <div className="rounded-md bg-[#f2f5f8] p-3 text-sm text-[#475467]" key={label}>
                      <p>{levelCopy[label as RiskLevel] ?? label}</p>
                      <p className="mt-1 font-semibold text-[#171b22]">{Math.round(value * 100)}%</p>
                    </div>
                  ))}
                </div>

                {result?.emotion ? (
                  <p className="rounded-md bg-[#f2f5f8] p-3 text-sm leading-6 text-[#475467]">
                    {result.emotion.suggestion}
                  </p>
                ) : null}
              </div>
            </div>
          </section>

          <section className="grid gap-6 lg:grid-cols-[1fr_340px]">
            <section className="rounded-lg border border-[#dde2ea] bg-white p-5 shadow-sm">
              <div className="mb-4 flex items-center gap-3">
                <AlertTriangle size={20} />
                <h2 className="text-lg font-semibold">{copy.factors}</h2>
              </div>
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
                    {copy.noFactors}
                  </div>
                )}
              </div>
              {evidenceTerms.length > 0 ? (
                <div className="mt-5 rounded-md border border-[#dde2ea] bg-[#fbfcfe] p-4">
                  <p className="text-sm font-medium text-[#344054]">{copy.modelEvidence}</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {evidenceTerms.map((term) => (
                      <span className="rounded-md bg-[#e6f4f1] px-2.5 py-1 text-sm font-medium text-[#0f766e]" key={term}>
                        {term}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
            </section>

            <section className="rounded-lg border border-[#dde2ea] bg-white p-5 shadow-sm">
              <div className="mb-4 flex items-center gap-3">
                <ShieldCheck size={20} />
                <h2 className="text-lg font-semibold">{copy.modelState}</h2>
              </div>
              <div className="space-y-3 text-sm text-[#475467]">
                <InfoRow label="Audio" value={result?.audio ? `${audioPercent}/100` : "--"} />
                <InfoRow label="Pressure" value={result?.emotion ? `${stressPercent}/100` : "--"} />
                <InfoRow label="Emotion" value={result?.emotion?.primary_emotion ?? "--"} />
                <InfoRow label="Text" value={result?.text ? `${textPercent}/100` : "--"} />
                <InfoRow label="Text ML" value={textModelPercent !== null ? `${textModelPercent}/100` : "--"} />
                <InfoRow label={copy.ruleSignal} value={rulePercent !== null ? `${rulePercent}/100` : "--"} />
                <InfoRow
                  label="Audio confidence"
                  value={
                    result?.fusion_diagnostics
                      ? `${Math.round(result.fusion_diagnostics.audio_confidence * 100)}%`
                      : "--"
                  }
                />
                <InfoRow
                  label="Text confidence"
                  value={
                    result?.fusion_diagnostics
                      ? `${Math.round(result.fusion_diagnostics.text_confidence * 100)}%`
                      : "--"
                  }
                />
                <InfoRow
                  label={copy.modalityAgreement}
                  value={
                    result?.fusion_diagnostics
                      ? `${Math.round(result.fusion_diagnostics.agreement * 100)}%`
                      : "--"
                  }
                />
                <InfoRow label="Label" value={result ? levelCopy[result.risk_level] : "--"} />
              </div>
              <div className="mt-5 space-y-3">
                {(result?.notes ?? ["Audio baseline ready.", "ASR and text ML ready."]).map((item) => (
                  <div className="rounded-md bg-[#f2f5f8] p-3 text-sm leading-6 text-[#475467]" key={item}>
                    {item}
                  </div>
                ))}
              </div>
            </section>
          </section>
        </div>
      </section>
    </main>
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
      <span className="max-w-40 break-words text-right font-medium text-[#171b22]">{value}</span>
    </div>
  );
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
