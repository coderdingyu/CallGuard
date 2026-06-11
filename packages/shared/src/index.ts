export type RiskLevel = "normal" | "low" | "medium" | "high";

export interface RiskFactor {
  group: string;
  keyword: string;
}

export interface AnalysisSummary {
  riskScore: number;
  riskLevel: RiskLevel;
  factors: RiskFactor[];
  suggestion: string;
}

export interface AudioAnalysisSummary {
  fileName: string;
  prediction: "normal" | "fraud";
  riskScore: number;
  riskLevel: RiskLevel;
  probabilities: {
    normal: number;
    fraud: number;
  };
  model: string;
  featureWindowSeconds: number;
}

export interface TranscriptionSummary {
  text: string;
  source: "manual" | "asr" | "none";
  language: string;
  languageProbability: number;
  durationSeconds: number;
  model: string;
}

export interface CallAnalysisSummary {
  prediction: "normal" | "fraud";
  riskScore: number;
  riskLevel: RiskLevel;
  transcript: string;
  transcriptSource: "manual" | "asr" | "none";
  asr: TranscriptionSummary | null;
  audio: AudioAnalysisSummary | null;
  text: AnalysisSummary | null;
  fusionWeights: {
    audio: number;
    text: number;
  };
  suggestion: string;
}
