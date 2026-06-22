export interface EmotionScore {
  label: string;
  score: number;
}

export interface PipelineResult {
  version: string;
  label: string;
  description: string;
  prompt: string;
  negative_prompt: string;
  keywords: string[];
  emotions: EmotionScore[];
  primary_emotion: string;
  primary_genre: string;
  guidance_scale: number;
  num_inference_steps: number;
  image_base64: string;
  image_backend: string;
}

export interface GenerateResponse {
  results: PipelineResult[];
  seed: number;
  nlp_backend: string;
}
