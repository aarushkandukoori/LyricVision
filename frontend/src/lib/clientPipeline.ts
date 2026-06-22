/** Client-side ArtML pipeline for GitHub Pages (no backend required). */

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

const STOP = new Set(
  "a an the and or but in on at to for of with by from is are was were be been being have has had do does did will would could should may might must i me my we our you your he him his she her it its they them their this that these those am so if as up out about into over after before between through during each few more most other some such no nor not only own same than too very can just now all when where how what which who why while again then once here there both any off until because".split(
    " ",
  ),
);

const EMOTION_WORDS: Record<string, Set<string>> = {
  sadness: new Set(["sad", "cry", "tears", "lost", "dark", "pain", "alone", "memories", "shadow", "fog", "broken", "empty", "gone", "wound", "drowning", "deep"]),
  joy: new Set(["joy", "happy", "shine", "bright", "dance", "alive", "thrive", "golden", "free", "love", "party", "confetti", "sun", "neon", "rhythm"]),
  anger: new Set(["anger", "fight", "fire", "storm", "rebel", "blood", "fierce", "hate"]),
  fear: new Set(["fear", "shadow", "dark", "creep", "tense", "wind"]),
  neutral: new Set(["still", "quiet", "contemplative"]),
};

const GENRE_LEXICON: Record<string, Set<string>> = {
  rap: new Set(["crown", "throne", "chain", "gold", "city", "skyline", "king", "rise", "hunger", "choir", "lambo"]),
  sad: new Set(["shadow", "rain", "tears", "lost", "memories", "dark", "fog", "broken", "alone", "silence", "flickering"]),
  pop: new Set(["dancing", "neon", "bright", "shine", "glitter", "confetti", "golden", "rhythm", "alive", "thrive"]),
  rock: new Set(["fire", "storm", "thunder", "fight", "rebel", "wild"]),
  electronic: new Set(["pulse", "synth", "wave", "digital", "neon", "electric", "glow"]),
  folk: new Set(["mountain", "river", "wind", "whisper", "barefoot", "gravel"]),
};

const VERSION_META: Record<string, { label: string; description: string; steps: number; gs: number }> = {
  v1: { label: "V1 — Original", description: "Generic prompt from main.ipynb — basic emotion + keyword list", steps: 15, gs: 7.5 },
  v2: { label: "V2 — Enhanced Style", description: "Adds cinematic style direction and high-contrast framing", steps: 30, gs: 7.5 },
  v3: { label: "V3 — Emotion Palette", description: "Emotion-weighted color mapping and visual metaphors", steps: 30, gs: 7.5 },
  v4: { label: "V4 — Narrative", description: "Story-driven prompt with cinematic composition language", steps: 30, gs: 7.5 },
  v5: { label: "V5 — Full Pipeline", description: "Genre-aware emotions, visual metaphors, negative prompt", steps: 30, gs: 7.5 },
};

const EMOTION_VISUAL: Record<string, { palette: string; lighting: string; scene: string; mood: string }> = {
  sadness: { palette: "deep cobalt blue and royal purple, indigo highlights", lighting: "soft rim lighting, single warm light breaking through fog", scene: "foggy park at dusk, lone figure under a streetlamp", mood: "melancholic, intimate, quiet" },
  joy: { palette: "warm golden yellow, coral pink, sky blue, vibrant orange", lighting: "golden hour sunlight, lens flares, soft bokeh", scene: "figure mid-leap with confetti, open summer sky", mood: "euphoric, weightless, radiant" },
  anger: { palette: "crimson red and inky black, hot ember orange", lighting: "hard backlight, sparks, harsh contrast", scene: "figure facing camera, fire smoldering behind them", mood: "fierce, kinetic, raw" },
  fear: { palette: "desaturated teal and charcoal, deep navy shadows", lighting: "low-key lighting, harsh single key light", scene: "figure looking up toward storm clouds", mood: "tense, restrained, atmospheric" },
  neutral: { palette: "muted earth tones, soft sage and dusty rose", lighting: "overcast diffused daylight", scene: "figure standing still, looking off-frame", mood: "contemplative, still, observational" },
};

const GENRE_STYLE: Record<string, string> = {
  sad: "2000s indie album cover, 35mm film grain, cinematic photograph",
  pop: "contemporary pop album art, glossy editorial photograph",
  rap: "hip-hop album cover, editorial fashion photograph, hard light",
  rock: "classic rock album cover, gritty 35mm film",
  electronic: "electronic album cover, synthwave aesthetic, neon-lit photograph",
  folk: "folk album cover, natural light, large format photograph",
};

const UNIVERSAL_NEGATIVE =
  "text, watermark, ugly, deformed, low quality, blurry, brown, tan, beige, hallway, corridor, stock photo";

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z']+/g) ?? [];
}

function preprocess(text: string): string[] {
  const cleaned = text.toLowerCase().replace(/\[.*?\]/g, " ").replace(/\s+/g, " ");
  return tokenize(cleaned).filter((t) => !STOP.has(t) && t.length > 1);
}

function extractKeywords(lyrics: string): string[] {
  const tokens = preprocess(lyrics);
  const freq = new Map<string, number>();
  for (const t of tokens) freq.set(t, (freq.get(t) ?? 0) + 1);
  return [...freq.entries()].sort((a, b) => b[1] - a[1]).map(([w]) => w).slice(0, 7);
}

function classifyEmotions(lyrics: string): EmotionScore[] {
  const tokens = new Set(tokenize(lyrics));
  const scores: Record<string, number> = {};
  for (const [emotion, words] of Object.entries(EMOTION_WORDS)) {
    scores[emotion] = 0.1;
    for (const w of words) if (tokens.has(w)) scores[emotion] += 1;
  }
  const total = Object.values(scores).reduce((a, b) => a + b, 0) || 1;
  return Object.entries(scores)
    .map(([label, score]) => ({ label, score: score / total }))
    .sort((a, b) => b.score - a.score);
}

function primaryGenre(lyrics: string): string {
  const tokens = tokenize(lyrics);
  const counts: Record<string, number> = {};
  for (const [genre, vocab] of Object.entries(GENRE_LEXICON)) {
    counts[genre] = tokens.filter((t) => vocab.has(t)).length;
  }
  const best = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  return best && best[1] > 0 ? best[0] : "pop";
}

function emotionTuples(emotions: EmotionScore[]): [string, number][] {
  return emotions.map((e) => [e.label, e.score]);
}

function buildV1(emotions: [string, number][], keywords: string[]): string {
  return `An album cover art that conveys ${emotions.slice(0, 3).map((e) => e[0]).join(", ")} emotions featuring ${keywords.slice(0, 5).join(", ")}`;
}

function buildV2(emotions: [string, number][], keywords: string[]): string {
  return `A dark, moody album cover art conveying ${emotions.slice(0, 3).map((e) => e[0]).join(", ")} emotions. Featured imagery: ${keywords.slice(0, 5).join(", ")}. Style: dramatic, cinematic, high contrast.`;
}

function buildV3(emotions: [string, number][], keywords: string[]): string {
  const primary = emotions[0]?.[0] ?? "emotional";
  const map: Record<string, string> = { sadness: "deep blues and purples", fear: "dark shadows and fog", joy: "bright warm colors and light", anger: "reds and intense contrast", neutral: "muted tones" };
  return `Album cover: ${primary.toUpperCase()}. Visual palette: ${map[primary] ?? "atmospheric"}. Key elements: ${keywords.slice(0, 5).join(", ")}. Professional music artwork, high quality.`;
}

function buildV4(emotions: [string, number][], keywords: string[]): string {
  const narratives: Record<string, string> = { sadness: "An introspective journey through melancholy", fear: "A tense, unsettling emotional landscape", joy: "A celebration of light and euphoria", anger: "Raw intensity and passionate conflict", neutral: "A contemplative scene" };
  const primary = emotions[0]?.[0] ?? "emotion";
  return `${narratives[primary] ?? "An emotional journey"}. Incorporates: ${keywords.slice(0, 5).join(", ")}. Professional album cover design, cinematic lighting, artfully composed.`;
}

function buildV5(emotions: EmotionScore[], keywords: string[], genre: string): { prompt: string; negative: string; gs: number } {
  const primary = emotions[0]?.label ?? "neutral";
  const visual = EMOTION_VISUAL[primary] ?? EMOTION_VISUAL.neutral;
  const style = GENRE_STYLE[genre] ?? GENRE_STYLE.pop;
  const prompt = `${visual.palette}, ${visual.lighting}, subject fills 60% of frame, medium shot, centered composition, ${visual.scene}, including ${keywords.slice(0, 4).join(", ")}, ${visual.mood}, ${style}, square album cover, 1:1 aspect, highly detailed`;
  return { prompt, negative: UNIVERSAL_NEGATIVE, gs: genre === "sad" ? 8.5 : 7.5 };
}

function buildPrompts(lyrics: string) {
  const keywords = extractKeywords(lyrics);
  const emotions = classifyEmotions(lyrics);
  const tuples = emotionTuples(emotions);
  const genre = primaryGenre(lyrics);
  const primary = emotions[0]?.label ?? "neutral";

  const builders: Record<string, () => { prompt: string; negative: string }> = {
    v1: () => ({ prompt: buildV1(tuples, keywords), negative: "" }),
    v2: () => ({ prompt: buildV2(tuples, keywords), negative: "" }),
    v3: () => ({ prompt: buildV3(tuples, keywords), negative: "" }),
    v4: () => ({ prompt: buildV4(tuples, keywords), negative: "" }),
    v5: () => {
      const v5 = buildV5(emotions, keywords, genre);
      return { prompt: v5.prompt, negative: v5.negative };
    },
  };

  return (["v1", "v2", "v3", "v4", "v5"] as const).map((version) => {
    const meta = VERSION_META[version];
    const { prompt, negative } = builders[version]();
    const gs = version === "v5" ? buildV5(emotions, keywords, genre).gs : meta.gs;
    return {
      version,
      label: meta.label,
      description: meta.description,
      prompt,
      negative_prompt: negative,
      keywords,
      emotions: emotions.slice(0, 4).map((e) => ({ label: e.label, score: Math.round(e.score * 1000) / 1000 })),
      primary_emotion: primary,
      primary_genre: genre,
      guidance_scale: gs,
      num_inference_steps: meta.steps,
    };
  });
}

async function fetchImage(prompt: string, negative: string, seed: number): Promise<string> {
  let full = prompt;
  if (negative) full += `. Avoid: ${negative.split(", ").slice(0, 6).join(", ")}`;
  const encoded = encodeURIComponent(full.slice(0, 900));
  const url = `https://image.pollinations.ai/prompt/${encoded}?width=512&height=512&seed=${seed}&nologo=true`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Image fetch failed (${resp.status})`);
  const blob = await resp.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      resolve(dataUrl.split(",")[1] ?? "");
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export async function generateClientSide(lyrics: string, seed: number): Promise<{ results: PipelineResult[]; seed: number; nlp_backend: string }> {
  const bundles = buildPrompts(lyrics);
  const results = await Promise.all(
    bundles.map(async (b) => {
      try {
        const image_base64 = await fetchImage(b.prompt, b.negative_prompt, seed);
        return { ...b, image_base64, image_backend: "pollinations" };
      } catch {
        return { ...b, image_base64: "", image_backend: "failed" };
      }
    }),
  );
  return { results, seed, nlp_backend: "client" };
}

export function useClientApi(): boolean {
  if (import.meta.env.VITE_USE_CLIENT_API === "true") return true;
  if (typeof window !== "undefined" && window.location.hostname.endsWith("github.io")) {
    return true;
  }
  return false;
}

export const SAMPLE_LYRICS = `When the night falls and shadows creep
I find myself lost in memories deep
Every word you said, every tear I've cried
In this darkness, I search for the light inside
Feel the weight of the world on my shoulders now
But I'll rise again, I'll figure it out somehow`;
