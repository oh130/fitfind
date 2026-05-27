export type RecommendationItem = {
  id: number;
  title: string;
  brand: string;
  price: string;
  reason: string;
  rank: number;
  score: number;
  accent: string;
  imageUrl?: string;
};

export type RecommendationStage = {
  label: string;
  value: string;
};

export type RecommendationBundle = {
  items: RecommendationItem[];
  totalLatency: string;
  stages: RecommendationStage[];
  persona: string;
};

export type OnboardingPersonaScores = Record<string, number>;

export type TargetAudience = "all" | "women" | "men" | "kids";

export type SearchItem = {
  id: number;
  title: string;
  brand: string;
  price: string;
  similarity: number;
  searchType: string;
  responseTime: string;
  summary: string;
  accent: string;
  imageUrl?: string;
};

export type PersonalizedSearchBundle = {
  similarity: {
    items: SearchItem[];
    responseTime: string;
  };
  personalized: {
    items: SearchItem[];
    responseTime: string;
    persona: string;
  };
};

export type BudgetSetItem = {
  id: number;
  title: string;
  brand: string;
  price: string;
  score: number;
  category: string;
  accent: string;
  imageUrl?: string;
};

export type BudgetSetBundle = {
  budget: number;
  setCount: number;
  sets: BudgetSetItem[][];
};

export type ResultExplanation = {
  id: number;
  reason: string;
  reason_source?: "gemini" | "local_fallback" | string;
};

const recommendationFallbackPalette = [
  "linear-gradient(135deg, #3f284f 0%, #121520 100%)",
  "linear-gradient(135deg, #65513a 0%, #171a23 100%)",
  "linear-gradient(135deg, #0a5960 0%, #13161d 100%)",
  "linear-gradient(135deg, #28465d 0%, #141720 100%)",
  "linear-gradient(135deg, #5a3f3f 0%, #15161f 100%)",
];

const searchFallbackPalette = [
  "linear-gradient(135deg, #35244d 0%, #161822 100%)",
  "linear-gradient(135deg, #84553a 0%, #1a1d26 100%)",
  "linear-gradient(135deg, #04545f 0%, #131620 100%)",
  "linear-gradient(135deg, #26314c 0%, #11151d 100%)",
  "linear-gradient(135deg, #5b402f 0%, #181720 100%)",
];

type ApiRecommendationItem = {
  id?: number;
  item_id?: number | string;
  product_id?: number | string;
  title?: string;
  name?: string;
  brand?: string;
  price?: string | number;
  price_estimated?: boolean;
  reason?: string;
  reason_text?: string;
  rank?: number;
  score?: number;
  accent?: string;
  image_url?: string;
  imageUrl?: string;
  img_url?: string;
  thumbnail_url?: string;
  thumbnailUrl?: string;
};

type ApiRecommendationResponse = {
  items?: ApiRecommendationItem[];
  recommendations?: ApiRecommendationItem[];
  totalLatency?: string;
  total_ms?: number;
  latency_ms?: number;
  candidate_ms?: number;
  ranking_ms?: number;
  reranking_ms?: number;
  pipeline_latency?: {
    candidate_ms?: number;
    ranking_ms?: number;
    reranking_ms?: number;
    total_ms?: number;
  };
  persona?: string;
};

type ApiSearchItem = {
  id?: number;
  item_id?: number | string;
  product_id?: number | string;
  title?: string;
  name?: string;
  brand?: string;
  price?: string | number;
  price_estimated?: boolean;
  summary?: string;
  description?: string;
  score?: number;
  similarity?: number;
  accent?: string;
  image_url?: string;
  imageUrl?: string;
  img_url?: string;
  thumbnail_url?: string;
  thumbnailUrl?: string;
};

type ApiSearchResponse =
  | {
      items?: ApiSearchItem[];
      results?: ApiSearchItem[];
      latency_ms?: number;
      total_ms?: number;
      mode?: string;
    }
  | ApiSearchItem[];

type ApiPersonalizedSearchResponse = {
  search_results?: ApiSearchItem[];
  personalized_results?: ApiRecommendationItem[];
  search_latency_ms?: number;
  personalized_latency?: {
    candidate_ms?: number;
    ranking_ms?: number;
    reranking_ms?: number;
    total_ms?: number;
  };
  persona?: string;
};

type ApiBudgetSetItem = {
  article_id?: number | string;
  product_id?: number | string;
  id?: number | string;
  name?: string;
  title?: string;
  brand?: string;
  price?: string | number;
  price_int?: number;
  price_estimated?: boolean;
  score?: number;
  category?: string;
  image_url?: string;
  imageUrl?: string;
  img_url?: string;
  thumbnail_url?: string;
  thumbnailUrl?: string;
};

type ApiBudgetSetResponse = {
  budget?: number;
  set_count?: number;
  sets?: ApiBudgetSetItem[][];
};

function getApiBaseUrl(): string {
  return (import.meta.env.VITE_API_BASE_URL ?? "").trim();
}

function buildApiUrl(path: string): string {
  const baseUrl = getApiBaseUrl();
  if (!baseUrl) {
    return path;
  }

  return `${baseUrl.replace(/\/$/, "")}${path}`;
}

function formatApiDetail(detail: unknown): string {
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }
        if (item && typeof item === "object" && "msg" in item) {
          return String((item as { msg: unknown }).msg);
        }
        return JSON.stringify(item);
      })
      .join(", ");
  }
  if (detail && typeof detail === "object") {
    return JSON.stringify(detail);
  }
  return "";
}

async function buildApiErrorMessage(response: Response, fallback: string): Promise<string> {
  const text = await response.text().catch(() => "");
  if (!text) {
    return `${fallback} (${response.status})`;
  }

  try {
    const payload = JSON.parse(text) as {
      detail?: unknown;
      message?: unknown;
      error?: unknown;
    };
    const detail =
      formatApiDetail(payload.detail) ||
      formatApiDetail(payload.message) ||
      formatApiDetail(payload.error);
    return detail ? `${fallback} (${response.status}): ${detail}` : `${fallback} (${response.status})`;
  } catch {
    return `${fallback} (${response.status}): ${text.slice(0, 240)}`;
  }
}

function toCurrencyLabel(value: string | number | undefined, estimated = false): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    if (value <= 0) {
      return "가격 정보 없음";
    }
    return `${estimated ? "추정 " : ""}${value.toLocaleString("ko-KR")}원`;
  }

  if (typeof value === "string" && value.trim()) {
    const trimmed = value.trim();
    if (trimmed === "0" || trimmed === "0원") {
      return "가격 정보 없음";
    }
    return estimated ? `추정 ${trimmed}` : trimmed;
  }

  return "가격 정보 없음";
}

function cleanDisplayText(value: string | undefined, fallback: string): string {
  return (value ?? fallback).replace(/Â·/g, "·").replace(/\s+/g, " ").trim();
}

function toLatencyLabel(value: number | string | undefined, fallback: string): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return `${Math.round(value)}ms`;
  }

  if (typeof value === "string" && value.trim()) {
    return value;
  }

  return fallback;
}

function toNumericId(...values: Array<number | string | undefined>): number {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }

    if (typeof value === "string") {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }

  return Date.now() + Math.floor(Math.random() * 1000);
}

function toImageUrl(item: {
  image_url?: string;
  imageUrl?: string;
  img_url?: string;
  thumbnail_url?: string;
  thumbnailUrl?: string;
}): string | undefined {
  return item.image_url ?? item.imageUrl ?? item.img_url ?? item.thumbnail_url ?? item.thumbnailUrl;
}

function toFiniteScore(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  return fallback;
}

function toRelativeScores(rawScores: number[]): number[] {
  if (rawScores.length === 0) {
    return [];
  }

  const finiteScores = rawScores.map((score) => (Number.isFinite(score) ? score : 0));
  const minScore = Math.min(...finiteScores);
  const shiftedScores = minScore < 0 ? finiteScores.map((score) => score - minScore) : finiteScores;
  const maxScore = Math.max(...shiftedScores);

  if (maxScore <= 0) {
    return finiteScores.map((_, index) => Math.max(0.5, 0.95 - index * 0.04));
  }

  return shiftedScores.map((score) => Math.max(0, Math.min(1, score / maxScore)));
}

function normalizeRecommendationBundle(
  payload: ApiRecommendationResponse,
  topN: number,
): RecommendationBundle {
  const rawItems = (payload.items ?? payload.recommendations ?? []).slice(0, topN);
  const pipelineLatency = payload.pipeline_latency;
  const totalLatencyMs = payload.total_ms ?? payload.latency_ms ?? pipelineLatency?.total_ms;
  const candidateMs = payload.candidate_ms ?? pipelineLatency?.candidate_ms;
  const rankingMs = payload.ranking_ms ?? pipelineLatency?.ranking_ms;
  const rerankingMs = payload.reranking_ms ?? pipelineLatency?.reranking_ms;
  const relativeScores = toRelativeScores(
    rawItems.map((item, index) => toFiniteScore(item.score, Math.max(0.5, 0.95 - index * 0.04))),
  );
  const items = rawItems.map((item, index) => ({
    id: toNumericId(item.id, item.item_id, item.product_id),
    title: cleanDisplayText(item.title ?? item.name, `Recommendation ${index + 1}`),
    brand: cleanDisplayText(item.brand, "Unknown Brand"),
    price: toCurrencyLabel(item.price, item.price_estimated),
    reason:
      item.reason_text ??
      item.reason ??
      "추천 이유 정보가 아직 제공되지 않았습니다.",
    rank: item.rank ?? index + 1,
    score: relativeScores[index] ?? Math.max(0.5, 0.95 - index * 0.04),
    accent: item.accent ?? recommendationFallbackPalette[index % recommendationFallbackPalette.length],
    imageUrl: toImageUrl(item),
  }));

  return {
    items,
    totalLatency: toLatencyLabel(totalLatencyMs ?? payload.totalLatency, "0ms"),
    persona: payload.persona ?? "개인화 추천",
    stages: [
      candidateMs !== undefined ? { label: "Candidate", value: toLatencyLabel(candidateMs, "0ms") } : null,
      rankingMs !== undefined ? { label: "Ranking", value: toLatencyLabel(rankingMs, "0ms") } : null,
      rerankingMs !== undefined ? { label: "Reranking", value: toLatencyLabel(rerankingMs, "0ms") } : null,
    ].filter((stage): stage is RecommendationStage => stage !== null),
  };
}

function normalizeSearchItems(
  payload: ApiSearchResponse,
  fallbackMode: string,
): { items: SearchItem[]; responseTime: string } {
  const rawItems = Array.isArray(payload) ? payload : payload.items ?? payload.results ?? [];
  const responseTime = Array.isArray(payload)
    ? "0ms"
    : toLatencyLabel(payload.latency_ms ?? payload.total_ms, "0ms");
  const searchType =
    fallbackMode === "multimodal" ? "텍스트 + 이미지" : fallbackMode === "image" ? "이미지" : "텍스트";

  return {
    responseTime,
    items: rawItems.map((item, index) => ({
      id: toNumericId(item.id, item.item_id, item.product_id),
      title: cleanDisplayText(item.title ?? item.name, `Search Result ${index + 1}`),
      brand: cleanDisplayText(item.brand, "Unknown Brand"),
      price: toCurrencyLabel(item.price, item.price_estimated),
      similarity:
        typeof item.similarity === "number"
          ? item.similarity
          : typeof item.score === "number"
            ? item.score
            : Math.max(0.5, 0.95 - index * 0.05),
      searchType,
      responseTime,
      summary: item.summary ?? item.description ?? "검색 결과 설명이 제공되지 않았습니다.",
      accent: item.accent ?? searchFallbackPalette[index % searchFallbackPalette.length],
      imageUrl: toImageUrl(item),
    })),
  };
}

function normalizePersonalizedSearchItems(
  rawItems: ApiRecommendationItem[],
  responseTime: string,
  persona: string,
): SearchItem[] {
  const relativeScores = toRelativeScores(
    rawItems.map((item, index) => toFiniteScore(item.score, Math.max(0.5, 0.95 - index * 0.04))),
  );

  return rawItems.map((item, index) => ({
    id: toNumericId(item.id, item.item_id, item.product_id),
    title: cleanDisplayText(item.title ?? item.name, `Personalized Result ${index + 1}`),
    brand: cleanDisplayText(item.brand, "Unknown Brand"),
    price: toCurrencyLabel(item.price, item.price_estimated),
    similarity: relativeScores[index] ?? Math.max(0.5, 0.95 - index * 0.04),
    searchType: "내 취향 반영",
    responseTime,
    summary:
      item.reason_text ??
      item.reason ??
      `${persona} 선호와 현재 검색 후보를 함께 반영해 재정렬한 결과입니다.`,
    accent: item.accent ?? recommendationFallbackPalette[index % recommendationFallbackPalette.length],
    imageUrl: toImageUrl(item),
  }));
}

function normalizeBudgetSetBundle(payload: ApiBudgetSetResponse): BudgetSetBundle {
  const sets = (payload.sets ?? []).map((setItems, setIndex) =>
    {
      const relativeScores = toRelativeScores(
        setItems.map((item, itemIndex) => toFiniteScore(item.score, Math.max(0.5, 0.9 - itemIndex * 0.08))),
      );

      return setItems.map((item, itemIndex) => ({
        id: toNumericId(item.id, item.article_id, item.product_id),
        title: cleanDisplayText(item.title ?? item.name, `Set Item ${itemIndex + 1}`),
        brand: cleanDisplayText(item.brand, "Unknown Brand"),
        price: toCurrencyLabel(item.price_int ?? item.price, item.price_estimated),
        score: relativeScores[itemIndex] ?? Math.max(0.5, 0.9 - itemIndex * 0.08),
        category: item.category ?? "Unknown Category",
        accent:
          recommendationFallbackPalette[(setIndex + itemIndex) % recommendationFallbackPalette.length],
        imageUrl: toImageUrl(item),
      }));
    },
  );

  return {
    budget: typeof payload.budget === "number" ? payload.budget : 0,
    setCount: typeof payload.set_count === "number" ? payload.set_count : sets.length,
    sets,
  };
}

export async function fetchRecommendations(
  userId: string,
  topN: number,
  _seed: number,
  options?: {
    personaHint?: string;
    personalizationWeight?: number;
    priceWeight?: number;
    popularityWeight?: number;
    includeReasons?: boolean;
  },
): Promise<RecommendationBundle> {
  const url = new URL(buildApiUrl("/api/recommend"), window.location.origin);
  url.searchParams.set("user_id", userId);
  url.searchParams.set("top_n", String(topN));
  if (options?.personaHint) {
    url.searchParams.set("persona_hint", options.personaHint);
  }
  if (options?.personalizationWeight !== undefined) {
    url.searchParams.set("personalization_weight", options.personalizationWeight.toFixed(2));
  }
  if (options?.priceWeight !== undefined) {
    url.searchParams.set("price_weight", options.priceWeight.toFixed(2));
  }
  if (options?.popularityWeight !== undefined) {
    url.searchParams.set("popularity_weight", options.popularityWeight.toFixed(2));
  }
  if (options?.includeReasons) {
    url.searchParams.set("include_reasons", "true");
  }

  const response = await fetch(url.toString(), {
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(await buildApiErrorMessage(response, "추천 API 호출 실패"));
  }

  const payload = (await response.json()) as ApiRecommendationResponse;
  return normalizeRecommendationBundle(payload, topN);
}

export async function fetchResultExplanations(input: {
  userId: string;
  query: string;
  persona?: string | null;
  targetAudience?: TargetAudience;
  items: Array<{
    id: number;
    title: string;
    brand: string;
    price: string;
  }>;
}): Promise<ResultExplanation[]> {
  const response = await fetch(buildApiUrl("/api/explain-results"), {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: input.userId,
      query: input.query,
      persona: input.persona ?? null,
      target_audience: input.targetAudience ?? "all",
      items: input.items,
    }),
  });

  if (!response.ok) {
    throw new Error(await buildApiErrorMessage(response, "AI 추천 이유 생성 실패"));
  }

  const payload = (await response.json()) as { items?: Array<{ id: number | string; reason?: string }> };
  return (payload.items ?? []).map((item) => ({
    id: toNumericId(item.id),
    reason: item.reason ?? "추천 이유 정보가 아직 제공되지 않았습니다.",
  }));
}

export async function fetchSearchResults(params: {
  query: string;
  imageBase64?: string | null;
  topK: number;
  mode: "text" | "image" | "multimodal";
}): Promise<{ items: SearchItem[]; responseTime: string }> {
  const response = await fetch(buildApiUrl("/api/search"), {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: params.query,
      image_base64: params.imageBase64 ?? null,
      top_k: params.topK,
    }),
  });

  if (!response.ok) {
    throw new Error(await buildApiErrorMessage(response, "검색 API 호출 실패"));
  }

  const payload = (await response.json()) as ApiSearchResponse;
  return normalizeSearchItems(payload, params.mode);
}

export async function fetchPersonalizedSearchResults(params: {
  userId: string;
  query: string;
  imageBase64?: string | null;
  topK: number;
  topN: number;
  mode: "text" | "image" | "multimodal";
  personaHint?: string | null;
  personalizationWeight?: number;
  targetAudience?: TargetAudience;
}): Promise<PersonalizedSearchBundle> {
  const response = await fetch(buildApiUrl("/api/personalized-search"), {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: params.userId,
      query: params.query,
      image_base64: params.imageBase64 ?? null,
      top_k: params.topK,
      top_n: params.topN,
      persona_hint: params.personaHint ?? null,
      personalization_weight: params.personalizationWeight ?? null,
      target_audience: params.targetAudience ?? "all",
    }),
  });

  if (!response.ok) {
    throw new Error(await buildApiErrorMessage(response, "개인화 검색 API 호출 실패"));
  }

  const payload = (await response.json()) as ApiPersonalizedSearchResponse;
  const searchPayload: ApiSearchResponse = {
    results: payload.search_results ?? [],
    latency_ms: payload.search_latency_ms,
  };
  const similarity = normalizeSearchItems(searchPayload, params.mode);
  const personalizedResponseTime = toLatencyLabel(payload.personalized_latency?.total_ms, "0ms");
  const persona = payload.persona ?? "개인화 검색";

  return {
    similarity,
    personalized: {
      items: normalizePersonalizedSearchItems(
        payload.personalized_results ?? [],
        personalizedResponseTime,
        persona,
      ),
      responseTime: personalizedResponseTime,
      persona,
    },
  };
}

export async function sendInteractionEvent(input: {
  userId: string;
  itemId?: number | string;
  eventType?: "click" | "purchase" | "view" | "cart" | "search";
  category?: string;
  queryText?: string;
}): Promise<void> {
  await fetch(buildApiUrl("/api/events"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: input.userId,
      item_id: input.itemId === undefined ? null : String(input.itemId),
      event_type: input.eventType ?? "click",
      category: input.category ?? null,
      query_text: input.queryText ?? null,
    }),
  });
}

export async function fetchOnboardingPersonaScores(input: {
  userId: string;
  description: string;
  styleChoices: string[];
  budgetRange?: string | null;
  targetAudience?: TargetAudience;
}): Promise<OnboardingPersonaScores> {
  const response = await fetch(buildApiUrl("/api/onboarding"), {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: input.userId,
      description: input.description,
      style_choices: input.styleChoices,
      budget_range: input.budgetRange ?? null,
      target_audience: input.targetAudience ?? "all",
    }),
  });

  if (!response.ok) {
    throw new Error(`Onboarding API failed with ${response.status}`);
  }

  const payload = (await response.json()) as { persona_scores?: OnboardingPersonaScores };
  return payload.persona_scores ?? {};
}

export async function selectOnboardingPersona(input: {
  userId: string;
  persona: string;
  personaScores?: OnboardingPersonaScores;
  targetAudience?: TargetAudience;
}): Promise<void> {
  const response = await fetch(buildApiUrl("/api/onboarding/select"), {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: input.userId,
      persona: input.persona,
      persona_scores: input.personaScores ?? null,
      target_audience: input.targetAudience ?? "all",
    }),
  });

  if (!response.ok) {
    throw new Error(`Onboarding select API failed with ${response.status}`);
  }
}

export async function fetchBudgetSets(input: {
  userId: string;
  budget: number;
  setCount?: number;
  query?: string | null;
  targetAudience?: TargetAudience;
}): Promise<BudgetSetBundle> {
  const url = new URL(buildApiUrl("/api/budget-set"), window.location.origin);
  url.searchParams.set("user_id", input.userId);
  url.searchParams.set("budget", String(input.budget));
  url.searchParams.set("set_count", String(input.setCount ?? 3));
  if (input.query?.trim()) {
    url.searchParams.set("query", input.query.trim());
  }
  url.searchParams.set("target_audience", input.targetAudience ?? "all");

  const response = await fetch(url.toString(), {
    method: "POST",
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Budget set API failed with ${response.status}`);
  }

  const payload = (await response.json()) as ApiBudgetSetResponse;
  return normalizeBudgetSetBundle(payload);
}
