import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  BudgetSetBundle,
  OnboardingPersonaScores,
  TargetAudience,
  fetchBudgetSets,
  fetchOnboardingPersonaScores,
  fetchPersonalizedSearchResults,
  fetchResultExplanations,
  selectOnboardingPersona,
} from "./api";
import { personaOptions } from "./personas";

type AppView = "landing" | "onboarding" | "search";
type SearchMode = "text" | "image" | "multimodal";
type SearchResultView = "similarity" | "personalized";

type SearchResult = {
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

type UploadedImage = {
  name: string;
  sizeLabel: string;
  base64: string;
  previewUrl: string;
};

const onboardingStyleOptions = ["casual", "minimal", "street", "sporty", "feminine", "classic"];

const targetAudienceOptions: Array<{ key: TargetAudience; label: string }> = [
  { key: "all", label: "전체" },
  { key: "women", label: "여성" },
  { key: "men", label: "남성" },
  { key: "kids", label: "키즈" },
];

const emptyBudgetSetBundle: BudgetSetBundle = {
  budget: 0,
  setCount: 0,
  sets: [],
};

function ResultVisual({
  imageUrl,
  title,
  accent,
}: {
  imageUrl?: string;
  title: string;
  accent: string;
}) {
  const [hasImageError, setHasImageError] = useState(false);
  const shouldShowImage = Boolean(imageUrl) && !hasImageError;

  return (
    <div className="result-visual" style={{ background: accent }}>
      {shouldShowImage ? (
        <img
          className="result-image"
          src={imageUrl}
          alt={title}
          loading="lazy"
          onError={() => setHasImageError(true)}
        />
      ) : (
        <div className="result-image-fallback" aria-label={`${title} 이미지 준비 중`}>
          <span>{title.slice(0, 1).toUpperCase()}</span>
        </div>
      )}
    </div>
  );
}

function toDisplayPercent(value: number): string {
  const normalizedValue = value > 1 ? value / 100 : value;
  const clampedValue = Math.max(0, Math.min(1, normalizedValue));
  return `${(clampedValue * 100).toFixed(1)}%`;
}

function App() {
  const [view, setView] = useState<AppView>("landing");
  const [selectedOnboardingPersona, setSelectedOnboardingPersona] = useState("trendsetter");
  const [query, setQuery] = useState("광택감 있는 블랙 아우터와 슬림 팬츠 조합");
  const [userId, setUserId] = useState("user_1024");
  const [uploadedImage, setUploadedImage] = useState<UploadedImage | null>(null);
  const [searchMode, setSearchMode] = useState<SearchMode>("multimodal");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [personalizedResults, setPersonalizedResults] = useState<SearchResult[]>([]);
  const [activeLatency, setActiveLatency] = useState("0ms");
  const [personalizedLatency, setPersonalizedLatency] = useState("0ms");
  const [searchResultView, setSearchResultView] = useState<SearchResultView>("similarity");
  const [searchResultPersona, setSearchResultPersona] = useState("개인화 검색");
  const [hasSearched, setHasSearched] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [recommendationWeight, setRecommendationWeight] = useState(0.7);
  const [targetAudience, setTargetAudience] = useState<TargetAudience>("all");
  const [isRefreshingRecommendations, setIsRefreshingRecommendations] = useState(false);
  const [recommendationError, setRecommendationError] = useState<string | null>(null);
  const [topN, setTopN] = useState(5);
  const [budget, setBudget] = useState("200000");
  const [budgetSets, setBudgetSets] = useState<BudgetSetBundle>(emptyBudgetSetBundle);
  const [isLoadingBudgetSets, setIsLoadingBudgetSets] = useState(false);
  const [budgetSetError, setBudgetSetError] = useState<string | null>(null);
  const [onboardingDescription, setOnboardingDescription] = useState("");
  const [selectedStyles, setSelectedStyles] = useState<string[]>(["minimal"]);
  const [personaScores, setPersonaScores] = useState<OnboardingPersonaScores>({});
  const [isAnalyzingOnboarding, setIsAnalyzingOnboarding] = useState(false);
  const [isSubmittingPersona, setIsSubmittingPersona] = useState(false);
  const [onboardingError, setOnboardingError] = useState<string | null>(null);
  const isManagingHistoryRef = useRef(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const budgetLabel = `${Number(budget || 0).toLocaleString("ko-KR")}원`;

  const helperMessage = useMemo(() => {
    if (searchMode === "text") {
      return "텍스트 질의만으로 유사 상품을 찾습니다.";
    }
    if (searchMode === "image") {
      return "업로드한 이미지 특징을 기반으로 시각적으로 비슷한 상품을 찾습니다.";
    }
    return "텍스트와 이미지 신호를 함께 반영해 더 강한 후보를 우선 정렬합니다.";
  }, [searchMode]);

  useEffect(() => {
    setPersonaScores({});
    setOnboardingError(null);
  }, [onboardingDescription, selectedStyles]);

  useEffect(() => {
    const handlePopState = (event: PopStateEvent) => {
      const nextView = event.state?.view as AppView | undefined;
      if (!nextView) {
        return;
      }
      isManagingHistoryRef.current = true;
      setView(nextView);
      isManagingHistoryRef.current = false;
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  useEffect(() => {
    if (isManagingHistoryRef.current) {
      return;
    }

    const currentView = window.history.state?.view as AppView | undefined;
    if (currentView === view) {
      return;
    }

    if (!currentView) {
      window.history.replaceState({ view }, "");
      return;
    }

    window.history.pushState({ view }, "");
  }, [view]);

  const clearSearchResults = () => {
    setResults([]);
    setPersonalizedResults([]);
    setActiveLatency("0ms");
    setPersonalizedLatency("0ms");
    setSearchResultPersona("개인화 검색");
    setSearchError(null);
    setHasSearched(false);
  };

  const clearUploadedImage = () => {
    setUploadedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    clearSearchResults();
    setSearchMode((currentMode) => (currentMode === "image" ? "text" : currentMode));
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      clearUploadedImage();
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const result = typeof reader.result === "string" ? reader.result : "";
      const [, base64 = ""] = result.split(",");
      const sizeInMb = file.size / (1024 * 1024);

      setUploadedImage({
        name: file.name,
        sizeLabel: `${sizeInMb.toFixed(2)}MB`,
        base64,
        previewUrl: result,
      });
      clearSearchResults();

      setSearchMode((currentMode) => (currentMode === "text" ? "multimodal" : currentMode));
    };

    reader.readAsDataURL(file);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const trimmedQuery = query.trim();
    const nextMode: SearchMode =
      trimmedQuery && uploadedImage ? "multimodal" : uploadedImage ? "image" : "text";

    setSearchMode(nextMode);
    setIsSearching(true);
    setHasSearched(true);
    setSearchError(null);
    setSearchResultView("personalized");

    try {
      const response = await fetchPersonalizedSearchResults({
        userId: userId.trim() || "anonymous",
        query: trimmedQuery,
        imageBase64: uploadedImage?.base64 ?? null,
        topK: 150,
        topN,
        mode: nextMode,
        personaHint: selectedOnboardingPersona,
        personalizationWeight: recommendationWeight,
        targetAudience,
      });

      setResults(response.similarity.items);
      setActiveLatency(response.similarity.responseTime);
      setPersonalizedResults(response.personalized.items);
      setPersonalizedLatency(response.personalized.responseTime);
      setSearchResultPersona(response.personalized.persona);
    } catch (error) {
      setResults([]);
      setActiveLatency("0ms");
      setPersonalizedResults([]);
      setPersonalizedLatency("0ms");
      setSearchResultPersona("개인화 검색");
      setSearchResultView("personalized");
      setSearchError(error instanceof Error ? error.message : "검색 결과를 불러오지 못했습니다.");
    } finally {
      setIsSearching(false);
    }
  };

  const toggleStyleChoice = (style: string) => {
    setSelectedStyles((current) =>
      current.includes(style) ? current.filter((value) => value !== style) : [...current, style],
    );
  };

  const updatePersonaScore = (personaKey: string, nextValue: number) => {
    setSelectedOnboardingPersona(personaKey);
    setPersonaScores((current) => {
      const clampedValue = Math.max(0, Math.min(100, Math.round(nextValue)));
      const nextScores = {
        ...current,
        [personaKey]: clampedValue,
      };
      const total = Object.values(nextScores).reduce((sum, value) => sum + value, 0);

      if (total > 100) {
        return current;
      }

      return nextScores;
    });
  };

  const goToOnboarding = () => {
    setView("onboarding");
  };

  const runOnboardingAnalysis = async () => {
    if (!userId.trim() || !onboardingDescription.trim()) {
      setOnboardingError("사용자 ID와 취향 입력 내용을 입력해 주세요.");
      return;
    }

    setIsAnalyzingOnboarding(true);
    setOnboardingError(null);

    try {
      const scores = await fetchOnboardingPersonaScores({
        userId: userId.trim(),
        description: onboardingDescription.trim(),
        styleChoices: selectedStyles,
        budgetRange: null,
        targetAudience,
      });

      setPersonaScores(scores);
      const topPersona = Object.entries(scores).sort((a, b) => b[1] - a[1])[0]?.[0];
      if (topPersona) {
        setSelectedOnboardingPersona(topPersona);
      }
    } catch {
      setOnboardingError("페르소나 분석에 실패했습니다. 백엔드 설정을 확인해 주세요.");
    } finally {
      setIsAnalyzingOnboarding(false);
    }
  };

  const loadBudgetSets = async () => {
    const parsedBudget = Number(budget);
    if (!userId.trim() || !Number.isFinite(parsedBudget) || parsedBudget <= 0) {
      setBudgetSetError("유효한 사용자 ID와 예산을 입력해 주세요.");
      return;
    }

    setIsLoadingBudgetSets(true);
    setBudgetSetError(null);

    try {
      const bundle = await fetchBudgetSets({
        userId: userId.trim(),
        budget: parsedBudget,
        setCount: 3,
        query: query.trim() || null,
        targetAudience,
      });
      setBudgetSets(bundle);
    } catch {
      setBudgetSetError("예산 세트 추천 결과를 불러오지 못했습니다.");
      setBudgetSets(emptyBudgetSetBundle);
    } finally {
      setIsLoadingBudgetSets(false);
    }
  };

  const loadAiRecommendations = async () => {
    if (!userId.trim()) {
      setRecommendationError("사용자 정보를 먼저 설정해 주세요.");
      return;
    }

    const targetResults = hasPersonalizedSearchResults ? personalizedResults : results;
    if (targetResults.length === 0) {
      setRecommendationError("먼저 검색 결과를 불러온 뒤 추천 이유를 확인해 주세요.");
      return;
    }

    setIsRefreshingRecommendations(true);
    setRecommendationError(null);

    try {
      const explanations = await fetchResultExplanations({
        userId: userId.trim(),
        query: query.trim(),
        persona: selectedOnboardingPersona,
        targetAudience,
        items: targetResults.map((item) => ({
          id: item.id,
          title: item.title,
          brand: item.brand,
          price: item.price,
        })),
      });

      const reasonById = new Map(explanations.map((item) => [item.id, item.reason]));
      const nextResults = targetResults.map((item) => ({
        ...item,
        summary: reasonById.get(item.id) ?? item.summary,
      }));

      if (hasPersonalizedSearchResults) {
        setPersonalizedResults(nextResults);
      } else {
        setResults(nextResults);
      }

      if (reasonById.size === 0) {
        setRecommendationError("현재 검색 결과에 대한 추천 이유를 생성하지 못했습니다.");
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "AI 추천 이유를 불러오지 못했습니다.";
      setRecommendationError(message);
    } finally {
      setIsRefreshingRecommendations(false);
    }
  };

  const startWithPersona = async () => {
    setIsSubmittingPersona(true);
    setOnboardingError(null);

    try {
      await selectOnboardingPersona({
        userId: userId.trim() || "anonymous",
        persona: selectedOnboardingPersona,
        personaScores,
        targetAudience,
      });
      setBudgetSets(emptyBudgetSetBundle);
      setView("search");
    } catch {
      setOnboardingError("선택한 페르소나를 저장하지 못했습니다.");
    } finally {
      setIsSubmittingPersona(false);
    }
  };

  const modeLabel =
    searchMode === "multimodal" ? "멀티모달" : searchMode === "image" ? "이미지" : "텍스트";
  const personaScoreTotal = Object.values(personaScores).reduce((sum, value) => sum + value, 0);
  const isPersonaScoreTotalValid = personaScoreTotal === 100;
  const hasPersonalizedSearchResults = hasSearched && personalizedResults.length > 0;
  const activeSearchResults = searchResultView === "personalized" ? personalizedResults : results;
  const activeSearchLatency = searchResultView === "personalized" ? personalizedLatency : activeLatency;
  const activeSearchScoreLabel = searchResultView === "personalized" ? "추천 점수" : "유사도";
  const mergedSearchResults = hasPersonalizedSearchResults ? personalizedResults : activeSearchResults;
  const mergedSearchLatency = hasPersonalizedSearchResults ? personalizedLatency : activeSearchLatency;
  const mergedSearchScoreLabel = hasPersonalizedSearchResults ? "추천 점수" : activeSearchScoreLabel;
  const searchEmptyMessage = !hasSearched
    ? "검색을 실행하면 유사도순 결과와 내 취향순 결과가 여기에 표시됩니다."
    : searchError
      ? searchError
      : searchResultView === "personalized"
        ? "검색 후보 안에서 개인화된 결과가 아직 없습니다. 검색을 다시 시도해 주세요."
        : "검색 결과가 없습니다. 검색어를 조금 더 구체적으로 바꿔 보세요.";

  if (view === "landing") {
    return (
      <div className="app-shell landing-shell">
        <section className="landing-panel">
          <h1>Fit-Find</h1>
          <p className="landing-description">
            멀티모달 검색과 개인화 추천을 결합한 패션 탐색 서비스.
          </p>
          <div className="landing-actions">
            <button type="button" className="primary-button landing-start-button" onClick={goToOnboarding}>
              시작하기
            </button>
          </div>
        </section>
      </div>
    );
  }

  if (view === "onboarding") {
    return (
      <div className="app-shell onboarding-shell">
        <section className="onboarding-panel">
          <div className="onboarding-copy">
            <p className="eyebrow">Personalization Setup</p>
            <h1>개인화 추천을 위한 페르소나 설정</h1>
            <p>취향 정보를 바탕으로 추천 결과를 맞춤 설정합니다.</p>
          </div>

          <div className="search-composer">
            <div className="target-audience-panel">
              <span>쇼핑 대상</span>
              <div className="target-audience-buttons" role="group" aria-label="쇼핑 대상 선택">
                {targetAudienceOptions.map((option) => (
                  <button
                    key={option.key}
                    type="button"
                    className={targetAudience === option.key ? "mini-button active" : "mini-button"}
                    onClick={() => setTargetAudience(option.key)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            <label className="user-id-field">
              <span>User ID</span>
              <input
                value={userId}
                onChange={(event) => setUserId(event.target.value)}
                placeholder="예: user_1024"
                aria-label="온보딩 사용자 ID"
              />
            </label>

            <label className="search-box">
              <span>취향 입력</span>
              <input
                value={onboardingDescription}
                onChange={(event) => setOnboardingDescription(event.target.value)}
                placeholder="예: 미니멀한 블랙 아우터와 실용적인 출근룩을 자주 입습니다."
                aria-label="온보딩 취향 입력"
              />
            </label>

            <div className="signal-list">
              {onboardingStyleOptions.map((style) => (
                <button
                  key={style}
                  type="button"
                  className={selectedStyles.includes(style) ? "mini-button active" : "mini-button"}
                  onClick={() => toggleStyleChoice(style)}
                >
                  {style}
                </button>
              ))}
            </div>

            <div className="recommendation-toolbar">
              <div className="recommendation-actions">
                <button type="button" className="mini-button" onClick={() => setView("landing")}>
                  이전
                </button>
                <button
                  type="button"
                  className="primary-button"
                  onClick={runOnboardingAnalysis}
                  disabled={isAnalyzingOnboarding}
                >
                  {isAnalyzingOnboarding ? "분석 중..." : "취향 분석하기"}
                </button>
              </div>
            </div>
          </div>

          {Object.keys(personaScores).length > 0 ? (
            <div className="persona-grid">
              {personaOptions.map((persona) => (
                <article
                  key={persona.key}
                  className={
                    persona.key === selectedOnboardingPersona ? "persona-option active" : "persona-option"
                  }
                >
                  <p className="persona-name">{persona.name}</p>
                  <h2>{persona.title}</h2>
                  <p className="persona-summary">{persona.summary}</p>
                  <div className="persona-score-row">
                    <strong>{personaScores[persona.key] ?? 0}%</strong>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={personaScores[persona.key] ?? 0}
                      onChange={(event) => updatePersonaScore(persona.key, Number(event.target.value))}
                      aria-label={`${persona.name} 비율 조절`}
                    />
                  </div>
                  <div className="persona-traits">
                    {persona.traits.map((trait) => (
                      <span key={trait} className="badge">
                        {trait}
                      </span>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          ) : null}

          <div className="onboarding-footer">
            <div className="persona-card">
              <span>총합 점수</span>
              <strong>{personaScoreTotal}%</strong>
            </div>
            <button
              type="button"
              className="primary-button"
              onClick={startWithPersona}
              disabled={
                isSubmittingPersona || Object.keys(personaScores).length === 0 || !isPersonaScoreTotalValid
              }
            >
              {isSubmittingPersona ? "저장 중..." : "검색 페이지로 이동"}
            </button>
          </div>

          {Object.keys(personaScores).length > 0 ? (
            <p className="persona-adjustment-note">
              슬라이더를 조정하면서 원하는 비중으로 맞춰 보세요.
              {!isPersonaScoreTotalValid ? " 전체 합계가 100%가 되어야 다음 단계로 진행할 수 있습니다." : ""}
            </p>
          ) : null}

          {onboardingError ? <p className="status-text">{onboardingError}</p> : null}
        </section>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Fit-Find</p>
          <h1>Fit-Find: 취향 기반 멀티모달 패션 검색 및 추천</h1>
        </div>
        <div className="topbar-meta">
          <span>현재 사용자: {userId}</span>
          <button type="button" className="mini-button" onClick={() => setView("onboarding")}>
            페르소나 다시 정하기
          </button>
        </div>
      </header>

      <main className="layout">
        <section className="hero-panel">
          <div className="hero-copy">
            <p className="eyebrow">Search Experience</p>
            <h2>검색 입력</h2>

            <div className="panel weight-panel">
              <div className="weight-copy">
                <p className="eyebrow">Result Balance</p>
                <h4>추천 반영도</h4>
              </div>
              <div className="weight-control">
                <div className="weight-labels">
                  <span>검색어 중심</span>
                  <span>취향 반영</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="5"
                  value={Math.round(recommendationWeight * 100)}
                  onChange={(event) => {
                    setRecommendationWeight(Number(event.target.value) / 100);
                    clearSearchResults();
                  }}
                  aria-label="추천 반영도"
                />
              </div>
            </div>

            <div className="search-actions">
              <button type="submit" form="search-composer-form" className="primary-button" disabled={isSearching}>
                {isSearching ? "검색 중..." : "검색 실행"}
              </button>
              <button
                type="button"
                className="primary-button"
                onClick={loadAiRecommendations}
                disabled={isRefreshingRecommendations}
              >
                {isRefreshingRecommendations ? "AI 추천 이유 불러오는 중..." : "AI 추천 이유"}
              </button>
            </div>
            <p className="search-hint">텍스트만, 이미지만, 또는 둘을 함께 사용해 검색할 수 있습니다.</p>
            {recommendationError ? <p className="status-text">{recommendationError}</p> : null}
          </div>

          <form id="search-composer-form" className="search-composer" onSubmit={handleSubmit}>
            <div className="search-tabs" aria-label="검색 모드">
              <button
                type="button"
                className={searchMode === "text" ? "active" : ""}
                onClick={() => {
                  setSearchMode("text");
                  clearSearchResults();
                }}
              >
                텍스트
              </button>
              <button
                type="button"
                className={searchMode === "image" ? "active" : ""}
                onClick={() => {
                  setSearchMode("image");
                  clearSearchResults();
                }}
              >
                이미지
              </button>
              <button
                type="button"
                className={searchMode === "multimodal" ? "active" : ""}
                onClick={() => {
                  setSearchMode("multimodal");
                  clearSearchResults();
                }}
              >
                텍스트 + 이미지
              </button>
            </div>

            <label className="search-box">
              <span>텍스트 검색어</span>
              <input
                value={query}
                onChange={(event) => {
                  setQuery(event.target.value);
                  clearSearchResults();
                }}
                placeholder="예: 광택감 있는 블랙 아우터와 슬림 팬츠 조합"
                aria-label="텍스트 검색어"
              />
            </label>

            <div className="composer-grid">
              <label className="upload-tile upload-label">
                <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} />
                <p>이미지 업로드</p>
                <span>
                  {uploadedImage
                    ? `${uploadedImage.name} · ${uploadedImage.sizeLabel}`
                    : "착장 사진, 스크린샷, 무드보드 이미지를 올려 보세요."}
                </span>
              </label>

              <div className="context-tile">
                <p>현재 검색 상태</p>
                <span>{helperMessage}</span>
              </div>
            </div>

            {uploadedImage ? (
              <div className="image-preview-card">
                <div className="image-preview-copy">
                  <div>
                    <p className="eyebrow">Selected Image</p>
                    <strong>{uploadedImage.name}</strong>
                  </div>
                  <button type="button" className="mini-button" onClick={clearUploadedImage}>
                    업로드 취소
                  </button>
                </div>
                <div className="image-preview-frame">
                  <img
                    src={uploadedImage.previewUrl}
                    alt={uploadedImage.name}
                    className="image-preview"
                  />
                </div>
              </div>
            ) : null}

            <div className="signal-list">
              <div className="signal-chip">
                <strong>입력 텍스트</strong>
                <span>{query.trim() || "텍스트 없이 이미지 기반 검색만 대기 중입니다."}</span>
              </div>
              <div className="signal-chip">
                <strong>업로드 이미지</strong>
                <span>{uploadedImage ? uploadedImage.name : "아직 업로드된 이미지가 없습니다."}</span>
              </div>
              <div className="signal-chip">
                <strong>실행 모드</strong>
                <span>{modeLabel}</span>
              </div>
            </div>
          </form>
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Search Results</p>
              <h3>{hasSearched ? "검색과 추천을 함께 반영한 결과" : "검색 결과"}</h3>
            </div>
            <div className="heading-metrics">
              <span className="metric">응답 시간 {mergedSearchLatency}</span>
              <span className="metric">결과 수 {mergedSearchResults.length}</span>
            </div>
          </div>

          <div className="recommendation-toolbar">
            <div className="recommendation-actions">
              <div className="topn-group" role="group" aria-label="Top N 검색 결과 개수">
                {[3, 5, 10].map((count) => (
                  <button
                    key={count}
                    type="button"
                    className={topN === count ? "mini-button active" : "mini-button"}
                    onClick={() => {
                      setTopN(count);
                      clearSearchResults();
                    }}
                  >
                    Top {count}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {mergedSearchResults.length === 0 ? (
            <div className="empty-state">
              <p>{searchEmptyMessage}</p>
            </div>
          ) : (
            <div className="result-list">
              {mergedSearchResults.map((item) => (
                <article key={item.id} className="result-card">
                  <ResultVisual imageUrl={item.imageUrl} title={item.title} accent={item.accent} />
                  <div className="result-meta">
                    <div className="result-topline">
                      <p>{item.brand}</p>
                      <strong>{item.price}</strong>
                    </div>
                    <h4>{item.title}</h4>
                    <p>{item.summary}</p>
                    <div className="result-stats">
                      <span className="badge">
                        {mergedSearchScoreLabel} {toDisplayPercent(item.similarity)}
                      </span>
                      <span className="badge">{item.searchType}</span>
                      <span className="badge">응답 {item.responseTime}</span>
                      {hasPersonalizedSearchResults ? (
                        <span className="badge">{searchResultPersona}</span>
                      ) : null}
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Budget Set</p>
              <h3>예산 맞춤 추천</h3>
            </div>
            <div className="heading-metrics">
              <span className="metric">예산 {budgetLabel}</span>
              <span className="metric">세트 수 {budgetSets.setCount}</span>
            </div>
          </div>

          <div className="recommendation-toolbar">
            <div className="recommendation-controls">
              <label className="user-id-field">
                <span>User ID</span>
                <input
                  value={userId}
                  onChange={(event) => setUserId(event.target.value)}
                  placeholder="예: user_1024"
                  aria-label="예산 세트 사용자 ID"
                />
              </label>
              <label className="user-id-field budget-field">
                <span>예산</span>
                <input
                  type="number"
                  min="0"
                  step="1000"
                  value={budget}
                  onChange={(event) => setBudget(event.target.value)}
                  placeholder="예: 200000"
                  aria-label="예산 세트 예산"
                />
              </label>
            </div>
            <div className="recommendation-actions">
              <button
                type="button"
                className="primary-button"
                onClick={loadBudgetSets}
                disabled={isLoadingBudgetSets}
              >
                {isLoadingBudgetSets ? "세트 구성 중..." : "예산 적용하기"}
              </button>
            </div>
          </div>

          {budgetSetError ? <p className="status-text">{budgetSetError}</p> : null}

          {budgetSets.sets.length === 0 ? (
            <p className="status-text">예산을 입력하고 적용하면 검색 결과에 반영됩니다.</p>
          ) : null}

          <div className="recommendation-list">
            {budgetSets.sets.map((setItems, setIndex) => (
              <article key={`set-${setIndex}`} className="panel">
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">Outfit Set</p>
                    <h3>세트 {setIndex + 1}</h3>
                  </div>
                  <div className="heading-metrics">
                    <span className="metric">
                      총액{" "}
                      {setItems
                        .reduce((sum, item) => sum + Number(item.price.replace(/[^0-9]/g, "") || 0), 0)
                        .toLocaleString("ko-KR")}
                      원
                    </span>
                  </div>
                </div>
                <div className="result-list">
                  {setItems.map((item) => (
                    <div key={`${setIndex}-${item.id}`} className="result-card">
                      <ResultVisual imageUrl={item.imageUrl} title={item.title} accent={item.accent} />
                      <div className="result-meta">
                        <div className="result-topline">
                          <p>{item.brand}</p>
                          <strong>{item.price}</strong>
                        </div>
                        <h4>{item.title}</h4>
                        <p>{item.category}</p>
                        <div className="result-stats">
                          <span className="badge">세트 점수 {toDisplayPercent(item.score)}</span>
                          <span className="badge">{item.category}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
