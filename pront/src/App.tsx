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

const flowSteps: Array<{ key: AppView; label: string; title: string }> = [
  { key: "landing", label: "01", title: "시작" },
  { key: "onboarding", label: "02", title: "취향 설정" },
  { key: "search", label: "03", title: "상품 검색" },
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

function parseWonAmount(value: string | number | undefined): number {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : 0;
  }
  return Number(String(value ?? "").replace(/[^0-9]/g, "") || 0);
}

function formatWonAmount(value: number): string {
  return `${Math.max(0, Math.round(value)).toLocaleString("ko-KR")}원`;
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
  const mergedSearchResults =
    activeSearchResults.length > 0
      ? activeSearchResults
      : hasPersonalizedSearchResults
        ? personalizedResults
        : results;
  const mergedSearchLatency =
    activeSearchResults.length > 0
      ? activeSearchLatency
      : hasPersonalizedSearchResults
        ? personalizedLatency
        : activeLatency;
  const mergedSearchScoreLabel =
    activeSearchResults.length > 0
      ? activeSearchScoreLabel
      : hasPersonalizedSearchResults
        ? "추천 점수"
        : "유사도";
  const selectedAudienceLabel =
    targetAudienceOptions.find((option) => option.key === targetAudience)?.label ?? "전체";
  const parsedBudget = parseWonAmount(budget);
  const searchEmptyMessage = !hasSearched
    ? "찾고 싶은 스타일을 입력하면 어울리는 상품을 골라 보여드릴게요."
    : searchError
      ? searchError
      : searchResultView === "personalized"
        ? "취향에 맞는 결과가 아직 없습니다. 검색어를 조금 더 구체적으로 바꿔 보세요."
        : "검색 결과가 없습니다. 이미지나 색상, 카테고리를 함께 입력해 보세요.";

  if (view === "landing") {
    return (
      <div className="studio-shell flow-shell">
        <aside className="studio-sidebar flow-sidebar">
          <div className="studio-brand">
            <div>
              <span>FitFind</span>
              <strong>AI 패션 탐색</strong>
            </div>
            <button type="button" onClick={goToOnboarding}>
              시작
            </button>
          </div>

          <div className="flow-steps" aria-label="서비스 진행 단계">
            {flowSteps.map((step) => (
              <div key={step.key} className={step.key === "landing" ? "flow-step active" : "flow-step"}>
                <span>{step.label}</span>
                <strong>{step.title}</strong>
              </div>
            ))}
          </div>
        </aside>

        <main className="flow-main">
          <section className="flow-hero">
            <p className="studio-kicker">Capstone Demo</p>
            <h1>텍스트와 이미지로 찾고, 취향과 예산으로 좁힙니다.</h1>
            <p>
              FitFind는 멀티모달 검색, 페르소나 기반 개인화, 예산 코디 추천을 하나의 흐름으로
              연결한 패션 탐색 서비스입니다.
            </p>
            <div className="flow-action-row">
              <button type="button" className="studio-primary-action" onClick={goToOnboarding}>
                취향 설정 시작
              </button>
              <button type="button" className="studio-secondary-action" onClick={() => setView("search")}>
                바로 검색하기
              </button>
            </div>
          </section>

          <section className="flow-feature-grid">
            <article>
              <span>Search</span>
              <strong>텍스트 + 이미지</strong>
              <p>문장과 이미지 신호를 함께 사용해 유사 상품을 찾습니다.</p>
            </article>
            <article>
              <span>Persona</span>
              <strong>취향 반영</strong>
              <p>온보딩 결과와 세션 행동을 추천 결과에 반영합니다.</p>
            </article>
            <article>
              <span>Budget</span>
              <strong>코디 세트</strong>
              <p>예산 안에서 실제 착용 가능한 아이템 조합을 구성합니다.</p>
            </article>
          </section>
        </main>
      </div>
    );
  }

  if (view === "onboarding") {
    return (
      <div className="studio-shell flow-shell">
        <aside className="studio-sidebar flow-sidebar">
          <div className="studio-brand">
            <div>
              <span>FitFind</span>
              <strong>{userId.trim() || "anonymous"}</strong>
            </div>
            <button type="button" onClick={() => setView("landing")}>
              처음
            </button>
          </div>

          <div className="flow-steps" aria-label="서비스 진행 단계">
            {flowSteps.map((step) => (
              <div key={step.key} className={step.key === "onboarding" ? "flow-step active" : "flow-step"}>
                <span>{step.label}</span>
                <strong>{step.title}</strong>
              </div>
            ))}
          </div>
        </aside>

        <main className="flow-main onboarding-studio-main">
          <section className="setup-board">
            <header className="setup-header">
              <div>
                <span className="studio-kicker">Persona Setup</span>
                <h1>취향을 먼저 잡고 검색으로 넘어갑니다.</h1>
              </div>
              <div className="setup-total">
                <span>총합</span>
                <strong>{personaScoreTotal}%</strong>
              </div>
            </header>

            <div className="setup-content">
              <div className="setup-form">
                <div className="setup-segmented" role="group" aria-label="쇼핑 대상 선택">
                  {targetAudienceOptions.map((option) => (
                    <button
                      key={option.key}
                      type="button"
                      className={targetAudience === option.key ? "active" : ""}
                      onClick={() => setTargetAudience(option.key)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>

                <label>
                  <span>User ID</span>
                  <input
                    value={userId}
                    onChange={(event) => setUserId(event.target.value)}
                    placeholder="user_1024"
                    aria-label="온보딩 사용자 ID"
                  />
                </label>

                <label>
                  <span>Style note</span>
                  <input
                    value={onboardingDescription}
                    onChange={(event) => setOnboardingDescription(event.target.value)}
                    placeholder="미니멀한 블랙 아우터와 실용적인 출근룩"
                    aria-label="온보딩 취향 입력"
                  />
                </label>

                <div className="style-chip-grid">
                  {onboardingStyleOptions.map((style) => (
                    <button
                      key={style}
                      type="button"
                      className={selectedStyles.includes(style) ? "active" : ""}
                      onClick={() => toggleStyleChoice(style)}
                    >
                      {style}
                    </button>
                  ))}
                </div>

                <div className="setup-actions">
                  <button type="button" onClick={() => setView("landing")}>
                    이전
                  </button>
                  <button type="button" className="primary" onClick={runOnboardingAnalysis} disabled={isAnalyzingOnboarding}>
                    {isAnalyzingOnboarding ? "분석 중" : "취향 분석"}
                  </button>
                </div>

                {onboardingError ? <p className="studio-alert light">{onboardingError}</p> : null}
              </div>

              <div className="setup-guide">
                <span>Next</span>
                <strong>분석 결과를 확인한 뒤 비중을 조절하세요.</strong>
                <p>
                  합계가 100%가 되면 개인화 검색에 사용할 페르소나 프로필로 저장할 수 있습니다.
                </p>
              </div>
            </div>
          </section>

          <section className="persona-board">
            <header className="studio-board-header compact">
              <div>
                <span className="studio-kicker">Persona Mix</span>
                <h2>{Object.keys(personaScores).length > 0 ? "분석된 취향 비중" : "분석 대기 중"}</h2>
              </div>
              <button
                type="button"
                className="studio-primary-action small"
                onClick={startWithPersona}
                disabled={
                  isSubmittingPersona || Object.keys(personaScores).length === 0 || !isPersonaScoreTotalValid
                }
              >
                {isSubmittingPersona ? "저장 중" : "검색으로 이동"}
              </button>
            </header>

            {Object.keys(personaScores).length > 0 ? (
              <>
                <div className="persona-studio-grid">
                  {personaOptions.map((persona) => (
                    <article
                      key={persona.key}
                      className={
                        persona.key === selectedOnboardingPersona ? "persona-studio-card active" : "persona-studio-card"
                      }
                    >
                      <div className="persona-studio-card-head">
                        <span>{persona.name}</span>
                        <strong>{personaScores[persona.key] ?? 0}%</strong>
                      </div>
                      <h3>{persona.title}</h3>
                      <p>{persona.summary}</p>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={personaScores[persona.key] ?? 0}
                        onChange={(event) => updatePersonaScore(persona.key, Number(event.target.value))}
                        aria-label={`${persona.name} 비율 조절`}
                      />
                      <div>
                        {persona.traits.map((trait) => (
                          <span key={trait}>{trait}</span>
                        ))}
                      </div>
                    </article>
                  ))}
                </div>
                <p className="persona-adjustment-note studio-note">
                  {!isPersonaScoreTotalValid
                    ? "전체 합계가 100%가 되어야 검색 페이지로 이동할 수 있습니다."
                    : "설정이 준비됐습니다. 검색 페이지로 이동해 개인화 결과를 확인하세요."}
                </p>
              </>
            ) : (
              <div className="studio-empty compact">
                <p>취향 설명을 입력하고 분석을 실행하세요.</p>
              </div>
            )}
          </section>
        </main>
      </div>
    );
  }

  return (
    <div className="studio-shell">
      <aside className="studio-sidebar">
        <div className="studio-brand">
          <div>
            <span>FitFind</span>
            <strong>{userId.trim() || "anonymous"}</strong>
          </div>
          <button type="button" onClick={() => setView("onboarding")}>
            취향 설정
          </button>
        </div>

        <div className="flow-steps compact" aria-label="서비스 진행 단계">
          {flowSteps.map((step) => (
            <div key={step.key} className={step.key === "search" ? "flow-step active" : "flow-step"}>
              <span>{step.label}</span>
              <strong>{step.title}</strong>
            </div>
          ))}
        </div>

        <form id="search-composer-form" className="studio-form" onSubmit={handleSubmit}>
          <label className="studio-query">
            <span>Query</span>
            <input
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                clearSearchResults();
              }}
              placeholder="블랙 아우터와 슬림 팬츠"
              aria-label="텍스트 검색어"
            />
          </label>

          <div className="studio-segmented" role="group" aria-label="쇼핑 대상 선택">
            {targetAudienceOptions.map((option) => (
              <button
                key={option.key}
                type="button"
                className={targetAudience === option.key ? "active" : ""}
                onClick={() => {
                  setTargetAudience(option.key);
                  clearSearchResults();
                }}
              >
                {option.label}
              </button>
            ))}
          </div>

          <label className={uploadedImage ? "studio-upload has-image" : "studio-upload"}>
            <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} />
            {uploadedImage ? (
              <>
                <img src={uploadedImage.previewUrl} alt={uploadedImage.name} />
                <span>{uploadedImage.name}</span>
              </>
            ) : (
              <>
                <strong>이미지 추가</strong>
                <span>사진 없이 검색 가능</span>
              </>
            )}
          </label>
          {uploadedImage ? (
            <button type="button" className="studio-link-button" onClick={clearUploadedImage}>
              이미지 제거
            </button>
          ) : null}

          <div className="studio-field-grid">
            <label>
              <span>User</span>
              <input
                value={userId}
                onChange={(event) => setUserId(event.target.value)}
                placeholder="user_1024"
                aria-label="사용자 ID"
              />
            </label>
            <label>
              <span>Budget</span>
              <input
                type="number"
                min="0"
                step="1000"
                value={budget}
                onChange={(event) => setBudget(event.target.value)}
                placeholder="200000"
                aria-label="예산"
              />
            </label>
          </div>

          <div className="studio-range">
            <div>
              <span>Preference</span>
              <strong>{Math.round(recommendationWeight * 100)}%</strong>
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

          <div className="studio-count-row" role="group" aria-label="검색 결과 개수">
            {[3, 5, 10].map((count) => (
              <button
                key={count}
                type="button"
                className={topN === count ? "active" : ""}
                onClick={() => {
                  setTopN(count);
                  clearSearchResults();
                }}
              >
                {count}
              </button>
            ))}
          </div>

          <div className="studio-actions">
            <button type="submit" className="studio-primary" disabled={isSearching}>
              {isSearching ? "검색 중" : "검색"}
            </button>
            <button type="button" onClick={loadBudgetSets} disabled={isLoadingBudgetSets}>
              {isLoadingBudgetSets ? "구성 중" : "코디"}
            </button>
            <button type="button" onClick={loadAiRecommendations} disabled={isRefreshingRecommendations}>
              {isRefreshingRecommendations ? "생성 중" : "이유"}
            </button>
          </div>

          {recommendationError ? <p className="studio-alert">{recommendationError}</p> : null}
          {budgetSetError ? <p className="studio-alert">{budgetSetError}</p> : null}

          <details className="studio-debug">
            <summary>debug</summary>
            <span>{helperMessage}</span>
            <span>{modeLabel}</span>
            <span>{mergedSearchLatency}</span>
          </details>
        </form>
      </aside>

      <main className="studio-main">
        <section className="studio-board">
          <header className="studio-board-header">
            <div>
              <span className="studio-kicker">Results</span>
              <h1>{hasSearched ? query || "이미지 검색" : "새 검색"}</h1>
            </div>
            <div className="studio-board-tools">
              <span>{selectedAudienceLabel}</span>
              <span>{mergedSearchResults.length} items</span>
              <span>{budgetLabel}</span>
            </div>
          </header>

          <div className="studio-tabs" role="group" aria-label="결과 정렬">
            <button
              type="button"
              className={searchResultView === "personalized" ? "active" : ""}
              onClick={() => setSearchResultView("personalized")}
            >
              취향순
            </button>
            <button
              type="button"
              className={searchResultView === "similarity" ? "active" : ""}
              onClick={() => setSearchResultView("similarity")}
            >
              유사도순
            </button>
          </div>

          {mergedSearchResults.length === 0 ? (
            <div className="studio-empty">
              <p>{searchEmptyMessage}</p>
            </div>
          ) : (
            <div className="studio-product-grid">
              {mergedSearchResults.map((item) => (
                <article key={item.id} className="studio-product-card">
                  <ResultVisual imageUrl={item.imageUrl} title={item.title} accent={item.accent} />
                  <div className="studio-product-copy">
                    <div>
                      <span>{item.brand}</span>
                      <strong>{item.price}</strong>
                    </div>
                    <h2>{item.title}</h2>
                    <p>{item.summary}</p>
                    <footer>
                      <span>{mergedSearchScoreLabel} {toDisplayPercent(item.similarity)}</span>
                      <span>{item.searchType}</span>
                      {hasPersonalizedSearchResults ? <span>{searchResultPersona}</span> : null}
                    </footer>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className="studio-outfit-board">
          <header className="studio-board-header compact">
            <div>
              <span className="studio-kicker">Outfits</span>
              <h2>예산 코디</h2>
            </div>
            <span className="studio-budget-pill">{budgetSets.setCount} sets</span>
          </header>

          {budgetSets.sets.length === 0 ? (
            <div className="studio-empty compact">
              <p>코디 결과 없음</p>
            </div>
          ) : (
            <div className="studio-outfit-list">
              {budgetSets.sets.map((setItems, setIndex) => {
                const totalPrice =
                  setItems[0]?.setTotalPrice ??
                  setItems.reduce((sum, item) => sum + parseWonAmount(item.price), 0);
                const budgetUsage = parsedBudget > 0 ? Math.min(100, (totalPrice / parsedBudget) * 100) : 0;

                return (
                  <article key={`set-${setIndex}`} className="studio-outfit-card">
                    <div className="studio-outfit-head">
                      <div>
                        <span>Set {setIndex + 1}</span>
                        <strong>{formatWonAmount(totalPrice)}</strong>
                      </div>
                      <em>{toDisplayPercent(setItems[0]?.setScore ?? 0)}</em>
                    </div>
                    <div className="studio-meter" aria-label={`예산 사용률 ${budgetUsage.toFixed(0)}%`}>
                      <span style={{ width: `${budgetUsage}%` }} />
                    </div>
                    <div className="studio-outfit-items">
                      {setItems.map((item) => (
                        <div key={`${setIndex}-${item.id}`} className="studio-outfit-item">
                          <ResultVisual imageUrl={item.imageUrl} title={item.title} accent={item.accent} />
                          <div>
                            <span>{item.category}</span>
                            <strong>{item.title}</strong>
                            <p>{item.price}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
