## 패션 도메인 Multimodal 검색 & Multi-Stage 추천 시스템

**FitFind** · 팀명: 사나이들 · 팀장: 손석범

| 이름 | 담당 |
|------|------|
| 손석범 | 프론트엔드, 평가 대시보드 |
| 오승민 | API Gateway, 인프라 |
| 이준원 | 데이터 파이프라인 |
| 장지원 | 추천 모델 (Two-Tower, Ranking, Re-ranking, MAB) |
| 홍찬근 | 검색 엔진 (CLIP + FAISS) |

<br>

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| 검색 | CLIP (`openai/clip-vit-base-patch32`), FAISS HNSW |
| 추천 | Two-Tower, LogReg Ranking, ε-Greedy MAB |
| LLM 연동 | Google Gemini API |
| 서빙 | FastAPI, Redis, Docker Compose |
| 프론트엔드 | React 18, Vite, TypeScript |
| 대시보드 | Streamlit |
| 데이터 | H&M Personalized Fashion Recommendations (Kaggle, 약 105만 고객 / 3150만 거래) |

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 멀티모달 검색 | 텍스트 + 이미지 동시 검색 (CLIP + FAISS HNSW) |
| 개인화 추천 | Two-Tower 후보 생성 → LogReg 랭킹 → MAB 탐색 |
| 페르소나 온보딩 | 9가지 쇼핑 성향 분류 후 Redis 저장, 추천에 반영 |
| 예산 기반 세트 추천 | 예산 내 겹치는 부위 없는 코디 세트 (상의·하의·아우터·신발·액세서리 등) |
| 실시간 세션 반영 | 클릭/장바구니 이벤트 → Redis → 즉시 추천 반영 |
| 평가 대시보드 | 검색 품질 지표, 추천 성능, A/B 테스트 결과 시각화 |


---

## 시스템 아키텍처

```
사용자 (Browser)
       │
       ▼
┌─────────────────────┐
│   Frontend :3000    │  React + Vite + TypeScript
└────────┬────────────┘
         │ HTTP
         ▼
┌─────────────────────────────────────────┐
│         API Gateway :8000               │
│  FastAPI — 단일 진입점                   │
│                                         │
│  POST /api/search          검색          │
│  GET  /api/recommend       개인화 추천   │
│  POST /api/set-recommend   세트 추천     │
│  POST /api/onboarding      페르소나 설정 │
│  POST /api/events          이벤트 기록   │
└────────┬────────────────────┬───────────┘
         │                    │
         ▼                    ▼
┌────────────────┐   ┌────────────────────┐
│ Search Engine  │   │    Rec-Models      │
│    :8002       │   │      :8003         │
│                │   │                    │
│  CLIP 임베딩   │   │  Two-Tower 후보    │
│  FAISS HNSW    │   │  → LogReg 랭킹     │
│  텍스트/이미지  │   │  → Re-ranking      │
│  멀티모달 검색 │   │  → ε-Greedy MAB   │
└────────────────┘   └──────────┬─────────┘
                                │
                     ┌──────────▼─────────┐
                     │    Redis :6379      │
                     │   Feature Store     │
                     │  - recent_clicks    │
                     │  - session_interest │
                     │  - persona profile  │
                     └─────────────────────┘

┌─────────────────────┐   ┌──────────────────────┐
│  Dashboard :8501    │   │  Simulator           │
│  Streamlit          │   │  행동 로그 자동 생성   │
│  - 검색 품질 지표   │   │  - search/view/cart  │
│  - 추천 성능 지표   │   │  - purchase events   │
│  - A/B 테스트 결과  │   └──────────────────────┘
└─────────────────────┘

┌──────────────────────┐
│  CT Pipeline         │
│  성능 모니터링 &      │
│  자동 재학습 트리거   │
└──────────────────────┘
```

<br>

| 서비스 | 주소 |
|--------|------|
| 프론트엔드 | http://localhost:3000 |
| API Gateway | http://localhost:8000 |
| Search Engine | http://localhost:8002 |
| Rec-Models | http://localhost:8003 |
| 평가 대시보드 | http://localhost:8501 |

---

<details>
<summary><b>실행 방법</b></summary>

<br>

### 사전 요구사항

- Docker Desktop (Docker Compose 포함)
- RAM 16GB 이상 권장
- [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) 데이터셋 (Kaggle)
- Google AI Studio에서 발급한 Gemini API 키

<br>

### 1단계 — 데이터셋 배치

Kaggle에서 다운로드한 파일을 다음 경로에 배치합니다.

```
data/raw/
├── articles.csv
├── customers.csv
└── transactions_train.csv
```

### 2단계 — 환경 변수 설정

`.env.example`을 복사해 `.env`를 생성하고 Gemini API 키를 입력합니다.

```bash
cp .env.example .env
```

### 3단계 — 데이터 파이프라인 실행 (최초 1회)

`dev` 모드로 실행되며 `data/processed/` 아래 전처리 파일 전체를 생성합니다. 소요 시간 약 45~60분.

```bash
docker compose run --rm data-pipeline
```

### 4단계 — 모델 학습 (최초 1회)

학습된 체크포인트는 `data/checkpoints/` 아래에 저장됩니다.

```bash
# Two-Tower 후보 모델
docker compose run --rm rec-models python rec_models/candidate/train_two_tower.py

# Ranking 모델
docker compose run --rm rec-models python rec_models/ranking/train_ranking.py
```

### 5단계 — 전체 서비스 실행

```bash
docker compose up
```

</details>

---

<details>
<summary><b>API 사용 예시</b></summary>

<br>

### 텍스트 검색

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "검정 오버핏 후드티", "top_k": 10}'
```

### 개인화 추천

```bash
curl "http://localhost:8000/api/recommend?user_id=U1234&top_n=10"
```

### 예산 기반 세트 추천

```bash
curl -X POST http://localhost:8000/api/set-recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "U1234", "budget": 150000, "num_sets": 3}'
```

### 이벤트 기록

```bash
curl -X POST http://localhost:8000/api/events \
  -H "Content-Type: application/json" \
  -d '{"user_id": "U1234", "item_id": "0706016001", "event_type": "click", "category": "상의"}'
```

</details>
