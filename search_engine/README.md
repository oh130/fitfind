# Search Engine

이 디렉토리는 멀티모달 검색 엔진 서비스 구현을 포함한다.  
검색 엔진은 OpenAI CLIP(`openai/clip-vit-base-patch32`)으로 텍스트/이미지를 임베딩하고, FAISS HNSW 인덱스를 통해 Top-K 유사 상품을 반환한다.

## 구현 범위

- 텍스트 검색: CLIP Text Encoder 사용
- 이미지 검색: CLIP Image Encoder 사용
- 하이브리드 검색: 텍스트+이미지 임베딩 평균 결합
- 벡터 검색: FAISS HNSW
- 응답 형식: 프로젝트 명세의 검색 API 필수 필드 준수

## 주요 파일

- [app.py](/C:/Users/user/multimodal-search/search_engine/app.py)
  검색 API 서버. `/search`, `/cross-similarity`, `/health` 엔드포인트를 제공한다.
- [search_engine.py](/C:/Users/user/multimodal-search/search_engine/search_engine.py)
  CLIP 임베딩, 상품 인덱싱, FAISS 검색 로직을 포함한다.
- [generate_search_metrics_report.py](/C:/Users/user/multimodal-search/search_engine/generate_search_metrics_report.py)
  검색 성능 평가셋 생성, API 호출, MRR/NDCG/지연시간 측정, JSON 리포트 저장.
- [evaluate_search_engine.py](/C:/Users/user/multimodal-search/search_engine/evaluate_search_engine.py)
  평가셋 CSV를 입력으로 받아 검색 API를 측정하는 범용 평가 스크립트.

## 실행 모드

검색 엔진은 두 가지 모드를 지원한다.

- `test`
  500개 샘플 기반 인덱스 또는 더미 상품으로 빠르게 실행한다.
- `production`
  H&M 전처리 결과를 기반으로 전체 상품 인덱스를 구성한다.

현재 컨테이너 실행 시 모드는 `SEARCH_ENGINE_MODE` 환경변수로 제어된다.

## 입력 데이터

검색 엔진은 다음 전처리 결과를 사용한다.

- 원본 데이터:
  - `data/raw/articles.csv`
  - `data/raw/customers.csv`
  - `data/raw/transactions_train.csv`
- 검색용 전처리 결과:
  - `data/processed/articles_feature.csv`
- 인덱스 캐시:
  - `data/faiss_index/search_test.index`
  - `data/faiss_index/search_test_metadata.json`
  - `data/faiss_index/search.index`
  - `data/faiss_index/search_metadata.json`

## API 규격

- 포트: `8002`
- 엔드포인트:
  - `POST /search`
  - `POST /cross-similarity`
  - `GET /health`

요청 예시:

```json
{
  "query": "blue jacket",
  "image_base64": null,
  "top_k": 10
}
```

응답 필수 필드:

```json
{
  "search_type": "text",
  "results": [
    {
      "product_id": "0825137001",
      "name": "SABLE denim jacket",
      "score": 0.794,
      "price": 0.0
    }
  ],
  "latency_ms": 42.0,
  "total_count": 10
}
```

`search_type`은 다음 중 하나다.

- `text`
- `image`
- `hybrid`

### `POST /cross-similarity`

추천 API Gateway의 예산 기반 세트 구성에서 사용한다. 요청한 상품 ID를 검색 인덱스의 상품 임베딩으로 변환한 뒤, 상품 간 cosine similarity 행렬을 반환한다.

요청 예시:

```json
{
  "article_ids": ["0825137001", "0717490032", "0673677002"]
}
```

응답 예시:

```json
{
  "similarity": {
    "0825137001": {
      "0717490032": 0.731245
    }
  },
  "article_ids": ["0825137001", "0717490032"],
  "missing_article_ids": ["0673677002"],
  "latency_ms": 4.21,
  "total_count": 2
}
```

## 로컬 실행

### 1. 패키지 설치

```powershell
pip install -r .\search_engine\requirements.txt
```

### 2. 검색 엔진 서버 실행

```powershell
$env:SEARCH_ENGINE_MODE="test"
python .\search_engine\app.py
```

기본 확인:

```powershell
Invoke-WebRequest -Uri http://localhost:8002/health -UseBasicParsing | Select-Object -Expand Content
```

### 3. 검색 API 호출 예시

```powershell
$body = @{
  query = "blue jacket"
  image_base64 = $null
  top_k = 10
} | ConvertTo-Json

Invoke-WebRequest `
  -Uri http://localhost:8002/search `
  -Method POST `
  -ContentType "application/json" `
  -Body $body `
  -UseBasicParsing | Select-Object -Expand Content
```

## Docker 실행

프로젝트 루트에서:

```powershell
docker-compose up --build
```

접속 주소:

- 검색 엔진: [http://localhost:8002](http://localhost:8002)
- Swagger UI: [http://localhost:8002/docs](http://localhost:8002/docs)
- API Gateway: [http://localhost:8000](http://localhost:8000)
- 평가 대시보드: [http://localhost:8501](http://localhost:8501)

## 검색 성능 지표 측정

검색 성능은 다음 항목을 측정한다.

- `HitRate@10`
- `MRR`
- `NDCG@10`
- 평균 API latency
- 평균 wall latency
- P95 wall latency

명세 기준 목표:

- 검색 응답 시간 `<= 200ms`
- `MRR >= 0.55`
- `NDCG@10 >= 0.50`

### 방법 1. 평가 리포트 생성

검색 엔진 서버가 `localhost:8002`에서 실행 중일 때:

```powershell
python .\search_engine\generate_search_metrics_report.py --endpoint http://localhost:8002/search
```

생성 결과:

- 평가셋 CSV: [search_eval_set.csv](/C:/Users/user/multimodal-search/evaluation/search_eval_set.csv)
- 검색 리포트 JSON: [search_metrics_report.json](/C:/Users/user/multimodal-search/evaluation/search_metrics_report.json)

콘솔에는 다음 정보가 출력된다.

- 샘플 수
- `HitRate@10`
- `MRR`
- `NDCG@10`
- 평균 지연시간
- 목표 통과 여부

### 방법 2. 평가 CSV를 이용한 재측정

이미 생성된 평가셋 CSV를 이용해서 다시 측정하려면:

```powershell
python .\search_engine\evaluate_search_engine.py --endpoint http://localhost:8002/search
```

기본 입력 파일은 `evaluation/search_eval_set.csv`다.

## 대시보드에서 검색 성능 확인

검색 성능 리포트를 생성한 뒤:

1. `docker-compose up --build`로 대시보드를 실행한다.
2. 브라우저에서 [http://localhost:8501](http://localhost:8501)에 접속한다.
3. 상단의 `Search Engine Quality` 섹션에서 다음을 확인한다.

- Samples
- MRR
- nDCG@10
- Avg latency
- P95 latency
- 목표 통과 여부 표

대시보드는 호스트의 `evaluation/search_metrics_report.json`을 직접 읽도록 연결되어 있으므로, 리포트를 새로 생성한 뒤 페이지를 새로고침하면 최신 결과가 반영된다.

## 최근 측정 예시

최근 생성된 검색 리포트 기준:

- `Samples evaluated`: `101`
- `MRR`: `0.7934`
- `NDCG@10`: `0.7161`
- `Avg wall latency`: 약 `33.20 ms`
- `P95 wall latency`: 약 `50.27 ms`

목표 달성 여부:

- `Latency <= 200ms`: PASS
- `MRR >= 0.55`: PASS
- `NDCG@10 >= 0.50`: PASS

## 현재 구현 메모

- `search_engine.py`는 self-contained하게 CLIP 로딩, 임베딩, FAISS 인덱싱, 검색까지 수행한다.
- `app.py`는 서비스 레벨에서 인덱스 캐시를 활용해 기동 시간을 줄인다.
- 테스트/평가 스크립트는 검색 엔진 코드를 크게 수정하지 않고 별도 파일로 분리했다.

## 참고

- 전체 시스템 아키텍처는 루트 [README.md](/C:/Users/user/multimodal-search/README.md)에 정리되어 있다.
- 추천 성능 및 A/B 테스트 시각화는 `evaluation/streamlit_app.py`에서 함께 제공된다.
