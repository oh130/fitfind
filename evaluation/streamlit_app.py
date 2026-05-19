from __future__ import annotations

import json
from pathlib import Path
from urllib import error, request

import altair as alt
import pandas as pd
import streamlit as st

SEARCH_REPORT_PATH = Path(__file__).resolve().with_name("search_metrics_report.json")
RECOMMENDATION_REPORT_PATH = (
    Path(__file__).resolve().parents[1] / "rec_models" / "reports" / "baseline" / "baseline_metrics.json"
)
RANKING_REPORTS_DIR = Path(__file__).resolve().parents[1] / "rec_models" / "reports" / "ranking_experiments"
BASELINE_REPORT_PATH = Path(__file__).resolve().parents[1] / "rec_models" / "reports" / "baseline" / "baseline_metrics.json"
PERSONA_AB_REPORT_PATH = (
    Path(__file__).resolve().parents[1] / "rec_models" / "reports" / "baseline" / "dev_spec_e2e_persona_optimized.json"
)
COVERAGE_AB_REPORT_PATH = (
    Path(__file__).resolve().parents[1] / "rec_models" / "reports" / "baseline" / "dev_e2e_twotower_serving_coverage_strong.json"
)
ONBOARDING_API_URL = "http://127.0.0.1:8000/api/onboarding"
RECOMMEND_API_URL = "http://127.0.0.1:8000/api/recommend"
BUDGET_SET_API_URL = "http://127.0.0.1:8000/api/budget-set"

PERSONA_LABELS = {
    "trendsetter": "트렌드세터형",
    "practical": "실용주의형",
    "value": "가성비추구형",
    "brand_loyal": "브랜드충성형",
    "impulse": "충동구매형",
    "careful": "신중탐색형",
    "repeat_stable": "반복구매형",
    "color_focus": "색상집중형",
    "category_focus": "카테고리집중형",
}

DEFAULT_PERSONA_SCORES = {
    "trendsetter": 28,
    "practical": 16,
    "value": 14,
    "brand_loyal": 7,
    "impulse": 6,
    "careful": 12,
    "repeat_stable": 5,
    "color_focus": 8,
    "category_focus": 4,
}

DEFAULT_ONBOARDING_RESPONSE = {
    "persona_scores": DEFAULT_PERSONA_SCORES,
}

DEFAULT_RECOMMENDATION_RESPONSE = {
    "user_id": "user_1024",
    "persona": "trendsetter",
    "recommendations": [
        {
            "product_id": "0825137001",
            "name": "Urban Edge Rider Jacket",
            "brand": "Mode Atelier",
            "category": "Outer",
            "price": 89000,
            "score": 0.94,
            "rank": 1,
            "reason": "ranking_score",
            "reason_text": "최근 탐색한 블랙 아우터 취향과 가장 가깝고, 실버 포인트 디테일이 잘 맞습니다.",
        },
        {
            "product_id": "0921184002",
            "name": "Minimal Zip Blouson",
            "brand": "Noir Form",
            "category": "Top",
            "price": 42000,
            "score": 0.9,
            "rank": 2,
            "reason": "session_interest_match",
            "reason_text": "미니멀한 출근룩 수요와 예산 범위를 함께 만족하는 안정적인 후보입니다.",
        },
        {
            "product_id": "0754401005",
            "name": "Chrome Detail Urban Rider",
            "brand": "Modu Lab",
            "category": "Accessory",
            "price": 58000,
            "score": 0.87,
            "rank": 3,
            "reason": "mab_exploration",
            "reason_text": "현재 취향과 유사하면서도 새로운 조합을 탐색하기 위한 실험 슬롯 상품입니다.",
        },
    ],
    "pipeline_latency": {
        "candidate_ms": 48,
        "ranking_ms": 61,
        "reranking_ms": 18,
        "total_ms": 127,
    },
}

DEFAULT_BUDGET_SET_RESPONSE = {
    "budget": 200000,
    "set_count": 2,
    "sets": [
        [
            {
                "article_id": "0825137001",
                "name": "Urban Edge Rider Jacket",
                "brand": "Mode Atelier",
                "category": "Outer",
                "price_int": 89000,
                "score": 0.94,
            },
            {
                "article_id": "0921184002",
                "name": "Minimal Zip Blouson",
                "brand": "Noir Form",
                "category": "Top",
                "price_int": 42000,
                "score": 0.9,
            },
            {
                "article_id": "0754401005",
                "name": "Chrome Detail Urban Rider",
                "brand": "Modu Lab",
                "category": "Accessory",
                "price_int": 58000,
                "score": 0.87,
            },
        ],
        [
            {
                "article_id": "0861123007",
                "name": "Blackline Cropped Moto",
                "brand": "Noir Craft",
                "category": "Outer",
                "price_int": 71000,
                "score": 0.91,
            },
            {
                "article_id": "0738829004",
                "name": "Gloss Rider Short",
                "brand": "Studio Hex",
                "category": "Bottom",
                "price_int": 58000,
                "score": 0.84,
            },
            {
                "article_id": "0910022008",
                "name": "Silver Trim Moto Crop",
                "brand": "Avenue N",
                "category": "Top",
                "price_int": 76000,
                "score": 0.88,
            },
        ],
    ],
}


def discover_ranking_reports() -> list[Path]:
    return sorted(RANKING_REPORTS_DIR.glob("*.json"))


def load_ranking_report(report_path: Path) -> dict:
    with report_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_json_report(report_path: Path) -> dict:
    with report_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_recommendation_variant(report: dict) -> tuple[dict, dict, dict]:
    if "recommendation" in report:
        metadata = report.get("metadata", {})
        current_model = report.get("recommendation", {}).get("current_model", {})
        ranking = report.get("ranking", {})
    else:
        experiment = report.get("experiment", {})
        metadata = {
            "generated_at_utc": experiment.get("generated_at_utc"),
            "source_data": experiment.get("data_path"),
            "top_k": experiment.get("config", {}).get("top_k"),
            "max_users": experiment.get("config", {}).get("max_users"),
        }
        current_model = report.get("metrics", {}).get("current_model", {})
        ranking = {}
    return metadata, current_model, ranking


def build_ab_comparison_rows(
    baseline_report: dict,
    variant_report: dict,
    variant_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_meta, baseline_metrics, baseline_ranking = extract_recommendation_variant(baseline_report)
    variant_meta, variant_metrics, variant_ranking = extract_recommendation_variant(variant_report)
    top_k = int((variant_meta.get("top_k") or baseline_meta.get("top_k") or 50))

    comparison_rows = [
        {
            "metric": "HitRate",
            "label": f"HitRate@{top_k}",
            "baseline": float(baseline_metrics.get(f"HitRate@{top_k}", 0.0)),
            "variant": float(variant_metrics.get(f"HitRate@{top_k}", 0.0)),
        },
        {
            "metric": "NDCG",
            "label": f"NDCG@{top_k}",
            "baseline": float(baseline_metrics.get(f"NDCG@{top_k}", 0.0)),
            "variant": float(variant_metrics.get(f"NDCG@{top_k}", 0.0)),
        },
        {
            "metric": "Coverage",
            "label": f"Coverage@{top_k}",
            "baseline": float(baseline_metrics.get(f"Coverage@{top_k}", 0.0)),
            "variant": float(variant_metrics.get(f"Coverage@{top_k}", 0.0)),
        },
    ]
    if baseline_ranking or variant_ranking:
        comparison_rows.append(
            {
                "metric": "Ranking AUC",
                "label": "Ranking AUC",
                "baseline": float(baseline_ranking.get("auc", 0.0)),
                "variant": float(variant_ranking.get("auc", baseline_ranking.get("auc", 0.0))),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df["absolute_diff"] = comparison_df["variant"] - comparison_df["baseline"]
    comparison_df["relative_lift"] = comparison_df.apply(
        lambda row: 0.0 if row["baseline"] == 0 else row["absolute_diff"] / row["baseline"],
        axis=1,
    )

    meta_df = pd.DataFrame(
        [
            {"field": "baseline_source", "value": baseline_meta.get("source_data", "-")},
            {"field": "variant_source", "value": variant_meta.get("source_data", "-")},
            {"field": "baseline_generated_at_utc", "value": baseline_meta.get("generated_at_utc", "-")},
            {"field": "variant_generated_at_utc", "value": variant_meta.get("generated_at_utc", "-")},
            {"field": "top_k", "value": top_k},
            {"field": "baseline_users", "value": baseline_metrics.get("users_evaluated", "-")},
            {"field": "variant_users", "value": variant_metrics.get("users_evaluated", "-")},
            {"field": "variant_label", "value": variant_label},
        ]
    )
    return comparison_df, meta_df


def fetch_onboarding_persona_scores(
    *,
    user_id: str,
    description: str,
    style_choices: list[str],
    budget_range: str | None = None,
) -> dict[str, float]:
    payload = json.dumps(
        {
            "user_id": user_id,
            "description": description,
            "style_choices": style_choices,
            "budget_range": budget_range,
        }
    ).encode("utf-8")
    req = request.Request(
        ONBOARDING_API_URL,
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=20) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    raw_scores = parsed.get("persona_scores", {})
    if not isinstance(raw_scores, dict):
        return {}
    return {str(key): float(value) for key, value in raw_scores.items()}


def fetch_recommendation_response(
    *,
    user_id: str,
    top_n: int = 3,
    include_reasons: bool = True,
) -> dict:
    from urllib.parse import urlencode

    query = urlencode(
        {
            "user_id": user_id,
            "top_n": top_n,
            "include_reasons": str(include_reasons).lower(),
        }
    )
    req = request.Request(
        f"{RECOMMEND_API_URL}?{query}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with request.urlopen(req, timeout=20) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        return DEFAULT_RECOMMENDATION_RESPONSE
    return parsed


def fetch_budget_set_response(
    *,
    user_id: str,
    budget: int,
    set_count: int = 2,
) -> dict:
    from urllib.parse import urlencode

    query = urlencode(
        {
            "user_id": user_id,
            "budget": budget,
            "set_count": set_count,
        }
    )
    req = request.Request(
        f"{BUDGET_SET_API_URL}?{query}",
        headers={"Accept": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=20) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        return DEFAULT_BUDGET_SET_RESPONSE
    return parsed


st.set_page_config(page_title="검색/추천 성능 대시보드", layout="wide")
st.title("검색/추천 성능 대시보드")
st.caption("검색, 추천, 랭킹, A/B Test 결과를 한 화면에서 확인합니다.")

st.subheader("검색 성능 지표")
if SEARCH_REPORT_PATH.exists():
    try:
        report = json.loads(SEARCH_REPORT_PATH.read_text(encoding="utf-8"))
        search_metrics = report.get("search", {})
        checks = report.get("checks", {})
        thresholds = report.get("thresholds", {})
        metadata = report.get("metadata", {})
        metric_k_value = int(metadata.get("metric_k", 10))
        ndcg_metric_name = f"NDCG@{metric_k_value}"

        search_cards = st.columns(5)
        search_cards[0].metric("평가 샘플 수", f"{metadata.get('samples_evaluated', 0)}")
        search_cards[1].metric("MRR", f"{search_metrics.get('MRR', 0.0):.4f}")
        search_cards[2].metric(
            f"nDCG@{metric_k_value}",
            f"{search_metrics.get(ndcg_metric_name, 0.0):.4f}",
        )
        search_cards[3].metric("평균 지연 시간", f"{search_metrics.get('avg_wall_latency_ms', 0.0):.2f} ms")
        search_cards[4].metric("P95 지연 시간", f"{search_metrics.get('p95_wall_latency_ms', 0.0):.2f} ms")

        status_df = pd.DataFrame(
            [
                {"metric": "MRR", "value": search_metrics.get("MRR", 0.0), "target": thresholds.get("mrr_min", 0.55)},
                {
                    "metric": f"nDCG@{metric_k_value}",
                    "value": search_metrics.get(ndcg_metric_name, 0.0),
                    "target": thresholds.get("ndcg_at_10_min", 0.50),
                },
                {
                    "metric": "Latency(ms)",
                    "value": search_metrics.get("avg_wall_latency_ms", 0.0),
                    "target": thresholds.get("latency_ms_max", 200.0),
                },
            ]
        )
        status_df["passed"] = [
            checks.get("mrr_meets_target", False),
            checks.get("ndcg_meets_target", False),
            checks.get("latency_within_200ms", False),
        ]
        status_df["passed"] = status_df["passed"].map({True: "PASS", False: "FAIL"})

        quality_df = status_df[status_df["metric"] != "Latency(ms)"].copy()
        latency_df = status_df[status_df["metric"] == "Latency(ms)"].copy()

        quality_chart = (
            alt.Chart(quality_df)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("metric:N", title=None),
                y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "passed:N",
                    legend=None,
                    scale=alt.Scale(domain=["PASS", "FAIL"], range=["#2E8B57", "#C0392B"]),
                ),
                tooltip=[
                    "metric",
                    alt.Tooltip("value:Q", format=".4f"),
                    alt.Tooltip("target:Q", format=".4f"),
                    alt.Tooltip("passed:N", title="status"),
                ],
            )
            .properties(height=240)
        )

        quality_target_rule = (
            alt.Chart(quality_df)
            .mark_rule(color="#2C3E50", strokeDash=[4, 4], strokeWidth=2)
            .encode(y="target:Q", x="metric:N")
        )

        latency_chart = (
            alt.Chart(latency_df)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#1F77B4")
            .encode(
                x=alt.X("metric:N", title=None),
                y=alt.Y("value:Q", title="ms"),
                tooltip=[
                    "metric",
                    alt.Tooltip("value:Q", format=".2f"),
                    alt.Tooltip("target:Q", format=".2f"),
                    alt.Tooltip("passed:N", title="status"),
                ],
            )
            .properties(height=240)
        )

        latency_target_rule = (
            alt.Chart(latency_df)
            .mark_rule(color="#C0392B", strokeDash=[4, 4], strokeWidth=2)
            .encode(y="target:Q")
        )

        search_chart_left, search_chart_right = st.columns(2)
        with search_chart_left:
            st.altair_chart(quality_chart + quality_target_rule, use_container_width=True)
        with search_chart_right:
            st.altair_chart(latency_chart + latency_target_rule, use_container_width=True)

        st.dataframe(status_df, use_container_width=True, hide_index=True)
        st.caption(f"리포트 파일: {SEARCH_REPORT_PATH.name}")
    except Exception as error:
        st.error(f"검색 리포트를 불러오는 중 오류가 발생했습니다: {error}")
else:
    st.info(
        "search_metrics_report.json 파일이 없습니다. "
        "`python .\\search_engine\\generate_search_metrics_report.py --endpoint http://localhost:8002/search` "
        "명령으로 리포트를 생성한 뒤 새로고침해 주세요."
    )

st.divider()
st.subheader("추천 성능 지표")
if RECOMMENDATION_REPORT_PATH.exists():
    try:
        report = json.loads(RECOMMENDATION_REPORT_PATH.read_text(encoding="utf-8"))
        metadata = report.get("metadata", {})
        candidate = report.get("candidate", {})
        ranking = report.get("ranking", {})
        recommendation = report.get("recommendation", {}).get("current_model", {})
        cold_start = recommendation.get("cold_start_subset", {})

        rec_cards = st.columns(6)
        rec_cards[0].metric("Recall@300", f"{candidate.get('Recall@300', 0.0):.4f}")
        rec_cards[1].metric("Ranking AUC", f"{ranking.get('auc', 0.0):.4f}")
        rec_cards[2].metric("HitRate@50", f"{recommendation.get('HitRate@50', 0.0):.4f}")
        rec_cards[3].metric("NDCG@50", f"{recommendation.get('NDCG@50', 0.0):.4f}")
        rec_cards[4].metric("Coverage@50", f"{recommendation.get('Coverage@50', 0.0):.4f}")
        rec_cards[5].metric("평가 사용자 수", f"{recommendation.get('users_evaluated', 0)}")

        recommendation_df = pd.DataFrame(
            [
                {"metric": "Recall@300", "value": candidate.get("Recall@300", 0.0), "group": "Candidate"},
                {"metric": "Ranking AUC", "value": ranking.get("auc", 0.0), "group": "Ranking"},
                {"metric": "HitRate@50", "value": recommendation.get("HitRate@50", 0.0), "group": "Recommendation"},
                {"metric": "NDCG@50", "value": recommendation.get("NDCG@50", 0.0), "group": "Recommendation"},
                {"metric": "Coverage@50", "value": recommendation.get("Coverage@50", 0.0), "group": "Recommendation"},
            ]
        )
        recommendation_chart = (
            alt.Chart(recommendation_df)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("metric:N", title=None),
                y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("group:N", legend=alt.Legend(title="파이프라인")),
                tooltip=["metric", alt.Tooltip("value:Q", format=".4f"), "group"],
            )
            .properties(height=280)
        )
        st.altair_chart(recommendation_chart, use_container_width=True)

        rec_meta_col, cold_start_col = st.columns(2)
        with rec_meta_col:
            st.markdown("**추천 리포트 요약**")
            st.write(
                {
                    "평가 행 수": metadata.get("rows", 0),
                    "사용자 수": metadata.get("users", 0),
                    "아이템 수": metadata.get("items", 0),
                    "top_k": metadata.get("top_k", 0),
                    "candidate_k": metadata.get("candidate_k", 0),
                }
            )
        with cold_start_col:
            st.markdown("**콜드 스타트 구간**")
            st.write(
                {
                    "평가 사용자 수": cold_start.get("users_evaluated", 0),
                    "HitRate@50": round(float(cold_start.get("HitRate@50", 0.0)), 4),
                    "NDCG@50": round(float(cold_start.get("NDCG@50", 0.0)), 4),
                    "Coverage@50": round(float(cold_start.get("Coverage@50", 0.0)), 4),
                }
            )
        st.caption(f"리포트 파일: {RECOMMENDATION_REPORT_PATH.name}")
    except Exception as error:
        st.error(f"추천 리포트를 불러오는 중 오류가 발생했습니다: {error}")
else:
    st.info("추천 성능 리포트 파일이 없습니다.")

st.divider()
demo_col_left, demo_col_right = st.columns(2)

with demo_col_left:
    st.subheader("온보딩 페르소나 분석")
    st.caption("온보딩 입력을 바탕으로 생성된 페르소나 점수를 시각화합니다.")
    onboarding_user_id = st.text_input("사용자 ID", value="demo_user_001", key="onboarding_user_id")
    onboarding_description = st.text_area(
        "취향 설명",
        value="미니멀하고 블랙 아우터를 좋아하고, 실용적이면서도 약간 트렌디한 스타일을 선호합니다.",
        height=120,
        key="onboarding_description",
    )
    onboarding_style_choices = st.multiselect(
        "선호 스타일",
        options=["casual", "minimal", "street", "sporty", "feminine", "classic"],
        default=["minimal", "classic"],
        key="onboarding_style_choices",
    )
    onboarding_budget_range = st.selectbox(
        "예산 범위",
        options=["none", "low", "mid", "high"],
        index=0,
        key="onboarding_budget_range",
    )

    persona_scores: dict[str, float] = st.session_state.get("onboarding_api_persona_scores", {})
    if st.button("페르소나 점수 생성", key="run_onboarding_api"):
        try:
            persona_scores = fetch_onboarding_persona_scores(
                user_id=onboarding_user_id.strip(),
                description=onboarding_description.strip(),
                style_choices=onboarding_style_choices,
                budget_range=None if onboarding_budget_range == "none" else onboarding_budget_range,
            )
            st.session_state["onboarding_api_persona_scores"] = persona_scores
        except error.HTTPError as http_error:
            st.error(f"온보딩 API 호출 오류: {http_error.code}")
        except Exception as api_error:
            st.error(f"온보딩 API 요청에 실패했습니다: {api_error}")

    persona_df = pd.DataFrame(
        [
            {"persona": PERSONA_LABELS.get(key, key), "score": value}
            for key, value in persona_scores.items()
        ]
    )
    if not persona_df.empty:
        persona_df = persona_df.sort_values("score", ascending=False)
    else:
        persona_df = pd.DataFrame(columns=["persona", "score"])

    top_persona = persona_df.iloc[0]["persona"] if not persona_df.empty else "-"
    st.metric("최상위 페르소나", top_persona)

    persona_chart = (
        alt.Chart(persona_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("score:Q", title="점수 (%)"),
            y=alt.Y("persona:N", sort="-x", title=None),
            color=alt.value("#1F77B4"),
            tooltip=["persona", alt.Tooltip("score:Q", format=".1f")],
        )
        .properties(height=320)
    )
    if not persona_df.empty:
        st.altair_chart(persona_chart, use_container_width=True)
        st.dataframe(persona_df, use_container_width=True, hide_index=True)
    else:
        st.info("입력값을 바탕으로 페르소나 점수를 생성해 보세요.")

with demo_col_right:
    st.subheader("예산 기반 세트 추천")
    st.caption("예산 안에서 구성한 추천 세트를 확인합니다.")
    budget_user_id = st.text_input("세트 추천 사용자 ID", value="user_1024", key="budget_user_id")
    budget_limit = st.number_input("예산", min_value=10000, value=DEFAULT_BUDGET_SET_RESPONSE["budget"], step=10000)
    if "budget_set_response" not in st.session_state:
        st.session_state["budget_set_response"] = DEFAULT_BUDGET_SET_RESPONSE

    budget_action_col, budget_state_col = st.columns([1, 3])
    with budget_action_col:
        if st.button("세트 추천 불러오기", key="load_budget_sets"):
            try:
                st.session_state["budget_set_response"] = fetch_budget_set_response(
                    user_id=budget_user_id.strip() or "user_1024",
                    budget=int(budget_limit),
                    set_count=2,
                )
            except error.HTTPError as http_error:
                st.error(f"세트 추천 API 호출 오류: {http_error.code}")
            except Exception as api_error:
                st.warning(f"세트 추천 API 연결에 실패해 예시 데이터를 표시합니다: {api_error}")
                st.session_state["budget_set_response"] = DEFAULT_BUDGET_SET_RESPONSE
    with budget_state_col:
        st.caption("API 응답이 가능하면 실제 세트 추천 결과를, 아니면 시연용 예시 데이터를 보여줍니다.")

    budget_response = st.session_state.get("budget_set_response", DEFAULT_BUDGET_SET_RESPONSE)
    budget_sets = budget_response.get("sets", DEFAULT_BUDGET_SET_RESPONSE["sets"])
    budget_rows = []
    for set_index, set_items in enumerate(budget_sets, start=1):
        for item in set_items:
            budget_rows.append(
                {
                    "set_name": f"세트 {set_index}",
                    "article_id": item.get("article_id", item.get("product_id", "-")),
                    "item_name": item.get("name", item.get("product_id", "-")),
                    "brand": item.get("brand", "-"),
                    "category": item.get("category", "-"),
                    "price": item.get("price_int", item.get("price", 0)),
                    "score": item.get("score", 0.0),
                }
            )
    budget_set_df = pd.DataFrame(budget_rows)
    if not budget_set_df.empty:
        budget_set_df["price"] = pd.to_numeric(budget_set_df["price"], errors="coerce").fillna(0).astype(int)
        total_by_set = budget_set_df.groupby("set_name", as_index=False)["price"].sum()
        total_by_set["within_budget"] = total_by_set["price"] <= int(budget_limit)

        budget_cards = st.columns(len(total_by_set) if len(total_by_set) > 0 else 1)
        for card, row in zip(budget_cards, total_by_set.itertuples(index=False)):
            card.metric(
                row.set_name,
                f"{int(row.price):,}원",
                "PASS" if row.within_budget else "OVER",
            )

        budget_chart = (
            alt.Chart(total_by_set)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("set_name:N", title=None),
                y=alt.Y("price:Q", title="total price"),
                color=alt.Color(
                    "within_budget:N",
                    scale=alt.Scale(domain=[True, False], range=["#2E8B57", "#C0392B"]),
                    legend=alt.Legend(title="within budget"),
                ),
                tooltip=["set_name", alt.Tooltip("price:Q", format=",.0f"), "within_budget"],
            )
            .properties(height=260)
        )
        budget_rule = alt.Chart(pd.DataFrame([{"budget": budget_limit}])).mark_rule(strokeDash=[4, 4]).encode(
            y="budget:Q"
        )
        st.altair_chart(budget_chart + budget_rule, use_container_width=True)
        st.dataframe(budget_set_df, use_container_width=True, hide_index=True)
    else:
        st.info("세트 추천 결과를 불러오면 이 영역에 조합 결과가 표시됩니다.")


st.divider()
st.subheader("추천 이유 예시")
st.caption("추천 이유 문장과 추천 파이프라인 지연 시간을 함께 보여줍니다.")

reason_user_id = st.text_input("추천 조회 사용자 ID", value="user_1024", key="reason_user_id")
if "recommendation_reason_response" not in st.session_state:
    st.session_state["recommendation_reason_response"] = DEFAULT_RECOMMENDATION_RESPONSE

reason_action_col, reason_state_col = st.columns([1, 3])
with reason_action_col:
    if st.button("추천 결과 불러오기", key="load_recommendation_reasons"):
        try:
            st.session_state["recommendation_reason_response"] = fetch_recommendation_response(
                user_id=reason_user_id.strip() or "user_1024",
                top_n=3,
                include_reasons=True,
            )
        except error.HTTPError as http_error:
            st.error(f"추천 API 호출 오류: {http_error.code}")
        except Exception as api_error:
            st.warning(f"추천 API 연결에 실패해 예시 데이터를 표시합니다: {api_error}")
            st.session_state["recommendation_reason_response"] = DEFAULT_RECOMMENDATION_RESPONSE
with reason_state_col:
    st.caption("API 응답이 가능하면 실제 추천 결과를, 아니면 시연용 예시 데이터를 보여줍니다.")

recommendation_response = st.session_state.get("recommendation_reason_response", DEFAULT_RECOMMENDATION_RESPONSE)
recommendation_items = recommendation_response.get("recommendations", DEFAULT_RECOMMENDATION_RESPONSE["recommendations"])
reason_df = pd.DataFrame(
    [
        {
            "rank": item.get("rank", index + 1),
            "product_id": item.get("product_id", "-"),
            "name": item.get("name", item.get("product_id", f"item_{index + 1}")),
            "reason": item.get("reason", "-"),
            "reason_text": item.get("reason_text", "추천 이유 문장이 제공되지 않았습니다."),
            "score": item.get("score", 0.0),
            "price": item.get("price", 0),
        }
        for index, item in enumerate(recommendation_items)
    ]
)

reason_cards = st.columns(len(reason_df) if len(reason_df) > 0 else 1)
for card, row in zip(reason_cards, reason_df.itertuples(index=False)):
    card.metric(f"Rank {row.rank}", row.name, f"{row.score:.2f}")

for row in reason_df.itertuples(index=False):
    with st.container():
        st.markdown(f"**#{row.rank} {row.name}**")
        st.write(
            {
                "product_id": row.product_id,
                "reason": row.reason,
                "reason_text": row.reason_text,
                "price": row.price,
                "score": round(float(row.score), 4),
            }
        )

latency_payload = recommendation_response.get("pipeline_latency", DEFAULT_RECOMMENDATION_RESPONSE["pipeline_latency"])
latency_df = pd.DataFrame(
    [
        {"stage": "Candidate", "latency_ms": latency_payload["candidate_ms"]},
        {"stage": "Ranking", "latency_ms": latency_payload["ranking_ms"]},
        {"stage": "Reranking", "latency_ms": latency_payload["reranking_ms"]},
        {"stage": "Total", "latency_ms": latency_payload["total_ms"]},
    ]
)
reason_latency_chart = (
    alt.Chart(latency_df)
    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#6C5CE7")
    .encode(
        x=alt.X("stage:N", title=None),
        y=alt.Y("latency_ms:Q", title="ms"),
        tooltip=["stage", "latency_ms"],
    )
    .properties(height=220)
)
st.altair_chart(reason_latency_chart, use_container_width=True)

st.divider()
ranking_col, ab_col = st.columns(2)

with ranking_col:
    st.subheader("랭킹 성능 지표")
    try:
        ranking_reports = discover_ranking_reports()
        if not ranking_reports:
            st.info("내부 랭킹 실험 리포트가 없습니다.")
        else:
            selected_report_name = st.selectbox(
                "랭킹 리포트 선택",
                options=[report.name for report in ranking_reports],
                index=0,
                key="ranking_report_select",
            )
            selected_report_path = next(report for report in ranking_reports if report.name == selected_report_name)
            ranking_report = load_ranking_report(selected_report_path)
            experiment = ranking_report.get("experiment", {})
            config = experiment.get("config", {})
            metrics = ranking_report.get("metrics", {})
            top_k = int(config.get("top_k", 50))

            st.caption(f"리포트 경로: `{selected_report_path}`")

            metrics_df = pd.DataFrame(
                [
                    {"metric": "Ranking AUC", "value": float(metrics.get("auc", 0.0))},
                    {"metric": f"HitRate@{top_k}", "value": float(metrics.get(f"HitRate@{top_k}", 0.0))},
                    {"metric": f"NDCG@{top_k}", "value": float(metrics.get(f"NDCG@{top_k}", 0.0))},
                ]
            )

            metric_cards = st.columns(5)
            metric_cards[0].metric("Ranking AUC", f"{metrics.get('auc', 0.0):.4f}")
            metric_cards[1].metric(f"HitRate@{top_k}", f"{metrics.get(f'HitRate@{top_k}', 0.0):.4f}")
            metric_cards[2].metric(f"NDCG@{top_k}", f"{metrics.get(f'NDCG@{top_k}', 0.0):.4f}")
            metric_cards[3].metric("평가 사용자 수", f"{int(metrics.get('users_evaluated', 0)):,}")
            metric_cards[4].metric("평가 행 수", f"{int(metrics.get('rows_evaluated', 0)):,}")

            ranking_chart = (
                alt.Chart(metrics_df)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("metric:N", title=None),
                    y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("metric:N", legend=None),
                    tooltip=["metric", alt.Tooltip("value:Q", format=".4f")],
                )
                .properties(height=320)
            )
            st.altair_chart(ranking_chart, use_container_width=True)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            ranking_meta_df = pd.DataFrame(
                [
                    {"field": "실험 이름", "value": experiment.get("name", "-")},
                    {"field": "생성 시각(UTC)", "value": experiment.get("generated_at_utc", "-")},
                    {"field": "데이터 경로", "value": experiment.get("data_path", "-")},
                    {"field": "모델 유형", "value": config.get("model_type", "-")},
                    {"field": "체크포인트 경로", "value": config.get("checkpoint_dir", "-")},
                    {"field": "최대 사용자 수", "value": config.get("max_users", "-")},
                ]
            )
    except Exception as error:
        st.error(f"랭킹 리포트를 불러오는 중 오류가 발생했습니다: {error}")

with ab_col:
    st.subheader("A/B Test")
    try:
        ab_specs = [
            {
                "title": "Baseline vs Persona Optimized",
                "baseline_path": BASELINE_REPORT_PATH,
                "variant_path": PERSONA_AB_REPORT_PATH,
                "variant_label": "Persona Optimized",
                "description": "기본 추천 대비 cold-start/persona 기능이 포함된 추천 variant 비교",
            },
            {
                "title": "Baseline vs Coverage Strong",
                "baseline_path": BASELINE_REPORT_PATH,
                "variant_path": COVERAGE_AB_REPORT_PATH,
                "variant_label": "Coverage Strong",
                "description": "기본 추천 대비 coverage/diversity 강화 추천 variant 비교",
            },
        ]

        for spec in ab_specs:
            st.markdown(f"**{spec['title']}**")
            st.caption(spec["description"])

            if not spec["baseline_path"].exists() or not spec["variant_path"].exists():
                st.warning("비교 리포트 파일이 없습니다.")
                continue

            baseline_report = load_json_report(spec["baseline_path"])
            variant_report = load_json_report(spec["variant_path"])
            comparison_df, comparison_meta_df = build_ab_comparison_rows(
                baseline_report=baseline_report,
                variant_report=variant_report,
                variant_label=spec["variant_label"],
            )

            card_cols = st.columns(len(comparison_df))
            for card, row in zip(card_cols, comparison_df.itertuples(index=False)):
                card.metric(
                    row.label,
                    f"{row.variant:.4f}",
                    delta=f"{row.absolute_diff:+.4f} ({row.relative_lift:+.2%})",
                )

            chart_df = comparison_df.melt(
                id_vars=["metric", "label", "absolute_diff", "relative_lift"],
                value_vars=["baseline", "variant"],
                var_name="group",
                value_name="value",
            )
            chart_df["group"] = chart_df["group"].map({"baseline": "Baseline", "variant": spec["variant_label"]})

            comparison_chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("label:N", title=None),
                    y=alt.Y("value:Q", title="Metric value", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("group:N", title=None),
                    xOffset="group:N",
                    tooltip=[
                        "label",
                        "group",
                        alt.Tooltip("value:Q", format=".4f"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(comparison_chart, use_container_width=True)

            display_df = comparison_df.loc[:, ["label", "baseline", "variant", "absolute_diff", "relative_lift"]].copy()
            display_df.columns = ["metric", "baseline", spec["variant_label"], "absolute_diff", "relative_lift"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as error:
        st.error(f"A/B Test 비교 중 오류가 발생했습니다: {error}")
