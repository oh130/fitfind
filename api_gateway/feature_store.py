"""
Redis 기반 Feature Store.

저장 구조:
  user:{user_id}:recent_clicks   — List<item_id>, 최근 20개
  user:{user_id}:session_interest — Hash<category, score>
  user:{user_id}:persona_scores   — JSON<persona, score>, 온보딩 페르소나 비율
  user:{user_id}:click_count     — 총 클릭 수 (int)
"""

import hashlib
import json
import redis

RECENT_CLICKS_MAX = 20
CLICK_TTL = 60 * 60 * 24 * 7  # 7일
QUERY_INTEREST_CACHE_TTL = 60 * 60 * 24 * 7  # 7일
SEARCH_INTENT_CACHE_TTL = 60 * 60 * 24 * 7  # 7일
FALLBACK_SEARCH_INTENT_CACHE_TTL = 60 * 60  # 1시간
PERSONA_SCORE_TTL = 60 * 60 * 24 * 30  # 30일


class RedisFeatureStore:
    def __init__(self, host: str = "redis", port: int = 6379, db: int = 0):
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    # ── 클릭 이벤트 ──────────────────────────────────────────
    def push_click(self, user_id: str, item_id: str) -> None:
        key = f"user:{user_id}:recent_clicks"
        self.r.lpush(key, item_id)
        self.r.ltrim(key, 0, RECENT_CLICKS_MAX - 1)
        self.r.expire(key, CLICK_TTL)
        self.r.incr(f"user:{user_id}:click_count")

    def get_recent_clicks(self, user_id: str, n: int = 10) -> list[str]:
        return self.r.lrange(f"user:{user_id}:recent_clicks", 0, n - 1)

    def get_click_count(self, user_id: str) -> int:
        val = self.r.get(f"user:{user_id}:click_count")
        return int(val) if val else 0

    # ── 세션 관심사 ──────────────────────────────────────────
    def set_session_interest(self, user_id: str, interest: dict) -> None:
        key = f"user:{user_id}:session_interest"
        self.r.set(key, json.dumps(interest), ex=CLICK_TTL)

    def get_session_interest(self, user_id: str) -> dict:
        val = self.r.get(f"user:{user_id}:session_interest")
        return json.loads(val) if val else {}

    # ── 온보딩 페르소나 비율 ────────────────────────────────
    def set_persona_scores(self, user_id: str, persona_scores: dict) -> None:
        self.r.set(f"user:{user_id}:persona_scores", json.dumps(persona_scores), ex=PERSONA_SCORE_TTL)

    def get_persona_scores(self, user_id: str) -> dict:
        val = self.r.get(f"user:{user_id}:persona_scores")
        if not val:
            return {}
        try:
            parsed = json.loads(val)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def invalidate_recommendation_cache(self, user_id: str) -> int:
        """Delete cached recommendation responses for one user.

        ':reasons' 키는 삭제하지 않는다 — 추천 이유는 Gemini 호출이 필요하므로
        검색/클릭 이벤트마다 재생성하지 않고 TTL(5분)이 만료될 때까지 유지한다.
        """
        pattern = f"cache:recommend:{user_id}:*"
        keys = [k for k in self.r.scan_iter(match=pattern, count=100)
                if not k.endswith(":reasons")]
        if keys:
            return int(self.r.delete(*keys))
        return 0

    # ── 검색어 관심사 캐시 ───────────────────────────────────
    def _query_interest_cache_key(self, query_text: str) -> str:
        normalized = query_text.strip().lower()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"cache:query_interest:{digest}"

    def get_query_interest_cache(self, query_text: str) -> dict | None:
        val = self.r.get(self._query_interest_cache_key(query_text))
        if not val:
            return None
        try:
            cached = json.loads(val)
        except json.JSONDecodeError:
            return None
        return cached if isinstance(cached, dict) else None

    def set_query_interest_cache(self, query_text: str, interest: dict) -> None:
        self.r.set(
            self._query_interest_cache_key(query_text),
            json.dumps(interest),
            ex=QUERY_INTEREST_CACHE_TTL,
        )

    # ── 검색 의도 캐시 ──────────────────────────────────────
    def _search_intent_cache_key(self, query_text: str) -> str:
        normalized = query_text.strip().lower()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"cache:search_intent:{digest}"

    def get_search_intent_cache(self, query_text: str) -> dict | None:
        val = self.r.get(self._search_intent_cache_key(query_text))
        if not val:
            return None
        try:
            cached = json.loads(val)
        except json.JSONDecodeError:
            return None
        return cached if isinstance(cached, dict) else None

    def set_search_intent_cache(self, query_text: str, intent: dict, *, fallback: bool = False) -> None:
        self.r.set(
            self._search_intent_cache_key(query_text),
            json.dumps(intent),
            ex=FALLBACK_SEARCH_INTENT_CACHE_TTL if fallback else SEARCH_INTENT_CACHE_TTL,
        )

    # ── 통합 조회 ────────────────────────────────────────────
    def get_user_features(self, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "recent_clicks": self.get_recent_clicks(user_id),
            "session_interest": self.get_session_interest(user_id),
            "persona_scores": self.get_persona_scores(user_id),
            "click_count": self.get_click_count(user_id),
        }
