"""
kr_daily_screen.py  (v3 — 미너비니 트렌드 템플릿 통합)
==============================================================================
코스피/코스닥 일일 팩터 + 미너비니 추세 스크리너 & 대시보드 생성기

v3 변경점:
  · 기술점수를 '미너비니 트렌드 템플릿(8기준)'으로 대체 — 두서없는 모멘텀이 아니라
    명확한 Stage 2 상승추세 체크리스트
  · ⭐ = 8기준 전부 통과 (미너비니 순수 게이트)
  · 미너비니 점수(0~100) = 8기준 중 충족 개수 → 종합 시그널의 기술 파트로 사용
  · "미너비니 통과만" 필터 + 종목별 8기준 점 체크리스트

미너비니 8기준 (SMA 기준, EMA 아님):
  1) 주가 > 50일선   2) 주가 > 150일선   3) 주가 > 200일선
  4) 50일선 > 150일선   5) 150일선 > 200일선   6) 200일선 1개월 상승
  7) 주가가 52주 고점의 25% 이내   8) RS ≥ 70 (유니버스 백분위 자체 산출)

주의: 시그널 등급은 기계적 모델 출력이며 매매 추천이 아니다. 개별 종목 정성 분석 필수.

데이터: pykrx (펀더멘털·시총·업종) + FinanceDataReader (수정주가 시계열)
        FDR이 액면분할/증자 반영 수정주가를 제공하므로 모멘텀 왜곡이 원천 제거됨.
설치:   pip install pykrx finance-datareader pandas numpy
실행:   python kr_daily_screen.py
        (KRX_ID / KRX_PW 환경변수 선행 필요 — pykrx 펀더멘털용)
런타임: 종목별 주가 히스토리 수집으로 6~9분 (진행표시 있음)
==============================================================================
"""

from __future__ import annotations
import json
import time
import datetime as dt
import numpy as np
import pandas as pd
from pykrx import stock
import FinanceDataReader as fdr

CONFIG = {
    "markets": ["KOSPI", "KOSDAQ"],
    "min_market_cap_krw": 300e9,
    "min_trading_value_krw": 1e9,
    "history_days": 520,            # 252거래일(52주) + 200MA 1개월추세 확보
    "weights": {"value": 0.25, "momentum": 0.50, "quality": 0.25},
    "signal_factor_w": 0.55,        # 백테스트 검증: 팩터 55% + 미너비니 45%가 최적
    "signal_mnv_w": 0.45,           #   (CAGR 42.6%, Sharpe 1.36, MDD -23.4%)
    "signal_entry_w": 0.0,          # 진입적합 반영 안 함 — 백테스트에서 수익 -12.8%p 악화
    "ext50_hot": 25.0,              # 과열 판정 기준 (표시 전용)
    "ext200_hot": 60.0,
    "base_low": 5.0,
    "base_high": 15.0,
    "exclude_hot": False,           # 과열 제외 비활성 (백테스트 기각)
    # 등급을 상대순위(백분위)로 부여 → 매일 일정 개수 유지
    # 강력매수 6% ≈ 30종목 = 백테스트가 실제 매수·검증한 포트폴리오 규모와 동일
    "rating_pcts": {"강력매수": 6, "매수": 20, "중립": 50, "매도": 80},
    "rs_threshold": 70,             # 미너비니 기준8: RS ≥ 70
    "winsorize_pct": 0.02,
    "api_sleep_sec": 0.30,
    "output_html": "kr_dashboard.html",
    "output_json": "kr_screen_data.json",
}

_NON_SECTOR_HINTS = [
    "코스피", "코스닥", "200", "150", "100", "50", "30", "대형", "중형", "소형",
    "배당", "가치", "성장", "우량", "변동성", "모멘텀", "퀄리티", "로우볼", "고배당",
    "ESG", "탄소", "지배구조", "리츠", "인프라", "우선주", "지수", "TOP", "섹터",
]

RATING_NAMES = ["강력매수", "매수", "중립", "매도", "강력매도"]


def _sleep(): time.sleep(CONFIG["api_sleep_sec"])
def _nearest_bday(d): return stock.get_nearest_business_day_in_a_week(d, prev=True)

def _recent_bday():
    d = dt.date.today()
    for _ in range(10):
        s = d.strftime("%Y%m%d")
        try:
            df = stock.get_market_cap_by_ticker(s, market="KOSPI")
            if not df.empty and df["종가"].sum() > 0:
                return s
        except Exception:
            pass
        d -= dt.timedelta(days=1)
    raise RuntimeError("최근 영업일을 찾지 못했습니다.")

def _days_ago(base, days):
    d = dt.date(int(base[:4]), int(base[4:6]), int(base[6:])) - dt.timedelta(days=days)
    return d.strftime("%Y%m%d")

def _winsorize(s, pct):
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)

def _zscore(s):
    s = _winsorize(s.astype(float), CONFIG["winsorize_pct"])
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).fillna(0.0)

def _pct_rank(s): return s.rank(pct=True) * 100
def _z_to_display(z): return ((z.clip(-2.5, 2.5) + 2.5) / 5.0 * 100)


# ------------------------------------------------------------------ 업종 분류
def build_sector_map(date) -> dict:
    mapping = {}
    for mkt in CONFIG["markets"]:
        try:
            codes = stock.get_index_ticker_list(date, market=mkt); _sleep()
        except Exception:
            continue
        for code in codes:
            try:
                name = stock.get_index_ticker_name(code)
            except Exception:
                continue
            if any(h in name for h in _NON_SECTOR_HINTS):
                continue
            members = None
            for attempt in (
                lambda: stock.get_index_portfolio_deposit_file(code, date),
                lambda: stock.get_index_portfolio_deposit_file(code),
            ):
                try:
                    members = attempt(); break
                except Exception:
                    continue
            _sleep()
            if not members:
                continue
            for t in members:
                mapping.setdefault(t, name)
    return mapping


# ------------------------------------------------------------------ 주가 수집 (FDR 수정주가)
def _fetch_close(ticker, start, end):
    """FinanceDataReader로 수정주가 종가 확보.
    FDR은 액면분할/증자를 반영한 수정주가를 기본 제공 → 별도 보정 불필요."""
    s = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    e = f"{end[:4]}-{end[4:6]}-{end[6:]}"
    try:
        df = fdr.DataReader(ticker, s, e)
    except Exception:
        return None
    if df is None or df.empty or "Close" not in df:
        return None
    close = df["Close"].astype(float)
    close = close[close > 0]
    return close if len(close) >= 50 else None


# ------------------------------------------------------------------ 미너비니 + 모멘텀
def compute_technicals(ticker, start, end) -> dict:
    """종목 주가에서 미너비니 8기준(1~7) + 모멘텀 + RS원자료 계산.
    기준8(RS)은 유니버스 전체 비교라 screen()에서 완성."""
    out = {"mom": np.nan, "rs_raw": np.nan, "align": "데이터없음",
           "c1": False, "c2": False, "c3": False, "c4": False,
           "c5": False, "c6": False, "c7": False, "dist_high": np.nan,
           "ext50": np.nan, "ext200": np.nan, "pullback": np.nan,
           "vcp": np.nan, "entry_score": 0.0, "state": "데이터없음"}

    close = _fetch_close(ticker, start, end)
    if close is None:
        return out
    n = len(close)
    if n < 50:
        return out
    price = close.iloc[-1]

    # 팩터용 12-1 모멘텀 (FDR 수정주가 기준이므로 상한 불필요)
    if n >= 252:
        mom = (close.iloc[-21] / close.iloc[-252] - 1) * 100
    elif n > 40:
        mom = (close.iloc[-21] / close.iloc[0] - 1) * 100
    else:
        mom = np.nan
    out["mom"] = mom

    # RS 원자료: IBD식 분기가중 (최근 분기 2배)
    if n >= 252:
        p = close
        q1 = p.iloc[-1] / p.iloc[-63] - 1
        q2 = p.iloc[-63] / p.iloc[-126] - 1
        q3 = p.iloc[-126] / p.iloc[-189] - 1
        q4 = p.iloc[-189] / p.iloc[-252] - 1
        out["rs_raw"] = 0.4 * q1 + 0.2 * q2 + 0.2 * q3 + 0.2 * q4

    ma50 = close.rolling(50).mean().iloc[-1] if n >= 50 else np.nan
    ma150 = close.rolling(150).mean().iloc[-1] if n >= 150 else np.nan
    ma200 = close.rolling(200).mean().iloc[-1] if n >= 200 else np.nan
    ma200_1mo = close.rolling(200).mean().iloc[-22] if n >= 222 else np.nan

    # 52주 고점 (최근 252거래일)
    hi52 = close.iloc[-252:].max() if n >= 252 else close.max()
    if hi52 > 0:
        out["dist_high"] = (price / hi52 - 1) * 100  # 음수 = 고점 아래 %

    # 미너비니 기준 1~7
    out["c1"] = bool(price > ma50) if not np.isnan(ma50) else False
    out["c2"] = bool(price > ma150) if not np.isnan(ma150) else False
    out["c3"] = bool(price > ma200) if not np.isnan(ma200) else False
    out["c4"] = bool(ma50 > ma150) if not (np.isnan(ma50) or np.isnan(ma150)) else False
    out["c5"] = bool(ma150 > ma200) if not (np.isnan(ma150) or np.isnan(ma200)) else False
    out["c6"] = bool(ma200 > ma200_1mo) if not (np.isnan(ma200) or np.isnan(ma200_1mo)) else False
    out["c7"] = bool(price >= 0.75 * hi52) if hi52 > 0 else False

    # 배열 표시 (참고용)
    if not (np.isnan(ma50) or np.isnan(ma150) or np.isnan(ma200)):
        if price > ma50 > ma150 > ma200:
            out["align"] = "정배열"
        elif price < ma50 < ma150 < ma200:
            out["align"] = "역배열"
        else:
            out["align"] = "혼조"

    # ---------------- 과열도 / 눌림목 판정 (진입 타이밍) ----------------
    # 이격도: 이동평균에서 얼마나 떠 있나
    if not np.isnan(ma50) and ma50 > 0:
        out["ext50"] = (price / ma50 - 1) * 100
    if not np.isnan(ma200) and ma200 > 0:
        out["ext200"] = (price / ma200 - 1) * 100

    # 최근 고점(60일) 대비 조정폭 — 5~20% 눌림이 건설적
    if n >= 60:
        hi60 = close.iloc[-60:].max()
        if hi60 > 0:
            out["pullback"] = (price / hi60 - 1) * 100   # 음수 = 고점 아래

    # 변동성 수축(VCP): 최근 20일 변동성 ÷ 이전 60일 변동성. <1 이면 수축
    if n >= 80:
        r = close.pct_change().dropna()
        v_recent = r.iloc[-20:].std()
        v_prior = r.iloc[-80:-20].std()
        if v_prior and v_prior > 0:
            out["vcp"] = float(v_recent / v_prior)

    # 상태 분류
    ext50, ext200 = out["ext50"], out["ext200"]
    uptrend_ok = out["c3"] and out["c5"]        # 200일선 위 + 150>200 (구조적 상승추세)
    if np.isnan(ext50):
        out["state"] = "데이터없음"
    elif not uptrend_ok:
        out["state"] = "이탈"                    # 추세 훼손
    elif (ext50 > CONFIG["ext50_hot"]) or (not np.isnan(ext200) and ext200 > CONFIG["ext200_hot"]):
        out["state"] = "과열"                    # 추세는 맞지만 너무 늘어짐
    elif -CONFIG["base_low"] <= ext50 <= CONFIG["base_high"]:
        out["state"] = "눌림/베이스"              # 이격 해소된 좋은 자리
    else:
        out["state"] = "양호"                    # 상승추세, 중간 지대

    # 진입적합도 점수 (0~100): 이격이 적을수록, 변동성 수축일수록, 고점 근처일수록 높음
    sc = 0.0
    if out["state"] in ("눌림/베이스", "양호"):
        # 이격 점수: 0~15% 구간이 이상적
        if not np.isnan(ext50):
            d = abs(ext50 - 7.5)
            sc += max(0.0, 55.0 * (1 - d / 25.0))
        # 변동성 수축 보너스
        if not np.isnan(out["vcp"]) and out["vcp"] < 1.0:
            sc += 25.0 * min(1.0, (1.0 - out["vcp"]) / 0.4)
        # 52주 고점 근접 보너스 (미너비니: 고점 근처에서 베이스)
        if not np.isnan(out["dist_high"]):
            sc += max(0.0, 20.0 * (1 - abs(out["dist_high"]) / 25.0))
    elif out["state"] == "과열":
        sc = 10.0        # 추세는 있으나 진입 부적합
    out["entry_score"] = float(min(100.0, sc))
    return out


# ------------------------------------------------------------------ 스크리닝
def screen(date) -> pd.DataFrame:
    frames = []
    for mkt in CONFIG["markets"]:
        cap = stock.get_market_cap_by_ticker(date, market=mkt); _sleep()
        fund = stock.get_market_fundamental_by_ticker(date, market=mkt); _sleep()
        df = cap.join(fund, how="inner")
        df["시장"] = mkt
        frames.append(df)
    uni = pd.concat(frames)
    uni = uni[uni["시가총액"] >= CONFIG["min_market_cap_krw"]]
    uni = uni[uni["거래대금"] >= CONFIG["min_trading_value_krw"]]
    print(f"  필터 통과: {len(uni)}종목. 종목별 주가/미너비니 지표 수집...")

    start = _nearest_bday(_days_ago(date, CONFIG["history_days"]))
    tech, tickers = {}, list(uni.index)
    for i, t in enumerate(tickers, 1):
        tech[t] = compute_technicals(t, start, date); _sleep()
        if i % 50 == 0:
            print(f"    {i}/{len(tickers)} ...")
    uni = uni.join(pd.DataFrame(tech).T)

    # RS 백분위 (유니버스 전체) → 기준8
    uni["rs"] = _pct_rank(uni["rs_raw"].astype(float))
    uni["c8"] = uni["rs"] >= CONFIG["rs_threshold"]

    # 미너비니 집계
    crit = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    uni["mnv_count"] = uni[crit].sum(axis=1).astype(int)
    uni["mnv_score"] = uni["mnv_count"] / 8 * 100
    uni["mnv_pass"] = uni["mnv_count"] == 8

    # 팩터
    uni["earnings_yield"] = np.where(uni["PER"] > 0, 1.0 / uni["PER"], np.nan)
    uni["book_yield"] = np.where(uni["PBR"] > 0, 1.0 / uni["PBR"], np.nan)
    uni["div_yield"] = uni["DIV"]
    uni["roe_proxy"] = np.where(uni["BPS"] > 0, uni["EPS"] / uni["BPS"] * 100, np.nan)
    uni["z_value"] = (_zscore(uni["earnings_yield"]) + _zscore(uni["book_yield"])
                      + _zscore(uni["div_yield"])) / 3
    uni["z_momentum"] = _zscore(uni["mom"].astype(float))
    uni["z_quality"] = _zscore(uni["roe_proxy"])
    w = CONFIG["weights"]
    uni["composite"] = (w["value"] * uni["z_value"] + w["momentum"] * uni["z_momentum"]
                        + w["quality"] * uni["z_quality"])
    uni["comp_pct"] = _pct_rank(uni["composite"])
    uni["v_disp"] = _z_to_display(uni["z_value"])
    uni["m_disp"] = _z_to_display(uni["z_momentum"])
    uni["q_disp"] = _z_to_display(uni["z_quality"])

    # 종합 시그널 = 팩터 + 미너비니 + 진입적합도
    uni["combined"] = (CONFIG["signal_factor_w"] * uni["comp_pct"]
                       + CONFIG["signal_mnv_w"] * uni["mnv_score"]
                       + CONFIG["signal_entry_w"] * uni["entry_score"].astype(float))
    # 과열 종목은 매수 등급에서 제외 (추세는 유효하나 진입 타이밍 부적합)
    if CONFIG["exclude_hot"]:
        hot = uni["state"] == "과열"
        uni.loc[hot, "combined"] = uni.loc[hot, "combined"].clip(upper=55)
    # 등급: 상대순위(백분위) 기준 — 매일 일정 개수 유지
    rank_pct = uni["combined"].rank(ascending=False, pct=True) * 100  # 1 = 최상위
    p = CONFIG["rating_pcts"]
    uni["rating"] = np.select(
        [rank_pct <= p["강력매수"], rank_pct <= p["매수"],
         rank_pct <= p["중립"], rank_pct <= p["매도"]],
        ["강력매수", "매수", "중립", "매도"],
        default="강력매도")

    sector_map = build_sector_map(date)
    uni["sector"] = uni.index.map(lambda t: sector_map.get(t, "기타/미분류"))
    uni["종목명"] = [stock.get_market_ticker_name(t) for t in uni.index]

    return uni.sort_values("combined", ascending=False)


def to_records(uni) -> list[dict]:
    def f(v, dec):
        return None if (v is None or (isinstance(v, float) and pd.isna(v))) else round(float(v), dec)
    recs = []
    for rank, (ticker, r) in enumerate(uni.iterrows(), 1):
        recs.append({
            "rank": rank, "ticker": ticker, "name": r["종목명"],
            "market": r["시장"], "sector": r["sector"], "rating": r["rating"],
            "combined": f(r["combined"], 1), "composite": f(r["composite"], 3),
            "comp_pct": f(r["comp_pct"], 1),
            "mnv_count": int(r["mnv_count"]), "mnv_score": f(r["mnv_score"], 0),
            "mnv_pass": bool(r["mnv_pass"]),
            "crit": [bool(r[c]) for c in ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]],
            "rs": f(r["rs"], 0), "align": r["align"], "dist_high": f(r["dist_high"], 1),
            "state": r["state"], "entry": f(r["entry_score"], 0),
            "ext50": f(r["ext50"], 1), "ext200": f(r["ext200"], 1),
            "pullback": f(r["pullback"], 1), "vcp": f(r["vcp"], 2),
            "v": f(r["v_disp"], 1), "m": f(r["m_disp"], 1), "q": f(r["q_disp"], 1),
            "per": f(r["PER"], 1), "pbr": f(r["PBR"], 2), "div": f(r["DIV"], 2),
            "roe": f(r["roe_proxy"], 1), "mom": f(r["mom"], 1),
            "cap": f(float(r["시가총액"]) / 1e8, 0),
        })
    return recs


def generate_html(records, meta) -> str:
    return (HTML_TEMPLATE
            .replace("/*__DATA__*/", json.dumps(records, ensure_ascii=False))
            .replace("/*__META__*/", json.dumps(meta, ensure_ascii=False)))


def run():
    date = _recent_bday()
    print(f"[기준일] {date}")
    print("[스크리닝] 코스피·코스닥 전체 (팩터+미너비니, 6~9분 소요)...")
    uni = screen(date)
    records = to_records(uni)
    w = CONFIG["weights"]
    counts = {name: int((uni["rating"] == name).sum()) for name in RATING_NAMES}
    meta = {
        "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "count": len(records), "sectors": len({r["sector"] for r in records}),
        "mnv_pass": int(uni["mnv_pass"].sum()),
        "state_counts": {s: int((uni["state"] == s).sum())
                         for s in ["눌림/베이스", "양호", "과열", "이탈"]},
        "weights": {"가치": w["value"], "모멘텀": w["momentum"], "품질": w["quality"]},
        "rating_counts": counts,
    }
    with open(CONFIG["output_json"], "w", encoding="utf-8") as fp:
        json.dump({"meta": meta, "records": records}, fp, ensure_ascii=False, indent=1)
    with open(CONFIG["output_html"], "w", encoding="utf-8") as fp:
        fp.write(generate_html(records, meta))
    print(f"\n완료. {len(records)}종목 / {meta['sectors']}업종 / 미너비니 통과 ⭐{meta['mnv_pass']}종목")
    print("  등급분포:", "  ".join(f"{k} {v}" for k, v in counts.items()))
    print("  상태분포:", "  ".join(f"{k} {v}" for k, v in meta["state_counts"].items()))
    print(f"  대시보드: {CONFIG['output_html']}  ← 더블클릭")


# ==================================================================== HTML
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KR 팩터 스크리너</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0b0f14; --panel:#141b23; --panel2:#1a222c; --row:#111820; --line:#28323e;
  --ink:#f4f3ef; --muted:#b3bdc9; --faint:#727d8a;
  --gold:#e6b053; --gold-dim:#6f5626;
  --up:#ee5a4d; --down:#4a8ce8;
  --vbar:#7aa3cc; --mbar:#e6b053; --qbar:#78c6a0;
  --r-sb:#ee5a4d; --r-b:#c46a5c; --r-n:#7a8592; --r-s:#5b8dce; --r-ss:#4a8ce8;
  --ok:#5fbf8c; --off:#3a444f;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);
  font-family:Pretendard,-apple-system,BlinkMacSystemFont,sans-serif;
  font-size:14px;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.mono{font-family:'IBM Plex Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:1400px;margin:0 auto;padding:26px 20px 90px}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;flex-wrap:wrap}
.title{font-size:27px;font-weight:800;letter-spacing:-.02em}
.title .k{color:var(--gold)}
.sub{color:var(--muted);font-size:13px;margin-top:5px;font-weight:500}
.wbadge{display:inline-flex;gap:9px;margin-top:7px;font-size:11.5px;color:var(--faint)}
.wbadge b{color:var(--gold);font-weight:600}
.metrics{display:flex;gap:24px;text-align:right}
.metric .n{font-size:23px;font-weight:700}
.metric .n.star{color:var(--gold)}
.metric .l{font-size:11px;color:var(--faint);text-transform:uppercase;letter-spacing:.09em;margin-top:2px}

.chips{display:flex;gap:9px;flex-wrap:wrap;margin:22px 0 4px}
.chip{display:flex;align-items:center;gap:9px;background:var(--panel);border:1px solid var(--line);
  border-radius:10px;padding:10px 15px;cursor:pointer;transition:.12s;min-width:108px}
.chip:hover{border-color:var(--faint)}
.chip.active{border-color:currentColor;background:var(--panel2)}
.chip .rn{font-size:12.5px;font-weight:700}
.chip .rc{font-size:18px;font-weight:700;margin-left:auto}
.chip .sw{width:9px;height:9px;border-radius:50%;background:currentColor}
.chip.all{color:var(--gold)} .chip.sb{color:var(--r-sb)} .chip.b{color:var(--r-b)}
.chip.n{color:var(--r-n)} .chip.s{color:var(--r-s)} .chip.ss{color:var(--r-ss)}
.chip .rn,.chip .rc{color:var(--ink)}

.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:16px 0 14px}
.controls input,.controls select{background:var(--panel2);border:1px solid var(--line);
  color:var(--ink);border-radius:9px;padding:10px 13px;font-size:13px;font-family:inherit;font-weight:500}
.controls input{min-width:200px}
.controls input::placeholder{color:var(--faint)}
.controls input:focus,.controls select:focus{outline:2px solid var(--gold-dim);border-color:var(--gold)}
.toggle{display:inline-flex;align-items:center;gap:8px;color:var(--muted);cursor:pointer;
  user-select:none;padding:9px 13px;border:1px solid var(--line);border-radius:9px;
  background:var(--panel2);font-size:13px;font-weight:500}
.toggle.on{color:var(--gold);border-color:var(--gold-dim)}
.count{margin-left:auto;color:var(--muted);font-size:12.5px;font-weight:500}

.tablecard{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden}
table{width:100%;border-collapse:collapse}
thead th{position:sticky;top:0;z-index:2;background:var(--panel2);color:var(--muted);
  font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;
  text-align:right;padding:13px 10px;border-bottom:1px solid var(--line);cursor:pointer;white-space:nowrap}
thead th.l{text-align:left}
thead th:hover{color:var(--ink)}
thead th .ar{color:var(--gold);font-size:9px;margin-left:3px}
tbody td{padding:12px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;font-weight:500}
tbody tr:hover{background:var(--row)}
tbody tr:last-child td{border-bottom:none}
tbody td.edge{border-left:3px solid transparent;padding-left:13px}
tr.sb td.edge{border-left-color:var(--r-sb)} tr.b td.edge{border-left-color:var(--r-b)}
tr.n td.edge{border-left-color:var(--r-n)} tr.s td.edge{border-left-color:var(--r-s)}
tr.ss td.edge{border-left-color:var(--r-ss)}

.rk{color:var(--faint);width:34px}
.nm{text-align:left}
.nm .n1{font-weight:700;color:var(--ink)}
.nm .star{color:var(--gold);margin-left:5px}
.nm .n2{color:var(--faint);font-size:11px}
.mk{font-size:9.5px;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:1px 5px;margin-left:6px}
.sec{text-align:left;color:var(--muted);font-size:12px;max-width:112px;overflow:hidden;text-overflow:ellipsis}

.badge{display:inline-block;font-size:12px;font-weight:700;padding:4px 11px;border-radius:999px;color:#fff}
.badge.sb{background:var(--r-sb)} .badge.b{background:var(--r-b)}
.badge.n{background:var(--r-n)} .badge.s{background:var(--r-s)} .badge.ss{background:var(--r-ss)}

/* 미너비니 셀 */
.mnv{display:inline-flex;align-items:center;gap:9px;justify-content:flex-end}
.mnv .frac{font-weight:700;width:30px;text-align:right}
.mnv .frac.full{color:var(--gold)}
.dots8{display:inline-grid;grid-template-columns:repeat(8,1fr);gap:3px}
.dots8 i{width:6px;height:6px;border-radius:1.5px;background:var(--off)}
.dots8 i.on{background:var(--ok)}
.rs{font-weight:600}
.rs.hi{color:var(--gold)}
.st{font-size:11px;font-weight:700;padding:3px 9px;border-radius:6px;white-space:nowrap}
.st.base{color:#5fbf8c;background:rgba(95,191,140,.15)}
.st.ok{color:var(--muted);background:var(--panel2)}
.st.hot{color:#ee5a4d;background:rgba(238,90,77,.15)}
.st.brk{color:#4a8ce8;background:rgba(74,140,232,.13)}
.st.nd{color:var(--faint)}
.entry{font-weight:700}
.entry.hi{color:#5fbf8c}.entry.mid{color:var(--muted)}.entry.lo{color:var(--faint)}
.ext.hot{color:#ee5a4d;font-weight:700}
.ext.good{color:#5fbf8c;font-weight:600}

.comp{display:flex;align-items:center;gap:9px;justify-content:flex-end;min-width:108px}
.comp .v{color:var(--gold);font-weight:700;width:40px;text-align:right}
.compbar{width:52px;height:6px;background:var(--panel2);border-radius:3px;overflow:hidden}
.compbar>i{display:block;height:100%;background:linear-gradient(90deg,var(--gold-dim),var(--gold))}
.finger{display:inline-flex;gap:5px;align-items:flex-end;height:22px}
.finger .fb{width:7px;background:var(--panel2);border-radius:2px;position:relative;height:100%}
.finger .fb>i{position:absolute;bottom:0;left:0;right:0;border-radius:2px}
.finger .fb.v>i{background:var(--vbar)} .finger .fb.m{width:9px} .finger .fb.m>i{background:var(--mbar)}
.finger .fb.q>i{background:var(--qbar)}
.fhead{display:inline-flex;gap:5px;font-size:9px;color:var(--faint)}
.fhead span{width:7px;text-align:center}.fhead span.m{width:9px;color:var(--gold)}
.align{font-size:10.5px;font-weight:700;padding:2px 7px;border-radius:5px}
.align.정배열{color:var(--up);background:rgba(238,90,77,.13)}
.align.역배열{color:var(--down);background:rgba(74,140,232,.13)}
.align.혼조{color:var(--muted);background:var(--panel2)}
.align.데이터없음{color:var(--faint);background:transparent}
.up{color:var(--up)} .down{color:var(--down)} .na{color:var(--faint)}
footer{margin-top:24px;color:var(--faint);font-size:11.5px;line-height:1.75}
footer b{color:var(--muted)}
@media(max-width:1080px){.hide-sm{display:none}.wrap{padding:18px 12px 60px}.title{font-size:21px}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <div class="title"><span class="k">KR</span> 팩터 스크리너</div>
      <div class="sub">코스피·코스닥 전체 · 모멘텀 틸트 3팩터 + 미너비니 추세 시그널 · <span id="mdate"></span> 기준</div>
      <div class="wbadge" id="wbadge"></div>
    </div>
    <div class="metrics">
      <div class="metric"><div class="n mono" id="mcount">–</div><div class="l">종목</div></div>
      <div class="metric"><div class="n mono star" id="mstar">–</div><div class="l">⭐미너비니</div></div>
      <div class="metric"><div class="n mono" id="msec">–</div><div class="l">업종</div></div>
    </div>
  </header>

  <div class="chips" id="chips"></div>

  <div class="controls">
    <input id="q" placeholder="종목명·코드 검색">
    <select id="sector"><option value="">전체 업종</option></select>
    <select id="market"><option value="">전체 시장</option><option>KOSPI</option><option>KOSDAQ</option></select>
    <div class="toggle" id="mnvToggle"><span>⭐ 미너비니 통과만</span></div>
    <div class="toggle" id="hotToggle"><span>🔥 과열 제외</span></div>
    <div class="toggle" id="baseToggle"><span>눌림/베이스만</span></div>
    <div class="count" id="count"></div>
  </div>

  <div class="tablecard">
    <table>
      <thead><tr>
        <th class="l edge" data-k="rank">#</th>
        <th class="l" data-k="name">종목</th>
        <th class="l" data-k="rating">시그널</th>
        <th data-k="combined">종합 <span class="ar">▼</span></th>
        <th data-k="state">상태</th>
        <th data-k="entry">진입적합</th>
        <th data-k="ext50">50일선이격</th>
        <th data-k="mnv_count">미너비니 8기준</th>
        <th data-k="rs">RS</th>
        <th data-k="composite" class="hide-sm">팩터점수</th>
        <th class="hide-sm"><div class="fhead"><span class="v">가</span><span class="m">모</span><span class="q">품</span></div></th>
        <th data-k="align" class="hide-sm">배열</th>
        <th data-k="mom">12-1모멘텀</th>
        <th data-k="per" class="hide-sm">PER</th>
        <th data-k="pbr" class="hide-sm">PBR</th>
        <th data-k="div" class="hide-sm">배당%</th>
        <th data-k="cap" class="hide-sm">시총(억)</th>
      </tr></thead>
      <tbody id="tb"></tbody>
    </table>
  </div>

  <footer>
    <div><b>미너비니 8기준</b> — ①주가&gt;50일선 ②&gt;150일선 ③&gt;200일선 ④50&gt;150 ⑤150&gt;200 ⑥200일선 1개월 상승 ⑦52주고점 -25% 이내 ⑧RS≥70. 점 8개가 각 기준 통과 여부. ⭐=8개 전부 통과.</div>
    <div><b>상태 분류</b> — 🔥과열: 50일선 +25%↑ 또는 200일선 +60%↑ 이격(추세는 유효하나 진입 늦음, 기본 제외). 눌림/베이스: 50일선 -5%~+15%로 이격 해소된 자리. 이탈: 200일선 아래 등 추세 훼손. <b>진입적합</b>은 이격 해소·변동성 수축(VCP)·52주 고점 근접을 0~100으로 정량화.</div>
    <div><b>등급은 상대순위 기준</b> — 강력매수 상위 6%(약 30종목)로, 백테스트가 실제 매수·검증한 포트폴리오 규모와 동일합니다. 매일 개수가 일정하게 유지됩니다.</div>
    <div><b>종합 시그널 = 팩터점수 55% + 미너비니 45%</b> (백테스트 검증 결과 최적 조합: CAGR 42.6%, Sharpe 1.36, MDD -23.4%). 미너비니 추세필터는 팩터 단독 대비 수익 +7.3%p·낙폭 8.8%p 개선.</div>
    <div><b>상태(눌림/과열)는 참고 정보</b> — 백테스트상 과열 종목 제외는 수익을 12.8%p 악화시켜 점수에 반영하지 않습니다. 필요시 상단 필터로 직접 적용 가능.</div>
    <div><b>주의</b> — 시그널 등급은 기계적 모델 출력이며 매매 추천이 아닙니다. 개별 종목 정성 분석이 반드시 병행되어야 합니다.</div>
    <div id="gen"></div>
  </footer>
</div>

<script>
const META = /*__META__*/;
const DATA = /*__DATA__*/;
const RCLS = {"강력매수":"sb","매수":"b","중립":"n","매도":"s","강력매도":"ss"};
const CRITNAME=["주가>50일선","주가>150일선","주가>200일선","50>150","150>200","200일선 1개월 상승","52주고점 -25% 이내","RS≥70"];

document.getElementById('mdate').textContent = META.date;
document.getElementById('mcount').textContent = META.count.toLocaleString();
document.getElementById('mstar').textContent = META.mnv_pass;
document.getElementById('msec').textContent = META.sectors;
document.getElementById('gen').textContent = '생성: ' + META.generated;
document.getElementById('wbadge').innerHTML =
  '가중치 &nbsp;가치 <b>'+META.weights['가치']+'</b> · 모멘텀 <b>'+META.weights['모멘텀']+'</b> · 품질 <b>'+META.weights['품질']+'</b>';

let ratingFilter='',mnvOnly=false,hotExclude=false,baseOnly=false;
const chipDefs=[['','전체','all',META.count],
  ['강력매수','강력매수','sb',META.rating_counts['강력매수']],
  ['매수','매수','b',META.rating_counts['매수']],
  ['중립','중립','n',META.rating_counts['중립']],
  ['매도','매도','s',META.rating_counts['매도']],
  ['강력매도','강력매도','ss',META.rating_counts['강력매도']]];
const chipsEl=document.getElementById('chips');
chipDefs.forEach(([val,label,cls,cnt])=>{
  const d=document.createElement('div');
  d.className='chip '+cls+(val===''?' active':'');
  d.innerHTML='<span class="sw"></span><span class="rn">'+label+'</span><span class="rc mono">'+(cnt||0)+'</span>';
  d.onclick=()=>{ratingFilter=val;document.querySelectorAll('.chip').forEach(c=>c.classList.remove('active'));d.classList.add('active');render();};
  chipsEl.appendChild(d);
});
[...new Set(DATA.map(d=>d.sector))].sort().forEach(s=>{
  const o=document.createElement('option');o.value=s;o.textContent=s;document.getElementById('sector').appendChild(o);
});

let sortKey='combined',sortDir=-1;
function fmt(v,d){return v==null?'<span class="na">–</span>':Number(v).toFixed(d);}
function bar(p){return '<div class="compbar"><i style="width:'+Math.max(2,p||0)+'%"></i></div>';}
function finger(d){const b=(c,v)=>'<span class="fb '+c+'"><i style="height:'+Math.max(3,v||0)+'%"></i></span>';
  return '<span class="finger">'+b('v',d.v)+b('m',d.m)+b('q',d.q)+'</span>';}
function momCell(v){if(v==null)return '<span class="na">–</span>';
  return '<span class="'+(v>=0?'up':'down')+'">'+(v>=0?'+':'')+v.toFixed(1)+'%</span>';}
function mnvCell(d){
  const dots=d.crit.map((x,i)=>'<i class="'+(x?'on':'')+'" title="'+CRITNAME[i]+(x?' ✓':' ✗')+'"></i>').join('');
  const full=d.mnv_count===8?' full':'';
  return '<span class="mnv"><span class="frac mono'+full+'">'+d.mnv_count+'/8</span><span class="dots8">'+dots+'</span></span>';
}
function rsCell(v){if(v==null)return '<span class="na">–</span>';
  return '<span class="rs mono'+(v>=70?' hi':'')+'">'+v.toFixed(0)+'</span>';}
const SCLS={"눌림/베이스":"base","양호":"ok","과열":"hot","이탈":"brk","데이터없음":"nd"};
function stateCell(s){return '<span class="st '+(SCLS[s]||'nd')+'">'+(s==="과열"?"🔥 과열":s)+'</span>';}
function entryCell(v){if(v==null)return '<span class="na">–</span>';
  const c=v>=70?'hi':(v>=45?'mid':'lo');
  return '<span class="entry '+c+'">'+v.toFixed(0)+'</span>';}
function extCell(v){if(v==null)return '<span class="na">–</span>';
  const c=v>25?'hot':(v>=-5&&v<=15?'good':'');
  return '<span class="ext '+c+'">'+(v>=0?'+':'')+v.toFixed(1)+'%</span>';}

function render(){
  const q=document.getElementById('q').value.trim().toLowerCase();
  const sec=document.getElementById('sector').value;
  const mk=document.getElementById('market').value;
  let rows=DATA.filter(d=>{
    if(ratingFilter && d.rating!==ratingFilter) return false;
    if(mnvOnly && !d.mnv_pass) return false;
    if(hotExclude && d.state==='과열') return false;
    if(baseOnly && d.state!=='눌림/베이스') return false;
    if(sec && d.sector!==sec) return false;
    if(mk && d.market!==mk) return false;
    if(q && !(d.name.toLowerCase().includes(q)||d.ticker.includes(q))) return false;
    return true;
  });
  rows.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];
    if(x==null)x=-1e9;if(y==null)y=-1e9;
    if(typeof x==='string')return sortDir*x.localeCompare(y);return sortDir*(x-y);});
  document.getElementById('count').textContent=rows.length.toLocaleString()+'종목';
  document.getElementById('tb').innerHTML=rows.map(d=>{
    const rc=RCLS[d.rating]||'n';
    return `<tr class="${rc}">
      <td class="rk mono edge">${d.rank}</td>
      <td class="nm"><span class="n1">${d.name}</span>${d.mnv_pass?'<span class="star">★</span>':''}<span class="mk">${d.market}</span><br><span class="n2 mono">${d.ticker}</span></td>
      <td><span class="badge ${rc}">${d.rating}</span></td>
      <td class="mono"><b style="color:var(--ink)">${d.combined!=null?d.combined.toFixed(0):'–'}</b></td>
      <td>${stateCell(d.state)}</td>
      <td class="mono">${entryCell(d.entry)}</td>
      <td class="mono">${extCell(d.ext50)}</td>
      <td>${mnvCell(d)}</td>
      <td>${rsCell(d.rs)}</td>
      <td class="hide-sm"><div class="comp"><span class="v mono">${d.composite!=null?d.composite.toFixed(2):'–'}</span>${bar(d.comp_pct)}</div></td>
      <td class="hide-sm">${finger(d)}</td>
      <td class="hide-sm"><span class="align ${d.align}">${d.align}</span></td>
      <td class="mono">${momCell(d.mom)}</td>
      <td class="mono hide-sm">${fmt(d.per,1)}</td>
      <td class="mono hide-sm">${fmt(d.pbr,2)}</td>
      <td class="mono hide-sm">${fmt(d.div,2)}</td>
      <td class="mono hide-sm">${d.cap?d.cap.toLocaleString():'<span class=na>–</span>'}</td>
    </tr>`;}).join('');
}
document.querySelectorAll('thead th[data-k]').forEach(th=>{
  th.onclick=()=>{const k=th.dataset.k;
    if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=(k==='rank'||k==='name'||k==='rating'||k==='align')?1:-1;}
    document.querySelectorAll('thead th .ar').forEach(a=>a.remove());
    const ar=document.createElement('span');ar.className='ar';ar.textContent=sortDir<0?'▼':'▲';th.appendChild(ar);
    render();};
});
document.getElementById('q').oninput=render;
document.getElementById('sector').onchange=render;
document.getElementById('market').onchange=render;
document.getElementById('mnvToggle').onclick=function(){mnvOnly=!mnvOnly;this.classList.toggle('on',mnvOnly);render();};
document.getElementById('hotToggle').onclick=function(){hotExclude=!hotExclude;this.classList.toggle('on',hotExclude);render();};
document.getElementById('baseToggle').onclick=function(){baseOnly=!baseOnly;this.classList.toggle('on',baseOnly);render();};
render();
</script>
</body>
</html>"""


if __name__ == "__main__":
    run()
