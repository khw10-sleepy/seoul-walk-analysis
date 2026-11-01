"""
서울 시민은 어떤 길을 '걷기 좋다'고 느낄까?
- 도로 요인과 보행환경 만족도의 관계 분석 (2026 데이터분석 연구발표회)

y (결과) = 그 길을 걷던 시민 20명이 매긴 '보행환경 만족도' 평균 (1~5점)
x (요인) = 보도너비, 차선 수, 횡단보도, 상업지역, CCTV 수 등
방법     = 데이터 점검 → 그래프 탐색 → 다중 선형회귀 (statsmodels)
실행     : streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="서울 보행 만족도 분석", page_icon="🚶", layout="wide")

# =========================================================
# 0. 데이터 불러오기
# =========================================================
@st.cache_data
def load():
    d = pd.read_csv("points.csv", dtype={"구코드": str})
    gu = {"11010": "종로구", "11020": "중구", "11030": "용산구", "11040": "성동구", "11050": "광진구",
          "11060": "동대문구", "11070": "중랑구", "11080": "성북구", "11090": "강북구", "11100": "도봉구",
          "11110": "노원구", "11120": "은평구", "11130": "서대문구", "11140": "마포구", "11150": "양천구",
          "11160": "강서구", "11170": "구로구", "11180": "금천구", "11190": "영등포구", "11200": "동작구",
          "11210": "관악구", "11220": "서초구", "11230": "강남구", "11240": "송파구", "11250": "강동구"}
    d["구"] = d["구코드"].str[:5].map(gu)
    return d

df = load()

# 요인 목록: 컬럼 → (보기 이름, '1만큼 늘면' 설명, 종류)
FACTORS = {
    "보도너비_m":     ("보도 너비(m)", "보도가 1m 넓어지면", "num"),
    "차선수":         ("차선 수", "차선이 1개 늘면", "num"),
    "CCTV_100m":      ("CCTV 수 (반경 100m)", "CCTV가 1대 늘면", "num"),
    "계단_100m":      ("계단 수 (반경 100m)", "계단이 1곳 늘면", "num"),
    "교차로신호_100m": ("신호 교차로 수 (반경 100m)", "교차로가 1곳 늘면", "num"),
    "상업지역":       ("상업지역", "상업지역이면", "bin"),
    "보도전용":       ("보도전용 길 (차 없음)", "보도전용 길이면", "bin"),
    "횡단보도":       ("횡단보도 있음", "횡단보도가 있으면", "bin"),
    "펜스":           ("보도·차도 분리 펜스", "펜스가 있으면", "bin"),
    "버스정류장":     ("버스정류장·지하철 근처", "정류장이 가까우면", "bin"),
    "버스전용차로":   ("버스전용차로", "버스전용차로가 있으면", "bin"),
    "장애물":         ("보도 장애물", "장애물이 있으면", "bin"),
    "점자블록":       ("점자블록", "점자블록이 있으면", "bin"),
    "경사로":         ("경사로", "경사로가 있으면", "bin"),
}
DEFAULT = ["보도너비_m", "차선수", "CCTV_100m", "계단_100m", "상업지역", "보도전용", "횡단보도", "펜스"]
NAME = {c: v[0] for c, v in FACTORS.items()}

# =========================================================
# 사이드바: 체크리스트 (회귀분석에 넣을 요인 고르기)
# =========================================================
st.sidebar.header("✅ 분석에 넣을 요인")
st.sidebar.caption("체크한 요인만 회귀분석(④ 탭)에 들어가요")
chosen = [c for c in FACTORS if st.sidebar.checkbox(NAME[c], value=c in DEFAULT, key=c)]
st.sidebar.divider()
use_gu = st.sidebar.toggle("동네(구) 차이 빼고 보기", value=True,
                           help="구마다 다른 분위기를 따로 떼어 내고, '같은 구 안에서' 길 특징의 효과만 봐요.")
alpha = st.sidebar.select_slider("'의미 있음' 기준 (p값)", [0.01, 0.05, 0.10], value=0.05)

# =========================================================
# 회귀분석 함수 (④, ⑤ 탭에서 같이 사용)
# =========================================================
def run_ols(cols, gu=True):
    data = df.dropna(subset=cols + ["만족도"]).copy()
    X = data[cols].astype(float)
    if gu:
        X = pd.concat([X, pd.get_dummies(data["구"], prefix="구", drop_first=True, dtype=float)], axis=1)
    m = sm.OLS(data["만족도"], sm.add_constant(X)).fit()
    beta = m.params[cols] * data[cols].std() / data["만족도"].std()   # 표준화계수
    res = pd.DataFrame({"요인": [NAME[c] for c in cols], "계수": m.params[cols].values,
                        "p값": m.pvalues[cols].values, "표준화계수": beta.values}, index=cols)
    return m, res

# =========================================================
# 화면
# =========================================================
st.title("🚶 서울 시민은 어떤 길을 '걷기 좋다'고 느낄까?")
st.caption("도로 요인과 보행환경 만족도의 관계 분석 · 데이터: 서울 열린데이터광장 외 공공데이터")

k1, k2, k3, k4 = st.columns(4)
k1.metric("조사 지점", f"{len(df):,}곳")
k2.metric("설문 응답자", f"{int(df['응답자수'].sum()):,}명")
k3.metric("평균 만족도", f"{df['만족도'].mean():.2f}점 / 5점")
k4.metric("분석 요인 후보", f"{len(FACTORS)}개")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["① 연구 소개", "② 데이터 점검", "③ 탐색 (그래프)", "④ 회귀분석", "⑤ 결론"])

# ---------------------------------------------------------
with tab1:
    st.header("연구 질문")
    st.markdown("""
> **보도 너비, 차선 수, 횡단보도, 상업지역, CCTV 같은 도로의 특징은
> 시민이 느끼는 '걷기 만족도'와 어떤 관계가 있을까?**

길 찾기 앱은 보통 '가장 빠른 길'만 알려 줘요. 하지만 사람들은 **걷기 좋은 길**을 고르기도 해요.
어떤 길이 걷기 좋은지 데이터로 알 수 있다면, 길 안내나 도시 계획에 쓸 수 있어요.
""")
    st.subheader("가설")
    st.markdown("""
1. 보도가 **넓을수록** 만족도가 높을 것이다.
2. 차선이 **많은**(차가 많은) 길일수록 만족도가 낮을 것이다.
3. 가게가 많은 **상업지역**은 만족도가 높을 것이다.
4. **CCTV**가 많은 길은 안전하게 느껴져 만족도가 높을 것이다.
""")
    st.subheader("사용한 데이터")
    st.dataframe(pd.DataFrame([
        ["서울시 유동인구 조사 – 속성조사 (2015)", "서울 열린데이터광장 OA-21706", "시민 2만 명의 보행환경 만족도 (y)"],
        ["서울시 유동인구 조사 – 조사지점정보 (2015)", "서울 열린데이터광장 OA-21707", "지점 위치, 보도너비, 차선 수, 횡단보도 등 (x)"],
        ["CCTV 설치 현황 (서울 25개 구)", "공공데이터포털", "지점 반경 100m 안 CCTV 대수 (x)"],
        ["계단 위치", "OpenStreetMap", "지점 반경 100m 안 계단 수 (x)"],
        ["교차로 정보 (신호정보)", "공공데이터포털 (한국지방행정연구원)", "지점 반경 100m 안 신호 교차로 수 (x)"],
    ], columns=["데이터", "출처", "이 연구에서 쓴 곳"]), hide_index=True, width="stretch")
    st.subheader("분석 방법")
    st.markdown("""
1. **데이터 점검** — 빈 값·이상한 값을 찾고 처리 기준을 정했어요.
2. **탐색** — 지도와 그래프로 만족도가 어디서 높고 낮은지 살펴봤어요.
3. **다중 선형회귀** — 여러 요인을 한꺼번에 넣고 각 요인이 만족도와 얼마나 관계있는지 계산했어요.
   `만족도 = b0 + b1×보도너비 + b2×차선수 + … + (구별 차이)`
""")

# ---------------------------------------------------------
with tab2:
    st.header("분석 전에 데이터 점검하기")
    st.markdown("AI에게 맡기기 전에 원본 파일을 직접 열어 보고 찾은 문제와 처리 기준이에요.")
    st.dataframe(pd.DataFrame([
        ["만족도가 글자로 적혀 있음 (매우불만족~매우만족)", "20,000명 전부", "1~5점 숫자로 바꿈 → 지점별 평균"],
        ["조사지점 파일에 같은 지점이 두 번 적힌 줄", "1,500줄 중 200줄", "지점코드 기준으로 한 줄만 남김"],
        ["조사지점 1,300곳 중 설문이 있는 곳", "1,000곳", "설문이 있는 1,000곳만 사용"],
        ["좌표가 비어 있는 지점", "4곳", "지도·CCTV 계산이 불가능해 제외 → 996곳"],
        ["용도구분(상업지역 여부)이 비어 있는 지점", "40곳", "'상업지역'을 넣을 때만 그 지점 제외"],
        ["'버스정류장'과 '지하철' 칸이 거의 똑같이 적힘", "98.6% 일치", "입력 오류로 보고 하나(버스정류장·지하철 근처)로 합침"],
        ["좌표가 위도·경도가 아닌 TM좌표(EPSG:5181)", "전체", "위도·경도로 변환 (안국역·서울역 위치로 확인)"],
        ["설문은 2015년, CCTV·교차로는 최근 자료", "전체", "결과 해석 시 한계로 밝힘"],
    ], columns=["찾은 문제", "규모", "처리 기준"]), hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("지점별 평균 만족도 분포")
        fig = px.histogram(df, x="만족도", nbins=30, labels={"만족도": "평균 만족도 (1~5점)"})
        fig.add_vline(x=df["만족도"].mean(), line_dash="dash", annotation_text=f"평균 {df['만족도'].mean():.2f}")
        fig.update_layout(yaxis_title="지점 수", height=360)
        st.plotly_chart(fig, width="stretch")
        st.caption("대부분 2~3.5점 사이. 3점(보통)보다 낮은 곳이 더 많아요.")
    with c2:
        st.subheader("빈 값 개수")
        na = df[list(FACTORS) + ["만족도"]].isna().sum().rename("빈 값").to_frame()
        na.index = [NAME.get(i, i) for i in na.index]
        st.dataframe(na, width="stretch", height=360)
    with st.expander("🗂️ 정리된 데이터 전체 보기 / 내려받기"):
        st.dataframe(df, width="stretch")
        st.download_button("CSV 내려받기", df.to_csv(index=False).encode("utf-8-sig"), "points.csv")

# ---------------------------------------------------------
with tab3:
    st.header("그래프로 살펴보기")
    st.subheader("📍 지점별 만족도 지도")
    def to_color(v):  # 1.5점 빨강 → 2.75점 노랑 → 4점 초록
        t = min(max((v - 1.5) / 2.5, 0), 1)
        return [int(255 * min(1, 2 - 2 * t)), int(220 * min(1, 2 * t)), 60, 210]
    mdf = df.assign(color=df["만족도"].apply(to_color), 점수=df["만족도"].round(2))
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=37.555, longitude=126.99, zoom=10.2),
        layers=[pdk.Layer("ScatterplotLayer", mdf, get_position="[lng, lat]",
                          get_fill_color="color", get_radius=170, pickable=True)],
        tooltip={"text": "{지점명} ({구})\n만족도 {점수}점"},
        map_style=None,
    ))
    st.caption("🟥 불만족 ↔ 🟩 만족. 점 하나 = 조사 지점 1곳 (시민 20명의 평균)")

    st.subheader("🏘️ 구별 평균 만족도")
    g = df.groupby("구")["만족도"].mean().sort_values().reset_index()
    fig = px.bar(g, x="만족도", y="구", orientation="h", height=620, text_auto=".2f",
                 color="만족도", color_continuous_scale="RdYlGn", range_color=[2.2, 3.2])
    fig.add_vline(x=df["만족도"].mean(), line_dash="dash")
    fig.update_layout(coloraxis_showscale=False, xaxis_range=[0, 5], xaxis_title="평균 만족도 (점, 5점 만점)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"가장 높은 구: {g.iloc[-1]['구']} ({g.iloc[-1]['만족도']:.2f}) · "
               f"가장 낮은 구: {g.iloc[0]['구']} ({g.iloc[0]['만족도']:.2f}). "
               "동네마다 차이가 있어서 회귀분석에서 '구 차이'를 따로 떼어 냈어요.")

    st.subheader("🔍 요인 하나씩 비교하기")
    f = st.selectbox("요인 고르기", list(FACTORS), format_func=lambda c: NAME[c])
    if FACTORS[f][2] == "bin":
        t = df.dropna(subset=[f]).assign(그룹=lambda x: x[f].map({1.0: "있음/예", 0.0: "없음/아니오"}))
        s = t.groupby("그룹")["만족도"].agg(평균만족도="mean", 지점수="count").reset_index()
        fig = px.bar(s, x="그룹", y="평균만족도", text_auto=".2f", height=380, hover_data=["지점수"])
        fig.update_layout(yaxis_range=[0, 5], yaxis_title="평균 만족도 (점, 5점 만점)")
        st.plotly_chart(fig, width="stretch")
        d_ = s.set_index("그룹")["평균만족도"]
        st.caption(f"차이: {abs(d_.iloc[0]-d_.iloc[1]):.2f}점. 세로축은 0부터 시작해요(실제 차이 크기 그대로).")
    else:
        fig = px.scatter(df, x=f, y="만족도", trendline="ols", opacity=0.4, height=420,
                         labels={f: NAME[f], "만족도": "평균 만족도"})
        st.plotly_chart(fig, width="stretch")
        r = df[[f, "만족도"]].corr().iloc[0, 1]
        st.caption(f"상관계수 r = {r:.3f} (−1~1, 0에 가까울수록 관계가 약함)")

# ---------------------------------------------------------
with tab4:
    st.header("다중 선형회귀 결과")
    st.markdown("왼쪽 사이드바에서 **요인을 체크/해제**하면 결과가 바로 바뀌어요.")
    if not chosen:
        st.info("왼쪽에서 요인을 하나 이상 골라 주세요.")
    else:
        model, res = run_ols(chosen, use_gu)
        res["의미 있음?"] = np.where(res["p값"] < alpha, "✅ 의미 있음", "— 우연일 수 있음")
        m1, m2, m3 = st.columns(3)
        m1.metric("R² (설명력)", f"{model.rsquared:.3f}", help="만족도 차이 중 몇 %를 이 모델이 설명하는지 (0~1)")
        m2.metric("분석한 지점", f"{int(model.nobs)}곳")
        m3.metric("의미 있는 요인", f"{(res['p값'] < alpha).sum()}개 / {len(res)}개")

        st.subheader("요인별 영향 크기 (표준화계수)")
        fig = px.bar(res.sort_values("표준화계수"), x="표준화계수", y="요인", orientation="h",
                     color="의미 있음?", height=80 + 40 * len(res),
                     color_discrete_map={"✅ 의미 있음": "#2e7d32", "— 우연일 수 있음": "#bdbdbd"})
        fig.add_vline(x=0)
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, width="stretch")
        st.caption("오른쪽(+) = 만족도가 높아지는 쪽, 왼쪽(−) = 낮아지는 쪽. 단위가 다른 요인끼리 비교하려고 표준화했어요.")

        st.subheader("숫자로 보기")
        st.dataframe(res[["요인", "계수", "p값", "의미 있음?"]]
                     .style.format({"계수": "{:+.3f}", "p값": "{:.3f}"}),
                     hide_index=True, width="stretch")

        st.subheader("🗣️ 쉬운 말로 풀면")
        sig = res[res["p값"] < alpha].sort_values("표준화계수", key=abs, ascending=False)
        if sig.empty:
            st.write("고른 요인 중에는 만족도와 뚜렷한 관계가 있는 것이 없어요.")
        for c, r in sig.iterrows():
            word = "높아요" if r["계수"] > 0 else "낮아요"
            st.write(f"- **{FACTORS[c][1]}** 만족도가 평균 **{abs(r['계수']):.3f}점 {word}** (p = {r['p값']:.3f})")
        st.write(f"- 이 모델은 지점마다 다른 만족도의 **{model.rsquared*100:.1f} 퍼센트**를 설명해요.")

        with st.expander("🔬 전체 통계표 (statsmodels 원본 출력)"):
            st.text(model.summary().as_text())

# ---------------------------------------------------------
with tab5:
    st.header("결론")
    m_base, r_base = run_ols(DEFAULT, gu=False)
    m_gu, r_gu = run_ols(DEFAULT, gu=True)
    st.markdown(f"""
### 알게 된 것
1. **상업지역은 걷기 만족도가 높았어요.** 같은 구 안에서도 상업지역이면 평균 **{r_gu.loc['상업지역','계수']:+.2f}점** (p = {r_gu.loc['상업지역','p값']:.3f}).
   → 가설 3 ✅ 지지. 볼거리·가게가 걷는 즐거움을 주는 것으로 보여요.
2. **보도가 넓을수록 만족도가 오히려 조금 낮았어요.** 1m당 **{r_gu.loc['보도너비_m','계수']:+.3f}점** (p = {r_gu.loc['보도너비_m','p값']:.3f}).
   → 가설 1 ❌. 넓은 보도는 대개 차 많고 시끄러운 **큰길 옆**에 있어서, 보도 자체보다 '큰길'이 만족도를 낮추는 것으로 추정해요.
3. **차선 수, CCTV, 계단, 횡단보도, 펜스는 뚜렷한 관계가 없었어요.** → 가설 2·4는 이 데이터로는 확인되지 않았어요.
4. **동네(구) 차이가 길 모양보다 컸어요.** 길 특징만 넣으면 R² = **{m_base.rsquared:.3f}**,
   구 차이를 더하면 R² = **{m_gu.rsquared:.3f}**. 같은 서울이라도 어느 동네냐가 더 중요했어요.

### 한계
- **연관 관계이지 원인이 아니에요.** 회귀분석은 "함께 나타난다"까지만 보여 줘요.
- 설문은 **2015년**, CCTV·교차로는 **최근** 자료라 시점이 달라요.
- 만족도는 날씨·기분·소음·풍경처럼 이 데이터에 없는 것들의 영향도 받아요. R²가 낮은 이유예요.

### 활용과 다음 연구
- 길 안내 앱에 '빠른 길' 말고 **'걷기 좋은 길'** 옵션을 넣을 때 상업지역·큰길 여부를 쓸 수 있어요.
- 소음(S-DoT), 녹지, 경사도 같은 요인을 더하면 설명력을 높일 수 있을 거예요.
""")
