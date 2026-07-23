# Automotive Design Studio Workspace v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Completely rewrite app.py from a data-statistics dashboard into an Automotive Design Studio Workspace with Vehicle Visual Studio, compact radar overview, and Design Space Map (2D scatter + density contours + red star marker).

**Architecture:** Single-file `app.py` (~650 lines) with 5 logical blocks: constants, data layer (50 clustered competitors), scoring engine (preserved from v1), chart factory (compact radar + design space map), and UI rendering (sidebar + vehicle studio + design space tabs). All state in Streamlit session.

**Tech Stack:** Python 3.10+, Streamlit, Pandas, NumPy, Plotly, SciPy

---

### Task 1: Constants & Data Layer — 50 Clustered Competitors

**Files:**
- Overwrite: `app.py`

- [ ] **Step 1: Write complete app.py with constants and data layer**

Write `app.py`:

```python
"""
Automotive Design Studio Workspace — v2
高级汽车设计专家工作台
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Automotive Design Studio",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# 区块 0: 常量定义
# ═══════════════════════════════════════════════════════════════

DIMENSION_MAP = {
    "比例":     ["轮高车高比_pct", "轴长比_pct"],
    "姿态":     ["窗身比_pct", "窗台线夹角_deg"],
    "型面":     ["曲率平滑度", "特征线连续性"],
    "细节品质": ["家族特征匹配度", "间隙段差_mm"],
    "工程平衡": ["头部空间_mm", "视野下沿角_deg"],
}

DIMENSION_ICONS = {
    "比例": "📐 比例", "姿态": "🚙 姿态", "型面": "✨ 型面",
    "细节品质": "💎 细节品质", "工程平衡": "⚙️ 工程平衡",
}

STYLE_PREFERENCES = {
    "运动": {"窗身比_pct": "low", "窗台线夹角_deg": "low", "轮高车高比_pct": "high"},
    "豪华": {"窗身比_pct": "high", "头部空间_mm": "high", "间隙段差_mm": "low", "视野下沿角_deg": "high"},
    "科技": {"曲率平滑度": "high", "特征线连续性": "high", "间隙段差_mm": "low", "轴长比_pct": "high"},
}

# 所有可选作坐标轴的参数
AXIS_PARAMS = [
    "轮高车高比_pct", "轴长比_pct",
    "窗身比_pct", "窗台线夹角_deg",
    "曲率平滑度", "特征线连续性",
    "家族特征匹配度", "间隙段差_mm",
    "头部空间_mm", "视野下沿角_deg",
]

PARAM_LABELS = {
    "轮高车高比_pct": "轮高车高比 (%)",
    "轴长比_pct": "轴长比 (%)",
    "窗身比_pct": "窗身比 (%)",
    "窗台线夹角_deg": "窗台线水平夹角 (°)",
    "曲率平滑度": "曲率平滑度评分",
    "特征线连续性": "特征线连续性评分",
    "家族特征匹配度": "家族特征匹配度",
    "间隙段差_mm": "间隙段差综合值 (mm)",
    "头部空间_mm": "头部空间余量 (mm)",
    "视野下沿角_deg": "前向视野下沿角 (°)",
}


# ═══════════════════════════════════════════════════════════════
# 区块 1: 数据层 — 50个聚类竞品散点
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def generate_competitor_clusters(seed: int = 42) -> pd.DataFrame:
    """生成约50个竞品散点，按类型聚类"""
    np.random.seed(seed)
    records = []

    # 定义各聚类的参数均值和标准差
    clusters = [
        # (类型, 风格, 数量, 车高均值, 轮胎高均值, 轴距均值, 车长均值,
        #  窗身比, 窗台线角, 曲率平滑, 特征线, 家族匹配, 间隙段差, 头部空间, 视野角,
        #  爆款概率)
        ("轿车", "运动", 17, 1430, 20, 685, 15, 2930, 40, 4850, 80,
         38.0, 1.8, 2.2, 0.7, 87, 5, 89, 5, 84, 5, 2.3, 0.3, 86, 5, 5.6, 0.6, 0.6),
        ("轿车", "豪华", 8,  1480, 25, 705, 15, 2940, 50, 4960, 80,
         42.0, 2.0, 3.0, 0.8, 84, 5, 87, 5, 90, 4, 2.6, 0.3, 94, 5, 6.3, 0.6, 0.7),
        ("SUV",  "豪华", 15, 1680, 40, 745, 20, 2920, 60, 4900, 120,
         43.0, 2.5, 3.2, 0.9, 81, 6, 83, 6, 90, 5, 2.9, 0.4, 102, 6, 7.2, 0.7, 0.65),
        ("猎装", "运动", 10, 1520, 30, 720, 18, 2960, 50, 4920, 100,
         40.0, 2.0, 2.4, 0.7, 85, 5, 87, 5, 85, 5, 2.5, 0.3, 90, 5, 6.2, 0.6, 0.55),
    ]

    for car_type, style, n, h_mean, h_std, t_mean, t_std, wb_mean, wb_std, L_mean, L_std, \
        wbr, wbr_std, wla, wla_std, curv, curv_std, feat, feat_std, \
        fam, fam_std, gap, gap_std, head, head_std, vis, vis_std, hot_prob in clusters:

        for i in range(n):
            h = np.random.normal(h_mean, h_std)
            t = np.random.normal(t_mean, t_std)
            wb = np.random.normal(wb_mean, wb_std)
            L = np.random.normal(L_mean, L_std)
            records.append({
                "车型": f"{car_type}-{style[:1]}{i+1:02d}",
                "类型": car_type,
                "风格": style,
                "车高_mm": round(h),
                "轮胎高度_mm": round(t),
                "轴距_mm": round(wb),
                "车长_mm": round(L),
                "窗身比_pct": round(np.random.normal(wbr, wbr_std), 1),
                "窗台线夹角_deg": round(np.random.normal(wla, wla_std), 1),
                "曲率平滑度": round(np.clip(np.random.normal(curv, curv_std), 60, 100)),
                "特征线连续性": round(np.clip(np.random.normal(feat, feat_std), 60, 100)),
                "家族特征匹配度": round(np.clip(np.random.normal(fam, fam_std), 60, 100)),
                "间隙段差_mm": round(np.random.normal(gap, gap_std), 1),
                "头部空间_mm": round(np.random.normal(head, head_std)),
                "视野下沿角_deg": round(np.random.normal(vis, vis_std), 1),
                "销量超5000": np.random.random() < hot_prob,
            })

    df = pd.DataFrame(records)
    df["轮高车高比_pct"] = (df["轮胎高度_mm"] / df["车高_mm"] * 100).round(1)
    df["轴长比_pct"] = (df["轴距_mm"] / df["车长_mm"] * 100).round(1)

    return df
```

- [ ] **Step 2: Verify data generation**

```bash
cd "E:/workspace-CA/styling evaluation system"
python -c "
exec(open('app.py').read().partition('# 区块 2')[0])
df = generate_competitor_clusters()
print(f'Rows: {len(df)}, Cols: {len(df.columns)}')
print(df.groupby('类型').size())
print('Columns:', df.columns.tolist())
"
```

Expected: `Rows: 50`, group sizes ~17/8/15/10, column list printed.

- [ ] **Step 3: Commit**

```bash
git add app.py && git commit -m "feat(v2): add constants and 50-point clustered data layer"
```

---

### Task 2: Scoring Engine — Preserve v1 Logic

**Files:**
- Modify: `app.py` (append scoring engine after data layer)

- [ ] **Step 1: Append scoring functions**

Append to `app.py`:

```python
# ═══════════════════════════════════════════════════════════════
# 区块 2: 计算引擎 — Z-score 评分
# ═══════════════════════════════════════════════════════════════

def calculate_z_score_based_rating(
    value: float, mean: float, std: float, style: str, metric_name: str
) -> float:
    """基于 Z-score + 风格加权的 0-10 标准分"""
    if std == 0 or np.isnan(std):
        return 5.0

    z_score = (value - mean) / std
    base_score = 5.0 + z_score * 2.0
    base_score = max(0.0, min(10.0, base_score))

    style_prefs = STYLE_PREFERENCES.get(style, {})
    preference = style_prefs.get(metric_name)

    if preference == "high" and z_score > 0:
        base_score += 0.5
    elif preference == "high" and z_score < 0:
        base_score -= 0.3
    elif preference == "low" and z_score < 0:
        base_score += 0.5
    elif preference == "low" and z_score > 0:
        base_score -= 0.3

    return round(max(0.0, min(10.0, base_score)), 1)


def calculate_dimension_scores(
    car_params: dict, df_comp: pd.DataFrame, style: str
) -> dict:
    """计算在研车型的五个维度得分"""
    dimension_scores = {}
    sub_scores = {}

    for dim, sub_params in DIMENSION_MAP.items():
        dim_sub_scores = []
        for param in sub_params:
            if param not in df_comp.columns:
                continue
            mean_val = df_comp[param].mean()
            std_val = df_comp[param].std()
            car_val = car_params.get(param, mean_val)
            score = calculate_z_score_based_rating(car_val, mean_val, std_val, style, param)
            sub_scores[param] = score
            dim_sub_scores.append(score)

        dimension_scores[dim] = round(np.mean(dim_sub_scores), 1) if dim_sub_scores else 5.0

    return {"dimension_scores": dimension_scores, "sub_scores": sub_scores}
```

- [ ] **Step 2: Verify scoring engine works**

```bash
cd "E:/workspace-CA/styling evaluation system"
python -c "
import pandas as pd, numpy as np
# Test: full file up to scoring
code = open('app.py').read()
exec(code.partition('# 区块 3')[0])
df = generate_competitor_clusters()
scores = calculate_dimension_scores(
    {'轮高车高比_pct': 48.0, '轴长比_pct': 60.0, '窗身比_pct': 38.0,
     '窗台线夹角_deg': 2.2, '曲率平滑度': 88, '特征线连续性': 90,
     '家族特征匹配度': 85, '间隙段差_mm': 2.3, '头部空间_mm': 88, '视野下沿角_deg': 5.8},
    df, '运动'
)
print('Dimension scores:', scores['dimension_scores'])
print('All scores in [0,10]:', all(0<=v<=10 for v in scores['dimension_scores'].values()))
"
```

Expected: all dimension scores printed and within [0, 10].

- [ ] **Step 3: Commit**

```bash
git add app.py && git commit -m "feat(v2): add scoring engine"
```

---

### Task 3: Chart Factory — Compact Radar + Design Space Map

**Files:**
- Modify: `app.py` (append chart functions)

- [ ] **Step 1: Append compact radar chart function**

Append to `app.py`:

```python
# ═══════════════════════════════════════════════════════════════
# 区块 3: 图表工厂
# ═══════════════════════════════════════════════════════════════

def create_compact_radar(dimension_scores: dict) -> go.Figure:
    """紧凑型小雷达图（用于顶部概览）"""
    dimensions = list(DIMENSION_MAP.keys())
    labels = [DIMENSION_ICONS[d] for d in dimensions]
    car_values = [dimension_scores.get(d, 5.0) for d in dimensions]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=car_values + [car_values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(239, 83, 80, 0.25)',
        name='在研车型',
        line=dict(color='#EF5350', width=2),
        marker=dict(color='#EF5350', size=6),
    ))

    # 基准线 5.0
    fig.add_trace(go.Scatterpolar(
        r=[5.0] * len(dimensions) + [5.0],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(66, 165, 245, 0.08)',
        name='竞品基准',
        line=dict(color='#42A5F5', width=1.5, dash='dash'),
        marker=dict(size=0),
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10],
                tickfont=dict(color='#aaa', size=9), gridcolor='rgba(255,255,255,0.08)'),
            angularaxis=dict(tickfont=dict(color='#e0e0e0', size=10),
                gridcolor='rgba(255,255,255,0.05)'),
            bgcolor='rgba(0,0,0,0)',
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0', size=10),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
        margin=dict(l=20, r=20, t=20, b=30),
        height=320,
    )
    return fig
```

- [ ] **Step 2: Append design space map function**

Append to `app.py`:

```python
def create_design_space_map(
    df: pd.DataFrame, x_param: str, y_param: str,
    car_x: float, car_y: float,
    hot_x: float, hot_y: float,
) -> go.Figure:
    """
    设计空间坐标系 — 二维散点 + 密度等高线 + 在研车型⭐

    Args:
        df: 竞品DataFrame
        x_param, y_param: 坐标轴参数列名
        car_x, car_y: 在研车型的X/Y坐标
        hot_x, hot_y: 销量>5000爆款均值的X/Y坐标
    """
    fig = go.Figure()

    # 图层1: 竞品灰色散点
    fig.add_trace(go.Scatter(
        x=df[x_param], y=df[y_param],
        mode='markers',
        name='竞品车型',
        marker=dict(color='#888888', size=9, opacity=0.55, symbol='circle'),
        hovertemplate=f'{PARAM_LABELS.get(x_param, x_param)}: %{{x:.2f}}<br>'
                      f'{PARAM_LABELS.get(y_param, y_param)}: %{{y:.2f}}<extra></extra>',
    ))

    # 图层2: 二维密度等高线
    try:
        x_vals = df[x_param].dropna()
        y_vals = df[y_param].dropna()
        if len(x_vals) > 3:
            hist2d = go.Histogram2dContour(
                x=x_vals, y=y_vals,
                colorscale=[[0, 'rgba(66,165,245,0)'], [1, 'rgba(66,165,245,0.35)']],
                showscale=False,
                contours=dict(coloring='lines', showlines=True, end=0.8, start=0.3, size=0.1),
                line=dict(color='rgba(66,165,245,0.5)', width=1),
                name='密度等高线',
                hoverinfo='none',
            )
            fig.add_trace(hist2d)
    except Exception:
        pass  # 密度等高线绘制失败时跳过

    # 图层3: 红色大五角星 — 在研车型
    fig.add_trace(go.Scatter(
        x=[car_x], y=[car_y],
        mode='markers+text',
        name='在研车型',
        marker=dict(color='#EF5350', size=28, symbol='star',
                     line=dict(color='white', width=2)),
        text=['★ 在研'],
        textposition='top center',
        textfont=dict(color='#EF5350', size=13, family='sans-serif'),
        hoverinfo='none',
    ))

    # 图层4: 绿色菱形 — 爆款均值
    fig.add_trace(go.Scatter(
        x=[hot_x], y=[hot_y],
        mode='markers',
        name='爆款均值',
        marker=dict(color='#66BB6A', size=16, symbol='diamond',
                     line=dict(color='white', width=1.5)),
        hoverinfo='none',
    ))

    x_label = PARAM_LABELS.get(x_param, x_param)
    y_label = PARAM_LABELS.get(y_param, y_param)

    fig.update_layout(
        title=dict(text=f'<b>Design Space Map</b>: {y_label} vs {x_label}',
                   font=dict(size=15, color='#e0e0e0')),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                     font=dict(size=10)),
        margin=dict(l=60, r=30, t=60, b=60),
        height=480,
    )

    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', zeroline=False)

    return fig
```

- [ ] **Step 3: Verify chart imports and quick render test**

```bash
cd "E:/workspace-CA/styling evaluation system"
python -c "
import plotly.graph_objects as go
import plotly.express as px
print('Plotly OK')
# Quick function existence check
code = open('app.py').read()
exec(code.partition('# 区块 4')[0])
print('create_compact_radar:', callable(create_compact_radar))
print('create_design_space_map:', callable(create_design_space_map))
"
```

Expected: `Plotly OK`, both functions callable.

- [ ] **Step 4: Commit**

```bash
git add app.py && git commit -m "feat(v2): add compact radar and design space map charts"
```

---

### Task 4: UI — Sidebar + Vehicle Studio

**Files:**
- Modify: `app.py` (append sidebar and vehicle studio functions)

- [ ] **Step 1: Append sidebar function**

Append to `app.py`:

```python
# ═══════════════════════════════════════════════════════════════
# 区块 4: UI 渲染
# ═══════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame) -> dict:
    """左侧边栏：参数输入 + 竞品选择"""
    st.sidebar.title("🎯 Design Studio Control")
    st.sidebar.markdown("---")

    car_name = st.sidebar.text_input("项目代号", value="Concept-X")
    car_style = st.sidebar.selectbox("目标风格", ["运动", "豪华", "科技"], index=0)

    st.sidebar.markdown("---")

    params = {}
    param_configs = {
        "📐 比例": [
            ("轮高车高比 (%)", "轮高车高比_pct", 40.0, 55.0, 48.0, 0.5),
            ("轴长比 (%)", "轴长比_pct", 55.0, 68.0, 60.0, 0.5),
        ],
        "🚙 姿态": [
            ("窗身比 (%)", "窗身比_pct", 28.0, 48.0, 38.0, 0.5),
            ("窗台线水平夹角 (°)", "窗台线夹角_deg", 1.0, 5.0, 2.2, 0.1),
        ],
        "✨ 型面": [
            ("曲率平滑度评分", "曲率平滑度", 60.0, 100.0, 87, 1),
            ("特征线连续性评分", "特征线连续性", 60.0, 100.0, 89, 1),
        ],
        "💎 细节品质": [
            ("家族特征匹配度", "家族特征匹配度", 60.0, 100.0, 84, 1),
            ("间隙段差综合值 (mm)", "间隙段差_mm", 1.0, 5.0, 2.3, 0.1),
        ],
        "⚙️ 工程平衡": [
            ("头部空间余量 (mm)", "头部空间_mm", 70.0, 120.0, 86, 1),
            ("前向视野下沿角 (°)", "视野下沿角_deg", 3.0, 10.0, 5.6, 0.1),
        ],
    }

    for section, controls in param_configs.items():
        with st.sidebar.expander(section, expanded=True):
            for label, col_name, vmin, vmax, vdef, step in controls:
                params[col_name] = st.slider(
                    label, min_value=vmin, max_value=vmax,
                    value=vdef, step=step, key=col_name,
                )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 对标竞品池")
    all_cars = df["车型"].tolist()
    selected_cars = st.sidebar.multiselect(
        "选择对标车型", options=all_cars, default=all_cars,
        help="取消勾选不需要对比的车型",
    )

    return {"car_name": car_name, "car_style": car_style, "params": params, "selected_cars": selected_cars}
```

- [ ] **Step 2: Append vehicle studio function**

Append to `app.py`:

```python
def render_vehicle_studio(dimension_scores: dict):
    """顶部核心视觉区：5列车型视角图 + 紧凑雷达图"""
    st.markdown("## 🚗 Vehicle Visual Studio")
    st.markdown("---")

    cols = st.columns(6)

    views = [
        ("正侧", "Side View"),
        ("正前", "Front View"),
        ("正后", "Rear View"),
        ("前45°", "Front 3/4"),
        ("后45°", "Rear 3/4"),
    ]

    for i, (cn_name, en_name) in enumerate(views):
        with cols[i]:
            # 灰色占位图
            placeholder = np.ones((300, 480, 3), dtype=np.uint8) * 35
            # 添加十字线标记中心
            h, w = 300, 480
            placeholder[h//2-1:h//2+2, :, :] = 60
            placeholder[:, w//2-1:w//2+2, :] = 60
            st.image(placeholder, caption=f"{cn_name} | {en_name}", use_container_width=True)

    # 第6列：紧凑雷达图
    with cols[5]:
        st.markdown("##### 🎯 五步法概览")
        fig_radar = create_compact_radar(dimension_scores)
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
```

- [ ] **Step 3: Verify sidebar + vehicle studio functions exist**

```bash
cd "E:/workspace-CA/styling evaluation system"
python -c "
code = open('app.py').read()
exec(code.partition('# 区块 5')[0])
print('render_sidebar:', callable(render_sidebar))
print('render_vehicle_studio:', callable(render_vehicle_studio))
"
```

- [ ] **Step 4: Commit**

```bash
git add app.py && git commit -m "feat(v2): add sidebar and vehicle studio UI"
```

---

### Task 5: UI — Design Space Tabs + Main Flow

**Files:**
- Modify: `app.py` (append design space tabs and main function)

- [ ] **Step 1: Append design space tabs function**

Append to `app.py`:

```python
def render_design_space_tabs(df: pd.DataFrame, car_params: dict):
    """中下部：5个Tab，每个含X/Y轴选择器 + 设计空间散点密度图"""
    st.markdown("## 🔬 Design Space Analysis")
    st.markdown("---")

    tab_labels = [DIMENSION_ICONS[d] for d in DIMENSION_MAP.keys()]
    dim_names = list(DIMENSION_MAP.keys())
    tabs = st.tabs(tab_labels)

    df_hot = df[df["销量超5000"] == True]

    axis_options = [(PARAM_LABELS.get(p, p), p) for p in AXIS_PARAMS]
    display_to_key = {d: k for d, k in axis_options}

    for i, dim in enumerate(dim_names):
        with tabs[i]:
            sub_params = DIMENSION_MAP[dim]
            # 默认轴：该维度的两个子参数
            default_x = sub_params[0]
            default_y = sub_params[1] if len(sub_params) > 1 else sub_params[0]

            col_sel_x, col_sel_y = st.columns([1, 1])

            with col_sel_x:
                x_display = st.selectbox(
                    "🔴 X 轴参数",
                    options=[d for d, _ in axis_options],
                    index=[k for _, k in axis_options].index(default_x),
                    key=f"x_axis_{dim}",
                )
            with col_sel_y:
                y_display = st.selectbox(
                    "🔵 Y 轴参数",
                    options=[d for d, _ in axis_options],
                    index=[k for _, k in axis_options].index(default_y),
                    key=f"y_axis_{dim}",
                )

            x_param = display_to_key[x_display]
            y_param = display_to_key[y_display]

            car_x = car_params.get(x_param, df[x_param].mean())
            car_y = car_params.get(y_param, df[y_param].mean())
            hot_x = df_hot[x_param].mean() if len(df_hot) > 0 else df[x_param].mean()
            hot_y = df_hot[y_param].mean() if len(df_hot) > 0 else df[y_param].mean()

            fig_map = create_design_space_map(
                df, x_param, y_param, float(car_x), float(car_y),
                float(hot_x), float(hot_y),
            )
            st.plotly_chart(fig_map, use_container_width=True)

            # 坐标位置信息
            st.caption(
                f"在研车型 ({PARAM_LABELS.get(x_param, x_param)}={car_x:.1f}, "
                f"{PARAM_LABELS.get(y_param, y_param)}={car_y:.1f}) | "
                f"爆款均值 ({hot_x:.1f}, {hot_y:.1f})"
            )
```

- [ ] **Step 2: Append main() function**

Append to `app.py`:

```python
# ═══════════════════════════════════════════════════════════════
# 区块 5: 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    st.title("🚗 Automotive Design Studio")
    st.caption("高级汽车设计专家工作台 · 五步法评价 · 设计空间分析")

    df = generate_competitor_clusters()
    inputs = render_sidebar(df)

    if not inputs["selected_cars"]:
        st.warning("⚠️ 请至少选择一款竞品车型作为对标")
        return

    df_comp = df[df["车型"].isin(inputs["selected_cars"])]

    if len(df_comp) == 0:
        st.warning("⚠️ 对标竞品池为空")
        return

    result = calculate_dimension_scores(inputs["params"], df_comp, inputs["car_style"])

    st.markdown(f"**项目代号:** {inputs['car_name']} | **风格:** {inputs['car_style']} | **对标竞品:** {len(df_comp)} 款")
    st.markdown("---")

    render_vehicle_studio(result["dimension_scores"])
    render_design_space_tabs(df_comp, inputs["params"])

    st.markdown("---")
    st.caption("© 2026 Automotive Design Studio Workspace v2 | Powered by Streamlit + Plotly")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Full syntax verification**

```bash
cd "E:/workspace-CA/styling evaluation system"
python -c "import py_compile; py_compile.compile('app.py', doraise=True); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 4: Run app end-to-end**

```bash
cd "E:/workspace-CA/styling evaluation system"
python -m streamlit run app.py
```

Expected: App launches on http://localhost:8501 with:
- Top: 5 car view columns (gray placeholders) + compact radar in 6th column
- Middle: 5 tabs with Design Space Map, each with X/Y axis dropdowns
- Left: Sidebar with 10 parameter sliders + competitor multi-select
- Red star marker moves when sliders change

- [ ] **Step 5: Final commit**

```bash
git add app.py && git commit -m "feat(v2): complete Automotive Design Studio Workspace"
```

---

## Verification Checklist

- [ ] `python -m streamlit run app.py` starts without errors
- [ ] Top row: 5 gray placeholder images with Chinese/English view labels
- [ ] Compact radar chart in 6th column (top right)
- [ ] 5 tabs with correct icons and labels
- [ ] Each tab: X-axis and Y-axis dropdown selectors work
- [ ] Each tab: Design Space Map with gray competitor scatter points (~50)
- [ ] Density contour lines visible on scatter plots
- [ ] Large red star (★) marks in-development vehicle position
- [ ] Green diamond marks hot-sale mean position
- [ ] Moving sidebar sliders → red star position updates in real time
- [ ] Competitor multi-select filtering works
- [ ] Dark sci-fi theme consistent (plotly_dark + Streamlit dark)
- [ ] Chinese text renders correctly in all labels, legends, tooltips

## Self-Review

**Spec coverage:**
- ✅ 50 clustered competitor points → Task 1
- ✅ Vehicle Visual Studio (5-column + compact radar) → Task 4
- ✅ Design Space Map (scatter + density + star) → Task 3
- ✅ Tab axis selectors (user-defined X/Y) → Task 5
- ✅ Sidebar with parameter sliders → Task 4
- ✅ Scoring engine preserved → Task 2
- ✅ Dark sci-fi theme → consistent across all tasks
- ✅ Real-time slider → star movement → Streamlit reactivity (automatic)

**Placeholder scan:** No TBD, TODO, or vague instructions.

**Type consistency:** `DIMENSION_MAP` keys, `AXIS_PARAMS`, `PARAM_LABELS` keys all match across functions. `render_sidebar()` return dict keys match usage in `main()`. Plotly trace names match spec.
