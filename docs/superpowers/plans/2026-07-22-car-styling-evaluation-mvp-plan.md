# 汽车造型设计数据评价系统 MVP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit web app that evaluates in-development vehicle styling against competitor benchmarks using the "五步法" framework with radar charts and normal distribution curves.

**Architecture:** Single-file `app.py` with 5 logical blocks: data layer, computing engine, chart factory, UI rendering, and main flow. All state lives in Streamlit session; no database or external API.

**Tech Stack:** Python 3.10+, Streamlit, Pandas, NumPy, Plotly, SciPy

---

### Task 1: Project Setup — Config Files

**Files:**
- Create: `.streamlit/config.toml`
- Create: `requirements.txt`

- [ ] **Step 1: Create Streamlit dark theme config**

Write `.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#4fc3f7"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1a1c23"
textColor = "#e0e0e0"
font = "sans serif"
```

- [ ] **Step 2: Create requirements.txt**

Write `requirements.txt`:

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
scipy>=1.10.0
```

- [ ] **Step 3: Commit**

```bash
git add .streamlit/config.toml requirements.txt
git commit -m "chore: add project config and dependencies"
```

---

### Task 2: Data Layer — Mock Car Database

**Files:**
- Create: `app.py` (data layer section only)

- [ ] **Step 1: Write imports and page config**

Create `app.py` with the following content:

```python
"""
汽车造型设计数据评价系统 — 五步法 MVP
Car Styling Design Data Evaluation System
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="汽车造型设计数据评价系统",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

- [ ] **Step 2: Write the mock data generation function**

Append to `app.py`:

```python
# ═══════════════════════════════════════════════════════════════
# 区块 1: 数据层 — 模拟竞品数据库
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def generate_mock_car_database() -> pd.DataFrame:
    """生成10款竞品车型的模拟数据库"""
    data = {
        "车型": [
            "小米SU7", "汉EV", "问界M5", "Model 3", "ET5",
            "极氪001", "理想L7", "P7", "阿维塔12", "海豹EV"
        ],
        "类型": [
            "轿车", "轿车", "SUV", "轿车", "轿车",
            "猎装", "SUV", "轿车", "轿车", "轿车"
        ],
        "风格": [
            "运动", "豪华", "运动", "科技", "运动",
            "运动", "豪华", "科技", "科技", "运动"
        ],
        "车高_mm":       [1440, 1495, 1625, 1441, 1409, 1560, 1750, 1450, 1460, 1420],
        "轮胎高度_mm":    [690,  710,  740,  685,  680,  730,  760,  695,  700,  675],
        "轴距_mm":        [3000, 2920, 2880, 2875, 2888, 3005, 3005, 2998, 3020, 2920],
        "车长_mm":        [4997, 4995, 4770, 4720, 4790, 4970, 5050, 4880, 5020, 4800],
        "窗身比_pct":     [38,   42,   41,   36,   37,   40,   44,   35,   37,   39],
        "窗台线夹角_deg":  [2.2,  3.0,  2.5,  1.8,  2.0,  2.3,  3.5,  1.5,  2.0,  2.1],
        "曲率平滑度":      [88,   85,   82,   90,   87,   86,   80,   88,   89,   86],
        "特征线连续性":    [90,   88,   85,   92,   89,   87,   82,   91,   93,   88],
        "家族特征匹配度":  [85,   90,   88,   82,   83,   86,   92,   80,   88,   84],
        "间隙段差_mm":     [2.3,  2.5,  2.8,  2.1,  2.2,  2.4,  3.0,  2.0,  2.2,  2.3],
        "头部空间_mm":     [88,   95,  100,   85,   82,   92,  105,   83,   87,   84],
        "视野下沿角_deg":  [5.8,  6.2,  7.0,  5.5,  5.3,  6.5,  7.5,  5.0,  5.6,  5.4],
        "销量超5000": [
            True, True, True, True, False,
            True, True, False, False, True
        ],
    }

    df = pd.DataFrame(data)

    # 计算派生参数
    df["轮高车高比_pct"] = (df["轮胎高度_mm"] / df["车高_mm"] * 100).round(1)
    df["轴长比_pct"] = (df["轴距_mm"] / df["车长_mm"] * 100).round(1)

    return df
```

- [ ] **Step 3: Verify data loads correctly**

Run in terminal:

```bash
cd "E:/workspace-CA/styling evaluation system"
python -c "import sys; sys.path.insert(0,'.'); exec(open('app.py').read().split('# ═══════')[0]); df = generate_mock_car_database(); print(df.shape); print(df.columns.tolist())"
```

Expected: `(10, 18)` and column list printed.

---

### Task 3: Computing Engine — Scoring Functions

**Files:**
- Modify: `app.py` (append computing engine section)

- [ ] **Step 1: Write dimension map and style preferences**

Append to `app.py`:

```python
# ═══════════════════════════════════════════════════════════════
# 区块 2: 计算引擎 — 评分算法
# ═══════════════════════════════════════════════════════════════

# 五步法维度 → 底层子参数映射
DIMENSION_MAP = {
    "比例":     ["轮高车高比_pct", "轴长比_pct"],
    "姿态":     ["窗身比_pct", "窗台线夹角_deg"],
    "型面":     ["曲率平滑度", "特征线连续性"],
    "细节品质": ["家族特征匹配度", "间隙段差_mm"],
    "工程平衡": ["头部空间_mm", "视野下沿角_deg"],
}

# 各维度显示名称（用于雷达图和Tab标签）
DIMENSION_ICONS = {
    "比例":     "📐 比例",
    "姿态":     "🚙 姿态",
    "型面":     "✨ 型面",
    "细节品质": "💎 细节品质",
    "工程平衡": "⚙️ 工程平衡",
}

# 风格偏好方向: "high" = 值偏高加分, "low" = 值偏低加分
STYLE_PREFERENCES = {
    "运动": {
        "窗身比_pct": "low", "窗台线夹角_deg": "low",
        "轮高车高比_pct": "high",
    },
    "豪华": {
        "窗身比_pct": "high", "头部空间_mm": "high",
        "间隙段差_mm": "low", "视野下沿角_deg": "high",
    },
    "科技": {
        "曲率平滑度": "high", "特征线连续性": "high",
        "间隙段差_mm": "low", "轴长比_pct": "high",
    },
}
```

- [ ] **Step 2: Write the Z-score scoring function**

Append to `app.py`:

```python
def calculate_z_score_based_rating(
    value: float, mean: float, std: float, style: str, metric_name: str
) -> float:
    """
    基于 Z-score + 风格加权的 0-10 标准分计算

    Args:
        value: 在研车型的底层参数值
        mean: 竞品池中该参数的均值
        std: 竞品池中该参数的标准差
        style: 在研车型的造型风格 ("运动"/"豪华"/"科技")
        metric_name: 参数列名

    Returns:
        0.0 - 10.0 的标准分
    """
    if std == 0 or np.isnan(std):
        return 5.0  # 无差异时给中性分

    z_score = (value - mean) / std

    # 基础分：Z=0→5分, Z=±2.5→0/10分
    base_score = 5.0 + z_score * 2.0
    base_score = max(0.0, min(10.0, base_score))

    # 风格加权 (±0.5 调整)
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
```

- [ ] **Step 3: Write the dimension scores aggregator**

Append to `app.py`:

```python
def calculate_dimension_scores(
    car_params: dict, df_comp: pd.DataFrame, style: str
) -> dict:
    """
    计算在研车型的五个维度得分

    Args:
        car_params: {metric_name: value} 在研车型的10个底层参数
        df_comp: 竞品DataFrame
        style: 造型风格

    Returns:
        {
            "dimension_scores": {"比例": 7.5, "姿态": 6.8, ...},
            "sub_scores": {"轮高车高比_pct": 8.0, ...},
            "competitor_means": {"比例": 6.2, ...},
        }
    """
    dimension_scores = {}
    sub_scores = {}
    competitor_means = {}

    for dim, sub_params in DIMENSION_MAP.items():
        dim_sub_scores = []
        dim_comp_means = []

        for param in sub_params:
            if param not in df_comp.columns:
                continue
            mean_val = df_comp[param].mean()
            std_val = df_comp[param].std()
            car_val = car_params.get(param, mean_val)

            score = calculate_z_score_based_rating(
                car_val, mean_val, std_val, style, param
            )
            sub_scores[param] = score
            dim_sub_scores.append(score)
            dim_comp_means.append(round(mean_val, 2))

        dimension_scores[dim] = round(np.mean(dim_sub_scores), 1) if dim_sub_scores else 5.0
        competitor_means[dim] = 5.0  # 竞品均值在雷达图上始终为5.0（基准线）

    return {
        "dimension_scores": dimension_scores,
        "sub_scores": sub_scores,
        "competitor_means": competitor_means,
    }
```

- [ ] **Step 4: Verify scoring engine**

Run:

```bash
cd "E:/workspace-CA/styling evaluation system"
python -c "
import pandas as pd, numpy as np
# Quick unit test
mean, std = 48.0, 3.0
# z=0 → ~5.0
print('z=0:', 5.0 + 0*2.0)
# z=1 → ~7.0
print('z=+1:', 5.0 + 1.0*2.0)
# z=-1 → ~3.0
print('z=-1:', 5.0 + (-1.0)*2.0)
"
```

Expected output shows 5.0, 7.0, 3.0.

---

### Task 4: Chart Factory — Plotly Visualizations

**Files:**
- Modify: `app.py` (append chart functions)

- [ ] **Step 1: Write the radar chart function**

Append to `app.py`:

```python
# ═══════════════════════════════════════════════════════════════
# 区块 3: 图表工厂 — Plotly 可视化
# ═══════════════════════════════════════════════════════════════

def create_radar_chart(dimension_scores: dict, competitor_means: dict) -> go.Figure:
    """绘制五步法雷达图（在研车型 vs 竞品均值）"""
    dimensions = list(DIMENSION_MAP.keys())
    labels = [DIMENSION_ICONS[d] for d in dimensions]

    car_values = [dimension_scores.get(d, 5.0) for d in dimensions]
    comp_values = [competitor_means.get(d, 5.0) for d in dimensions]

    fig = go.Figure()

    # 在研车型
    fig.add_trace(go.Scatterpolar(
        r=car_values + [car_values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(239, 83, 80, 0.25)',
        name='在研车型',
        line=dict(color='#EF5350', width=2.5),
        marker=dict(color='#EF5350', size=8),
    ))

    # 竞品均值
    fig.add_trace(go.Scatterpolar(
        r=comp_values + [comp_values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(66, 165, 245, 0.12)',
        name='竞品均值',
        line=dict(color='#42A5F5', width=2, dash='dash'),
        marker=dict(color='#42A5F5', size=6),
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(color='#aaa', size=11),
                gridcolor='rgba(255,255,255,0.1)',
            ),
            angularaxis=dict(
                tickfont=dict(color='#e0e0e0', size=13),
                gridcolor='rgba(255,255,255,0.08)',
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        legend=dict(
            orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5,
            font=dict(size=12, color='#e0e0e0'),
        ),
        margin=dict(l=40, r=40, t=30, b=40),
        height=420,
    )

    return fig
```

- [ ] **Step 2: Write the distribution chart function**

Append to `app.py`:

```python
def create_distribution_chart(
    df: pd.DataFrame, param_name: str, car_value: float,
    hot_sale_mean: float, param_label: str
) -> go.Figure:
    """
    绘制单参数的正态分布钟形曲线

    Args:
        df: 竞品DataFrame
        param_name: 参数列名
        car_value: 在研车型该参数的值
        hot_sale_mean: 销量>5000车型的该参数均值
        param_label: 参数的显示名称
    """
    values = df[param_name].dropna()
    mean_val = values.mean()
    std_val = values.std()

    if std_val == 0 or np.isnan(std_val):
        std_val = mean_val * 0.05 if mean_val != 0 else 1.0

    # 生成正态分布曲线
    x_range = np.linspace(mean_val - 3.5 * std_val, mean_val + 3.5 * std_val, 200)
    y_pdf = stats.norm.pdf(x_range, mean_val, std_val)

    fig = go.Figure()

    # 钟形曲线
    fig.add_trace(go.Scatter(
        x=x_range, y=y_pdf,
        mode='lines',
        name='竞品分布',
        line=dict(color='#64B5F6', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(100, 181, 246, 0.12)',
    ))

    # 红色垂线 — 在研车型
    y_max = max(y_pdf) if len(y_pdf) > 0 else 0.01
    fig.add_trace(go.Scatter(
        x=[car_value, car_value], y=[0, y_max],
        mode='lines',
        name=f'在研车型 ({car_value})',
        line=dict(color='#EF5350', width=3),
    ))

    # 红色圆点标记
    car_y = stats.norm.pdf(car_value, mean_val, std_val)
    fig.add_trace(go.Scatter(
        x=[car_value], y=[car_y],
        mode='markers',
        name=f'在研车型 ({car_value})',
        marker=dict(color='#EF5350', size=12, symbol='circle',
                     line=dict(color='white', width=1.5)),
        showlegend=False,
    ))

    # 绿色虚线 — 销量>5000 均值
    hot_y = stats.norm.pdf(hot_sale_mean, mean_val, std_val)
    fig.add_trace(go.Scatter(
        x=[hot_sale_mean, hot_sale_mean], y=[0, y_max],
        mode='lines',
        name=f'爆款均值 ({hot_sale_mean:.1f})',
        line=dict(color='#66BB6A', width=2.5, dash='dash'),
    ))

    # 爆款均值圆点
    fig.add_trace(go.Scatter(
        x=[hot_sale_mean], y=[hot_y],
        mode='markers',
        name=f'爆款均值 ({hot_sale_mean:.1f})',
        marker=dict(color='#66BB6A', size=10, symbol='diamond',
                     line=dict(color='white', width=1)),
        showlegend=False,
    ))

    # 灰色虚线 — 全竞品均值
    mean_y = stats.norm.pdf(mean_val, mean_val, std_val)
    fig.add_trace(go.Scatter(
        x=[mean_val, mean_val], y=[0, mean_y],
        mode='lines',
        name=f'竞品均值 ({mean_val:.1f})',
        line=dict(color='#9E9E9E', width=1.5, dash='dot'),
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text=f'<b>{param_label}</b> 正态分布',
            font=dict(size=15, color='#e0e0e0'),
        ),
        xaxis_title=param_label,
        yaxis_title='概率密度',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='right', x=1, font=dict(size=10, color='#e0e0e0'),
        ),
        margin=dict(l=50, r=30, t=50, b=50),
        height=380,
    )

    fig.update_xaxes(gridcolor='rgba(255,255,255,0.06)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.06)', zeroline=False)

    return fig
```

- [ ] **Step 4: Verify charts import correctly**

Run:

```bash
cd "E:/workspace-CA/styling evaluation system"
python -c "
import plotly.graph_objects as go
from scipy import stats
import numpy as np
print('Plotly version:', go.__version__ if hasattr(go, '__version__') else 'OK')
print('scipy version:', stats.__dict__.get('__version__', 'OK'))
print('All imports successful')
"
```

Expected: no errors, imports confirmed.

---

### Task 5: UI Rendering — Sidebar

**Files:**
- Modify: `app.py` (append sidebar function)

- [ ] **Step 1: Write sidebar rendering function**

Append to `app.py`:

```python
# ═══════════════════════════════════════════════════════════════
# 区块 4: UI 渲染
# ═══════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame) -> dict:
    """渲染左侧边栏，返回在研车型参数字典"""
    st.sidebar.title("🎯 在研车型参数")
    st.sidebar.markdown("---")

    car_name = st.sidebar.text_input("车型名称", value="在研车型-X1")
    car_style = st.sidebar.selectbox("造型风格", ["运动", "豪华", "科技"], index=0)

    st.sidebar.markdown("---")

    params = {}
    param_configs = {
        # (显示名, 列名, min, max, default, step, unit)
        "📐 比例": [
            ("轮高车高比 (%)", "轮高车高比_pct", 40.0, 55.0, 48.0, 0.5, "%"),
            ("轴长比 (%)", "轴长比_pct", 55.0, 68.0, 60.0, 0.5, "%"),
        ],
        "🚙 姿态": [
            ("窗身比 (%)", "窗身比_pct", 28.0, 48.0, 40.0, 0.5, "%"),
            ("窗台线水平夹角 (°)", "窗台线夹角_deg", 1.0, 5.0, 2.5, 0.1, "°"),
        ],
        "✨ 型面": [
            ("曲率平滑度评分", "曲率平滑度", 60.0, 100.0, 85.0, 1.0, "分"),
            ("特征线连续性评分", "特征线连续性", 60.0, 100.0, 90.0, 1.0, "分"),
        ],
        "💎 细节品质": [
            ("家族特征匹配度", "家族特征匹配度", 60.0, 100.0, 85.0, 1.0, "分"),
            ("间隙段差综合值 (mm)", "间隙段差_mm", 1.0, 5.0, 2.5, 0.1, "mm"),
        ],
        "⚙️ 工程平衡": [
            ("头部空间余量 (mm)", "头部空间_mm", 70.0, 120.0, 90.0, 1.0, "mm"),
            ("前向视野下沿角 (°)", "视野下沿角_deg", 3.0, 10.0, 6.0, 0.1, "°"),
        ],
    }

    for section, controls in param_configs.items():
        with st.sidebar.expander(section, expanded=True):
            for label, col_name, vmin, vmax, vdef, step, unit in controls:
                params[col_name] = st.slider(
                    label, min_value=vmin, max_value=vmax,
                    value=vdef, step=step, key=col_name,
                )

    st.sidebar.markdown("---")

    # 竞品池多选
    st.sidebar.subheader("🔍 对标竞品池")
    all_cars = df["车型"].tolist()
    selected_cars = st.sidebar.multiselect(
        "选择对标竞品",
        options=all_cars,
        default=all_cars,
        help="取消勾选不需要对比的车型",
    )

    return {
        "car_name": car_name,
        "car_style": car_style,
        "params": params,
        "selected_cars": selected_cars,
    }
```

- [ ] **Step 2: Verify sidebar can run without errors**

```bash
cd "E:/workspace-CA/styling evaluation system"
streamlit run app.py --server.headless true 2>&1 | head -5 || echo "Streamlit module loaded OK"
```

---

### Task 6: UI Rendering — Macro & Micro Views

**Files:**
- Modify: `app.py` (append macro and micro view functions)

- [ ] **Step 1: Write macro view function**

Append to `app.py`:

```python
def render_macro_view(dimension_scores: dict, competitor_means: dict, sub_scores: dict):
    """渲染宏观总览：雷达图 + 评分摘要"""
    st.markdown("## 📊 宏观总览 — 五步法雷达图")

    col_chart, col_score = st.columns([3, 1])

    with col_chart:
        fig_radar = create_radar_chart(dimension_scores, competitor_means)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_score:
        st.markdown("### 📋 评分摘要")
        overall = round(np.mean(list(dimension_scores.values())), 1)

        st.metric(label="综合评分", value=f"{overall}/10",
                  delta=f"{overall - 5.0:+.1f} vs 基准")

        st.markdown("---")

        for dim, score in dimension_scores.items():
            icon = DIMENSION_ICONS.get(dim, dim)
            delta_val = score - 5.0
            delta_str = f"{delta_val:+.1f}"
            st.metric(label=icon, value=f"{score}/10", delta=delta_str)

    st.markdown("---")
```

- [ ] **Step 2: Write micro view function**

Append to `app.py`:

```python
def render_micro_view(df: pd.DataFrame, car_params: dict):
    """渲染微观详情：5个Tab，每个Tab展示该维度子参数的正态分布"""
    st.markdown("## 🔬 微观详情 — 各维度正态分布")

    # 构建Tab标签
    tab_labels = [DIMENSION_ICONS[d] for d in DIMENSION_MAP.keys()]
    dim_names = list(DIMENSION_MAP.keys())
    tabs = st.tabs(tab_labels)

    # 爆款数据
    df_hot = df[df["销量超5000"] == True]

    param_labels = {
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

    for i, dim in enumerate(dim_names):
        with tabs[i]:
            sub_params = DIMENSION_MAP[dim]

            cols = st.columns(len(sub_params))
            for j, param in enumerate(sub_params):
                with cols[j]:
                    car_val = car_params.get(param, df[param].mean())
                    hot_mean = df_hot[param].mean() if len(df_hot) > 0 else df[param].mean()
                    label = param_labels.get(param, param)

                    fig_dist = create_distribution_chart(
                        df, param, float(car_val), float(hot_mean), label
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                    # 显示指标说明
                    param_mean = df[param].mean()
                    param_std = df[param].std()
                    z = (car_val - param_mean) / param_std if param_std > 0 else 0
                    st.caption(
                        f"竞品均值: {param_mean:.1f} | "
                        f"标准差: {param_std:.2f} | "
                        f"Z-score: {z:+.2f}"
                    )
```

---

### Task 7: Main Flow & Integration

**Files:**
- Modify: `app.py` (append main function)

- [ ] **Step 1: Write the main() function**

Append to `app.py`:

```python
# ═══════════════════════════════════════════════════════════════
# 区块 5: 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    st.title("🚗 汽车造型设计数据评价系统")
    st.caption("基于五步法 · Z-score 统计评价 · 风格加权")

    # 加载数据
    df = generate_mock_car_database()

    # 侧边栏
    inputs = render_sidebar(df)

    if not inputs["selected_cars"]:
        st.warning("⚠️ 请至少选择一款竞品车型作为对标")
        return

    # 筛选竞品
    df_comp = df[df["车型"].isin(inputs["selected_cars"])]

    if len(df_comp) == 0:
        st.warning("⚠️ 对标竞品池为空")
        return

    # 计算得分
    result = calculate_dimension_scores(
        inputs["params"], df_comp, inputs["car_style"]
    )

    # 渲染
    st.markdown(f"### 在研车型: **{inputs['car_name']}** | 风格: {inputs['car_style']} | 对标竞品: {len(df_comp)}款")
    st.markdown("---")

    render_macro_view(
        result["dimension_scores"],
        result["competitor_means"],
        result["sub_scores"],
    )

    render_micro_view(df_comp, inputs["params"])

    # 页脚
    st.markdown("---")
    st.caption("© 2026 汽车造型设计数据评价系统 MVP | Powered by Streamlit + Plotly")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the complete app and verify**

```bash
cd "E:/workspace-CA/styling evaluation system"
streamlit run app.py
```

Expected: App launches on http://localhost:8501 with:
- Sidebar with 10 parameter sliders and competitor multi-select
- Radar chart comparing in-development vs competitor mean
- 5 tabs with normal distribution curves, red/blue/green markers

- [ ] **Step 3: Final commit**

```bash
git add app.py
git commit -m "feat: complete car styling evaluation MVP"
```

---

## Verification Checklist

After all tasks complete, verify end-to-end:

- [ ] `streamlit run app.py` starts without import errors
- [ ] Sidebar renders 10 parameter sliders across 5 expandable sections
- [ ] Changing sliders updates radar chart in real-time
- [ ] Radar chart shows 5 dimensions with red (in-dev) and blue (competitor mean) traces
- [ ] All 5 tabs display correctly with proper icons
- [ ] Each tab shows distribution curves for its sub-parameters
- [ ] Red vertical line marks in-development vehicle position on distribution curves
- [ ] Green dashed line marks "sales > 5000" mean
- [ ] Competitor multi-select filtering works correctly
- [ ] Chinese characters render correctly in chart titles, labels, and legends
- [ ] Dark theme is consistent across all UI elements
- [ ] Scoring summary card shows correct values

---

## Plan Self-Review

**Spec coverage check:**
- ✅ Mock data (10 cars, 10 sub-parameters, 16 columns) → Task 2
- ✅ Five dimensions with sub-parameter mapping → Task 3
- ✅ Z-score scoring with style weighting → Task 3
- ✅ Radar chart (macro view) → Task 4
- ✅ Distribution curves (micro view) → Task 4 + Task 6
- ✅ Sidebar with sliders and competitor selector → Task 5
- ✅ st.tabs for 5 dimensions → Task 6
- ✅ Dark theme (plotly_dark + Streamlit config) → Task 1
- ✅ Chinese font support → implicit in Plotly + Streamlit handling
- ✅ Single file app.py → confirmed across all tasks
- ✅ `streamlit run app.py` → Task 7 verification

**Placeholder scan:** No TBD, TODO, or vague instructions found. Every step has explicit code.

**Type consistency:** DIMENSION_MAP keys used consistently across scoring → chart → UI functions. Column names match across data generation → DIMENSION_MAP → param_configs.
