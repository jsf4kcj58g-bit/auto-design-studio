"""
Automotive Design Studio Workspace — v3
高级汽车设计专家工作台 (Real Data Edition)
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import os, sys

st.set_page_config(
    page_title="Automotive Design Studio",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# 区块 0: 常量定义
# ═══════════════════════════════════════════════════════════════

# Excel 列名 → 内部简写映射（用于代码内部引用）
COL_MAP = {
    "车型": "车型",
    "类型": "类型",
    "车格": "车格",
    "造型风格": "造型风格",
    "C 轮胎高度": "轮胎高度_mm",
    "D 轴距": "轴距_mm",
    "E 车长": "车长_mm",
    "F 车宽": "车宽_mm",
    "G 离地高度": "离地高度_mm",
    "H 车高": "车高_mm",
    "I 车身高度": "车身高度_mm",
    "J 车窗高度": "车窗高度_mm",
    "窗身比": "窗身比_pct",
    "M 窗台线水平夹角": "窗台线夹角_deg",
    "N 前挡风夹角": "前挡风夹角_deg",
    "O 后挡风夹角": "后挡风夹角_deg",
    "P 引擎盖水平夹角": "引擎盖夹角_deg",
    "Q A柱水平夹角": "A柱夹角_deg",
    "R C柱水平夹角": "C柱夹角_deg",
    "FY B柱到轮眉Y向距离平均值": "B柱轮眉Y向_mm",
    "A 前保与 机盖转折 点离地高度": "前保离地_mm",
    "B1 前轮胎到 前轮眉间隙": "前轮眉间隙_mm",
    "B2 后轮胎到 后轮眉间隙": "后轮眉间隙_mm",
    "K 后轮眉到窗台线Z向": "后轮眉窗台线Z_mm",
    "L 前轮眉到引擎盖Z向": "前轮眉引擎盖Z_mm",
    "RH 尾灯Z向高度": "尾灯高度_mm",
    "TY 顶盖在尾翼处Y向宽度": "顶盖尾翼Y宽_mm",
    "是否连续5个月销量上5000": "销量超5000",
}

# 五步法维度 → 子参数映射
DIMENSION_MAP = {
    "比例":     ["轮高车高比_pct", "轴长比_pct"],
    "姿态":     ["窗身比_pct", "窗台线夹角_deg"],
    "型面":     ["型面评分", "曲面连续性评分"],
    "细节品质": ["品质评分", "间隙段差评分"],
    "工程平衡": ["工程评分", "视野空间评分"],
}

DIMENSION_ICONS = {
    "比例": "📐 比例", "姿态": "🚙 姿态", "型面": "✨ 型面",
    "细节品质": "💎 细节品质", "工程平衡": "⚙️ 工程平衡",
}

STYLE_PREFERENCES = {
    "运动": {"窗身比_pct": "low", "窗台线夹角_deg": "low", "轮高车高比_pct": "high"},
    "商务": {"窗身比_pct": "high", "窗台线夹角_deg": "neutral", "轮高车高比_pct": "high"},
    "硬派": {"窗身比_pct": "high", "窗台线夹角_deg": "high", "轮高车高比_pct": "high"},
    "稳重": {"窗身比_pct": "high", "窗台线夹角_deg": "high", "轮高车高比_pct": "neutral"},
}

# 所有可选作坐标轴的参数
AXIS_PARAMS = [
    "轮高车高比_pct", "轴长比_pct",
    "窗身比_pct", "窗台线夹角_deg",
    "前挡风夹角_deg", "后挡风夹角_deg",
    "引擎盖夹角_deg", "A柱夹角_deg", "C柱夹角_deg",
    "型面评分", "曲面连续性评分",
    "品质评分", "间隙段差评分",
    "工程评分", "视野空间评分",
]

PARAM_LABELS = {
    "轮高车高比_pct": "轮高车高比 (%)",
    "轴长比_pct": "轴长比 (%)",
    "窗身比_pct": "窗身比 (%)",
    "窗台线夹角_deg": "窗台线水平夹角 (°)",
    "前挡风夹角_deg": "前挡风夹角 (°)",
    "后挡风夹角_deg": "后挡风夹角 (°)",
    "引擎盖夹角_deg": "引擎盖水平夹角 (°)",
    "A柱夹角_deg": "A柱水平夹角 (°)",
    "C柱夹角_deg": "C柱水平夹角 (°)",
    "型面评分": "型面评分 (0-10)",
    "曲面连续性评分": "曲面连续性评分 (0-10)",
    "品质评分": "品质评分 (0-10)",
    "间隙段差评分": "间隙段差评分 (0-10)",
    "工程评分": "工程评分 (0-10)",
    "视野空间评分": "视野空间评分 (0-10)",
}


# ═══════════════════════════════════════════════════════════════
# 区块 1: 数据 I/O
# ═══════════════════════════════════════════════════════════════

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "benchmark_data_clean.xlsx")


@st.cache_data(ttl=3600)
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    读取 Excel 并清洗数据：
    - 清除列名中的换行符
    - 清洗百分比列（窗身比等）
    - 填充缺失值
    - 重命名列为内部简写
    - 计算派生参数
    - 生成虚拟维度列
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 找不到数据文件: {file_path}")

    df = pd.read_excel(file_path, engine='openpyxl')

    # 清洗列名：去除换行符和多余空格
    df.columns = [str(c).replace('\n', ' ').replace('\r', '').strip() for c in df.columns]

    # --- 清洗窗身比列（可能含 % 符号） ---
    if "窗身比" in df.columns:
        df["窗身比"] = df["窗身比"].apply(_clean_percentage)

    # --- 检查并清洗其他可能的百分比列 ---
    for col in df.columns:
        if col not in ("车型", "类型", "车格", "造型风格", "窗身比", "是否连续5个月销量上5000"):
            # 确保数值型
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 过滤脏数据：填充 NaN ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # --- 处理布尔列 ---
    if "是否连续5个月销量上5000" in df.columns:
        df["是否连续5个月销量上5000"] = df["是否连续5个月销量上5000"].astype(str).str.contains("是|✓|true|True|1", na=False)

    # --- 重命名为内部简写 ---
    for excel_col, internal_col in COL_MAP.items():
        if excel_col in df.columns:
            df[internal_col] = df[excel_col]

    # --- 计算派生参数 ---
    if "轮胎高度_mm" in df.columns and "车高_mm" in df.columns:
        df["轮高车高比_pct"] = (df["轮胎高度_mm"] / df["车高_mm"] * 100).round(1)

    if "轴距_mm" in df.columns and "车长_mm" in df.columns:
        df["轴长比_pct"] = (df["轴距_mm"] / df["车长_mm"] * 100).round(1)

    # --- 生成虚拟维度列（0-10 正态分布） ---
    rng = np.random.default_rng(42)
    n = len(df)

    # 型面相关虚拟列 — 与造型风格关联
    df["型面评分"] = _gen_virtual_score(n, rng, df, "造型风格",
        {"运动": (7.5, 1.2), "商务": (7.8, 1.1), "硬派": (7.0, 1.2), "稳重": (6.5, 1.3)},
        default=(7.0, 1.2))
    df["曲面连续性评分"] = _gen_virtual_score(n, rng, df, "造型风格",
        {"运动": (7.8, 1.0), "商务": (8.0, 1.0), "硬派": (7.5, 1.1), "稳重": (6.8, 1.2)},
        default=(7.2, 1.1))

    # 品质相关虚拟列
    df["品质评分"] = _gen_virtual_score(n, rng, df, "车格",
        {"大型": (8.0, 1.0), "中型": (7.0, 1.2), "SUV": (7.5, 1.1)},
        default=(6.8, 1.3))
    df["间隙段差评分"] = _gen_virtual_score(n, rng, df, "车格",
        {"大型": (7.8, 0.9), "中型": (6.8, 1.1), "SUV": (7.0, 1.2)},
        default=(6.5, 1.3))

    # 工程相关虚拟列
    df["工程评分"] = _gen_virtual_score(n, rng, df, "车格",
        {"大型": (7.5, 1.1), "中型": (6.8, 1.2), "SUV": (8.0, 1.0)},
        default=(7.0, 1.2))
    df["视野空间评分"] = _gen_virtual_score(n, rng, df, "车格",
        {"大型": (7.2, 1.2), "中型": (6.5, 1.3), "SUV": (8.5, 0.8)},
        default=(6.8, 1.3))

    return df


def _clean_percentage(val) -> float:
    """清洗百分比值：去除 % 符号，自动识别小数比例并转为百分比"""
    if isinstance(val, str):
        val = float(val.replace('%', '').strip())
    result = float(val) if pd.notna(val) else np.nan
    # 如果值是 0-1 之间的小数比例，乘以 100 转为百分比
    if not np.isnan(result) and 0 < result < 1:
        result = round(result * 100, 1)
    return result


def _gen_virtual_score(n: int, rng: np.random.Generator, df: pd.DataFrame,
                       group_col: str, style_means: dict, default: tuple) -> np.ndarray:
    """按分组生成正态分布的虚拟评分（0-10 裁剪）"""
    scores = np.zeros(n)
    default_mean, default_std = default
    for i in range(n):
        group = df[group_col].iloc[i] if group_col in df.columns else None
        mean, std = style_means.get(group, (default_mean, default_std))
        scores[i] = rng.normal(mean, std)
    return np.clip(np.round(scores, 1), 0, 10)


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

    # 竞品均值基准线
    competitor_means = {dim: 5.0 for dim in DIMENSION_MAP}

    return {"dimension_scores": dimension_scores, "sub_scores": sub_scores, "competitor_means": competitor_means}


# ═══════════════════════════════════════════════════════════════
# 区块 3: 图表工厂
# ═══════════════════════════════════════════════════════════════

def create_compact_radar(dimension_scores: dict, competitor_means: dict) -> go.Figure:
    """紧凑型小雷达图（用于顶部概览）"""
    dimensions = list(DIMENSION_MAP.keys())
    labels = [DIMENSION_ICONS[d] for d in dimensions]
    car_values = [dimension_scores.get(d, 5.0) for d in dimensions]
    comp_values = [competitor_means.get(d, 5.0) for d in dimensions]

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

    fig.add_trace(go.Scatterpolar(
        r=comp_values + [comp_values[0]],
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


def create_design_space_map(
    df: pd.DataFrame, x_param: str, y_param: str,
    car_x: float, car_y: float,
    hot_x: float, hot_y: float,
) -> go.Figure:
    """设计空间坐标系 — 2D 散点 + 密度等高线 + 在研车型⭐"""
    fig = go.Figure()

    # 图层1: 竞品散点 — 灰色圆点
    hover_text = df.get("车型", pd.Series(range(len(df)))).astype(str)
    fig.add_trace(go.Scatter(
        x=df[x_param], y=df[y_param],
        mode='markers',
        name='竞品车型',
        marker=dict(color='#78909C', size=8, opacity=0.55, symbol='circle'),
        text=hover_text,
        hovertemplate=f'%{{text}}<br>{PARAM_LABELS.get(x_param, x_param)}: %{{x:.2f}}<br>{PARAM_LABELS.get(y_param, y_param)}: %{{y:.2f}}<extra></extra>',
    ))

    # 图层2: 二维密度等高线
    try:
        x_vals = df[x_param].dropna().astype(float)
        y_vals = df[y_param].dropna().astype(float)
        if len(x_vals) > 5:
            hist2d = go.Histogram2dContour(
                x=x_vals, y=y_vals,
                colorscale=[[0, 'rgba(66,165,245,0)'], [1, 'rgba(66,165,245,0.35)']],
                showscale=False,
                contours=dict(coloring='lines', showlines=True, start=0.15, end=0.85, size=0.08),
                line=dict(color='rgba(66,165,245,0.45)', width=1),
                name='密度等高线',
                hoverinfo='none',
            )
            fig.add_trace(hist2d)
    except Exception:
        pass

    # 图层3: 红色大五角星 — 在研车型
    fig.add_trace(go.Scatter(
        x=[car_x], y=[car_y],
        mode='markers+text',
        name='在研车型',
        marker=dict(color='#EF5350', size=28, symbol='star',
                     line=dict(color='white', width=2)),
        text=['★ 在研'],
        textposition='top center',
        textfont=dict(color='#EF5350', size=13),
        hoverinfo='none',
    ))

    # 图层4: 绿色菱形 — 爆款均值（仅当有爆款数据时）
    if len(df[df.get("销量超5000", False) == True]) > 0:
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


# ═══════════════════════════════════════════════════════════════
# 区块 4: UI 渲染
# ═══════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame) -> dict:
    """左侧边栏：全局过滤 + 竞品搜索 + 在研参数"""
    st.sidebar.title("🎯 Design Studio Control")

    # ---- 刷新按钮 ----
    if st.sidebar.button("🔄 刷新数据库", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")

    # ---- 在研车型基本信息 ----
    car_name = st.sidebar.text_input("项目代号", value="Concept-X")

    # 动态获取实际存在的风格
    available_styles = sorted(df["造型风格"].dropna().unique().tolist()) if "造型风格" in df.columns else ["运动", "商务", "硬派", "稳重"]
    car_style = st.sidebar.selectbox("目标风格", available_styles, index=0)

    st.sidebar.markdown("---")

    # ---- 在研车型图片上传 ----
    st.sidebar.subheader("📷 在研车型图片")
    view_labels = ["正侧", "正前", "正后", "前45°", "后45°"]
    uploaded_images = {}
    with st.sidebar.expander("上传5个视角图片", expanded=False):
        for v in view_labels:
            uploaded_images[v] = st.file_uploader(
                f"{v}视图", type=["png", "jpg", "jpeg", "webp"],
                key=f"img_{v}", label_visibility="collapsed"
            )
            if uploaded_images[v] is None:
                st.caption(f"  ⬆ {v} — 未上传")

    st.sidebar.markdown("---")

    # ---- 全局过滤器 ----
    st.sidebar.subheader("🔍 竞品筛选")
    car_segments = ["全部"] + sorted(df["车格"].dropna().unique().tolist()) if "车格" in df.columns else ["全部"]
    selected_segment = st.sidebar.selectbox("选择车格", car_segments, index=0)

    styles = ["全部"] + sorted(df["造型风格"].dropna().unique().tolist()) if "造型风格" in df.columns else ["全部"]
    selected_style_filter = st.sidebar.selectbox("选择造型风格", styles, index=0)

    st.sidebar.markdown("---")

    # ---- 竞品车型搜索/多选 ----
    st.sidebar.subheader("🚙 对标竞品池")
    all_cars = sorted(df["车型"].dropna().unique().tolist())
    selected_cars = st.sidebar.multiselect(
        "搜索并选择对标车型",
        options=all_cars,
        default=all_cars[:min(15, len(all_cars))],
        help="输入车型名称搜索，勾选用于对比的车型",
    )

    st.sidebar.markdown("---")

    # ---- 在研车型参数滑块 ----
    st.sidebar.subheader("📐 在研车型参数")

    # 动态计算默认值（所选竞品的均值）
    df_filtered = _apply_filters(df, selected_segment, selected_style_filter, selected_cars)

    params = {}
    param_configs = {
        "📐 比例": [
            ("轮高车高比 (%)", "轮高车高比_pct", 38.0, 58.0),
            ("轴长比 (%)", "轴长比_pct", 52.0, 70.0),
        ],
        "🚙 姿态": [
            ("窗身比 (%)", "窗身比_pct", 25.0, 52.0),
            ("窗台线水平夹角 (°)", "窗台线夹角_deg", 0.5, 6.0),
        ],
        "✨ 型面 (虚拟)": [
            ("型面评分", "型面评分", 0.0, 10.0),
            ("曲面连续性评分", "曲面连续性评分", 0.0, 10.0),
        ],
        "💎 细节品质 (虚拟)": [
            ("品质评分", "品质评分", 0.0, 10.0),
            ("间隙段差评分", "间隙段差评分", 0.0, 10.0),
        ],
        "⚙️ 工程平衡 (虚拟)": [
            ("工程评分", "工程评分", 0.0, 10.0),
            ("视野空间评分", "视野空间评分", 0.0, 10.0),
        ],
    }

    for section, controls in param_configs.items():
        with st.sidebar.expander(section, expanded=True):
            for label, col_name, vmin, vmax in controls:
                default_val = float(df_filtered[col_name].mean()) if col_name in df_filtered.columns and len(df_filtered) > 0 else (vmin + vmax) / 2.0
                default_val = round(max(vmin, min(vmax, default_val)), 1)

                step = 0.5 if vmax <= 10.0 else 0.1
                params[col_name] = st.slider(
                    label, min_value=vmin, max_value=vmax,
                    value=default_val, step=step, key=col_name,
                )

    return {
        "car_name": car_name, "car_style": car_style,
        "selected_segment": selected_segment, "selected_style": selected_style_filter,
        "params": params, "selected_cars": selected_cars,
        "uploaded_images": uploaded_images,
    }


def _apply_filters(df: pd.DataFrame, segment: str, style: str, selected_cars: list) -> pd.DataFrame:
    """应用车格和风格过滤"""
    result = df.copy()
    if segment != "全部" and "车格" in result.columns:
        result = result[result["车格"] == segment]
    if style != "全部" and "造型风格" in result.columns:
        result = result[result["造型风格"] == style]
    if selected_cars:
        result = result[result["车型"].isin(selected_cars)]
    return result


def render_vehicle_studio(dimension_scores: dict, competitor_means: dict, uploaded_images: dict = None):
    """顶部核心视觉区：5列车型视角图 + 紧凑雷达图"""
    st.markdown("## 🚗 Vehicle Visual Studio")

    cols = st.columns(6)

    views = [
        ("正侧", "Side View"),
        ("正前", "Front View"),
        ("正后", "Rear View"),
        ("前45°", "Front 3/4"),
        ("后45°", "Rear 3/4"),
    ]

    uploaded_images = uploaded_images or {}

    for i, (cn_name, en_name) in enumerate(views):
        with cols[i]:
            img = uploaded_images.get(cn_name)
            if img is not None:
                st.image(img, caption=f"{cn_name} | {en_name}", use_container_width=True)
            else:
                placeholder = np.ones((280, 450, 3), dtype=np.uint8) * 30
                h, w = 280, 450
                placeholder[h//2-1:h//2+2, :, :] = 55
                placeholder[:, w//2-1:w//2+2, :] = 55
                st.image(placeholder, caption=f"{cn_name} | {en_name} (占位图)", use_container_width=True)

    with cols[5]:
        st.markdown("##### 🎯 五步法概览")
        fig_radar = create_compact_radar(dimension_scores, competitor_means)
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")


def render_design_space_tabs(df: pd.DataFrame, car_params: dict):
    """中下部：5个Tab，每个含X/Y轴选择器 + 设计空间散点密度图"""
    st.markdown("## 🔬 Design Space Analysis")

    tab_labels = [DIMENSION_ICONS[d] for d in DIMENSION_MAP.keys()]
    dim_names = list(DIMENSION_MAP.keys())
    tabs = st.tabs(tab_labels)

    df_hot = df[df["销量超5000"] == True] if "销量超5000" in df.columns else pd.DataFrame()

    axis_options = [(PARAM_LABELS.get(p, p), p) for p in AXIS_PARAMS if p in df.columns]
    display_to_key = {d: k for d, k in axis_options}

    for i, dim in enumerate(dim_names):
        with tabs[i]:
            sub_params = DIMENSION_MAP[dim]
            default_x = sub_params[0] if sub_params[0] in df.columns else axis_options[0][1]
            default_y = sub_params[1] if len(sub_params) > 1 and sub_params[1] in df.columns else axis_options[-1][1]

            # 计算默认索引
            keys_list = [k for _, k in axis_options]
            idx_x = keys_list.index(default_x) if default_x in keys_list else 0
            idx_y = keys_list.index(default_y) if default_y in keys_list else min(1, len(keys_list) - 1)

            col_sel_x, col_sel_y = st.columns([1, 1])
            display_options = [d for d, _ in axis_options]

            with col_sel_x:
                x_display = st.selectbox(
                    "🔴 X 轴参数", options=display_options, index=idx_x, key=f"x_axis_{dim}"
                )
            with col_sel_y:
                y_display = st.selectbox(
                    "🔵 Y 轴参数", options=display_options, index=idx_y, key=f"y_axis_{dim}"
                )

            x_param = display_to_key[x_display]
            y_param = display_to_key[y_display]

            car_x = float(car_params.get(x_param, df[x_param].mean()))
            car_y = float(car_params.get(y_param, df[y_param].mean()))
            hot_x = float(df_hot[x_param].mean()) if len(df_hot) > 0 else float(df[x_param].mean())
            hot_y = float(df_hot[y_param].mean()) if len(df_hot) > 0 else float(df[y_param].mean())

            fig_map = create_design_space_map(df, x_param, y_param, car_x, car_y, hot_x, hot_y)
            st.plotly_chart(fig_map, use_container_width=True)

            st.caption(
                f"在研车型 ({PARAM_LABELS.get(x_param, x_param)}={car_x:.1f}, "
                f"{PARAM_LABELS.get(y_param, y_param)}={car_y:.1f}) | "
                f"竞品池: {len(df)} 款"
            )


# ═══════════════════════════════════════════════════════════════
# 区块 5: 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    st.title("🚗 Automotive Design Studio")
    st.caption("高级汽车设计专家工作台 · 真实数据库驱动 · 设计空间分析")

    # === 数据加载 ===
    try:
        df = load_and_clean_data(EXCEL_PATH)
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.info("请确保 benchmark data.xlsx 文件在应用同级目录下。")
        return
    except Exception as e:
        st.error(f"❌ 数据加载失败: {e}")
        return

    # === 侧边栏 ===
    inputs = render_sidebar(df)

    if not inputs["selected_cars"]:
        st.warning("⚠️ 请在侧边栏「对标竞品池」中至少选择一款车型")
        return

    # === 过滤竞品数据 ===
    df_comp = _apply_filters(df, inputs["selected_segment"], inputs["selected_style"], inputs["selected_cars"])

    if len(df_comp) == 0:
        st.warning("⚠️ 当前过滤条件下没有匹配的竞品车型，请调整筛选条件")
        return

    # === 计算评分 ===
    result = calculate_dimension_scores(inputs["params"], df_comp, inputs["car_style"])

    # === 顶部信息栏 ===
    st.markdown(
        f"**项目代号:** {inputs['car_name']} | "
        f"**风格:** {inputs['car_style']} | "
        f"**车格:** {inputs['selected_segment']} | "
        f"**对标竞品:** {len(df_comp)} 款"
    )
    st.markdown("---")

    # === 渲染 ===
    render_vehicle_studio(result["dimension_scores"], result["competitor_means"], inputs.get("uploaded_images"))
    render_design_space_tabs(df_comp, inputs["params"])

    st.markdown("---")
    st.caption(f"© 2026 Automotive Design Studio Workspace v3 | 数据库: {len(df)} 款车型 | Powered by Streamlit + Plotly")


if __name__ == "__main__":
    main()
