# Automotive Design Studio Workspace v2 — 设计文档

> 日期: 2026-07-23 | 状态: Approved | 重构范围: 完全重写 app.py

## 1. 概述

将 v1 "数据统计看板" 升级为 "高级汽车设计专家工作台 (Automotive Design Studio Workspace)"。核心变化：增加车辆视觉预览区，废弃钟形正态分布曲线，引入设计空间坐标系 (Design Space Map) 二维散点密度图。

## 2. 数据模型

### 2.1 竞品数据库：50 个聚类散点

按车型类型分 4 组，每组用 `np.random.normal` 生成：

| 组 | 数量 | 类型特征 |
|----|------|----------|
| 轿车-运动 | 17 | 低窗身比(~38%), 低窗台线角(~2.2°), 高轮高比(~48%) |
| 轿车-豪华 | 8 | 中窗身比(~42%), 中窗台线角(~3.0°), 高头部空间(~95mm) |
| SUV | 15 | 高窗身比(~43%), 高车高(~1680mm), 高头部空间(~100mm) |
| 猎装/跨界 | 10 | 中等参数，跨SUV和轿车之间 |

每个散点在 10 个底层参数 + `轮高车高比_pct` + `轴长比_pct` 派生参数均有数值。保留 `风格`、`类型`、`销量超5000` 标记字段。

### 2.2 维度映射（不变）

```python
DIMENSION_MAP = {
    "比例":     ["轮高车高比_pct", "轴长比_pct"],
    "姿态":     ["窗身比_pct", "窗台线夹角_deg"],
    "型面":     ["曲率平滑度", "特征线连续性"],
    "细节品质": ["家族特征匹配度", "间隙段差_mm"],
    "工程平衡": ["头部空间_mm", "视野下沿角_deg"],
}
```

### 2.3 所有可用作坐标轴的参数

10 个底层参数 + 2 个派生参数 = 12 个可选轴参数，通过下拉选择器让用户在 Tab 内自由配对 X/Y 轴。

## 3. 评分引擎（保留 v1 逻辑）

- `calculate_z_score_based_rating()` — Z-score + 风格加权，0-10 分
- `calculate_dimension_scores()` — 聚合子参数得分到维度
- 雷达图竞品基准线始终为 5.0

## 4. UI 架构：三大区域

### 4.1 顶部：Vehicle Visual Studio

```
st.columns(6)
  col1-5: 占位图 (正侧/正前/正后/前45°/后45°) + 视角名
  col6:   紧凑雷达图 (height=320, 仅展示概览)
```

- 5 张图片使用 `st.image()` + 灰色占位符 `np.ones((400,600,3))*40`
- 雷达图缩小版（`create_compact_radar()`），height=320

### 4.2 中下部：Design Space Analysis

**Tab 结构：**
```
st.tabs(["📐 比例", "🚙 姿态", "✨ 型面", "💎 细节品质", "⚙️ 工程平衡"])
```

**每个 Tab 内部：**
```
col_x, col_y = st.columns(2)
  col_x: X轴参数下拉选择器
  col_y: Y轴参数下拉选择器

create_design_space_map(df, x_param, y_param, car_x, car_y, df_hot)
  → Plotly Figure (二维散点 + 密度等高线 + 红色★)
```

### 4.3 左侧边栏（保留）

- 车型名称输入
- 造型风格选择
- 10 个参数滑块（5 组 expander）
- 竞品池多选框（默认全选 50 个散点）

## 5. 图表设计

### 5.1 紧凑雷达图 `create_compact_radar()`

- height=320，legend 放底部
- 与 v1 相同的数据结构，仅缩小尺寸

### 5.2 设计空间坐标系 `create_design_space_map()`

**图层（从底到顶）：**
1. 灰色小圆点散点 — 竞品池全部数据（size=8, color='#888', opacity=0.6）
2. 二维密度等高线 — `go.Histogram2dContour` 或 `px.density_contour`（配色 #42A5F5 蓝调，半透明）
3. 红色大五角星 — 在研车型位置（size=24, symbol='star', color='#EF5350', 白色描边）
4. 绿色菱形 — 销量>5000 爆款均值点（size=14, symbol='diamond', color='#66BB6A'）

**样式：**
- 模板: `plotly_dark`
- 网格线颜色: `rgba(255,255,255,0.04)` 极弱化
- 坐标轴标题: 映射中文 PARAM_LABELS
- height=450

## 6. 视觉主题

| 属性 | 值 |
|------|-----|
| 主题 | `plotly_dark` + Streamlit dark |
| 竞品散点 | `#888888` 灰色，opacity=0.6 |
| 密度等高线 | `#42A5F5` 蓝色调 |
| 在研车型标记 | `#EF5350` 红色 ★，size=24 |
| 爆款均值标记 | `#66BB6A` 绿色 ◆ |
| 网格线 | 极弱 `rgba(255,255,255,0.04)` |

## 7. 文件变更

```
app.py — 完全重写（~650行）
.streamlit/config.toml — 不变
requirements.txt — 不变
```

## 8. 验证标准

- [ ] `python -m streamlit run app.py` 无报错
- [ ] 顶部 5 列占位图正常显示
- [ ] 紧凑雷达图在顶部第 6 列正常渲染
- [ ] 5 个 Tab 均可切���，每个 Tab 内 X/Y 轴下拉选择器正常工作
- [ ] 散点图显示 ~50 个灰色竞品散点 + 密度等高线
- [ ] 红色五角星标记在研车型位置，滑动侧边栏滑块时实时移动
- [ ] 绿色菱形标记爆款均值位置
- [ ] 竞品池多选框可过滤散点
- [ ] 深色科技感主题统一
