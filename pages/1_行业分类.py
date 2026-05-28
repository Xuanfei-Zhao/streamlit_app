# pages/行业分类.py
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from typing import Dict, List, Optional
import plotly.graph_objects as go
# ------------------------------



# 缓存加载函数
@st.cache_data
def load_industry_mapping():
    """加载公司-行业映射Excel，返回DataFrame，标准化股票代码为6位字符串"""
    file_path = Path("complete_company_industry_mapping_v5_qwen_level1_final.xlsx")
    if not file_path.exists():
        st.error("行业映射文件不存在，请检查文件路径")
        return pd.DataFrame()
    
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # 查找股票代码列（可能是 stock_code 或 symbol）
    code_col = None
    for col in ['stock_code', 'symbol']:
        if col in df.columns:
            code_col = col
            break
    if code_col is None:
        st.error(f"找不到股票代码列，可用列：{df.columns.tolist()}")
        return pd.DataFrame()
    
    # 标准化：统一转为字符串，前补零至6位
    df['symbol'] = df[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
    
    # 查找公司名称列
    name_col = None
    for col in ['stock_name', 'company_name', 'name']:
        if col in df.columns:
            name_col = col
            break
    if name_col:
        df['name'] = df[name_col]
    else:
        df['name'] = '未知'
    
    return df

#加载行业TOP10关键词
@st.cache_data
def load_industry_top10_keywords():
    """加载行业TOP10关键词宽表（无年份）"""
    import glob
    files = glob.glob("step5_industry_top10_keywords_wide_all.csv")
    if not files:
        st.warning("未找到行业TOP10关键词文件")
        return pd.DataFrame()
    
    # 根据后缀选择读取方式
    if files[0].endswith('.csv'):
        df = pd.read_csv(files[0], encoding='utf-8')
    else:
        df = pd.read_excel(files[0], engine='openpyxl')

    if 'industry_label' in df.columns and 'industry' not in df.columns:
        df.rename(columns={'industry_label': 'industry'}, inplace=True)

    return df

@st.cache_data
def load_similar_table():
    # 请修改为你的实际文件路径
    df = pd.read_csv("similar_company_top10_5year_tfidf_svd_v2.csv")  # 或 .xlsx
    # 标准化股票代码（如果需要）
    df['source_stock_code'] = df['source_stock_code'].astype(str).str.zfill(6)
    df['target_stock_code'] = df['target_stock_code'].astype(str).str.zfill(6)
    return df

#加载跨行业数据
@st.cache_data
def load_cross_industry():
    file_path = Path("stage17_final_cross_industry_mapping_table(2).xlsx")
    if not file_path.exists():
        st.warning(f"跨行业文件不存在: {file_path}")
        return pd.DataFrame()
    df = pd.read_excel(file_path, engine='openpyxl')
    # 确定股票代码列
    if 'stock_code_norm' in df.columns:
        code_col = 'stock_code_norm'
    elif 'stock_code' in df.columns:
        code_col = 'stock_code'
    else:
        st.warning("跨行业文件中无股票代码列")
        return pd.DataFrame()
    
    # 标准化：去除 .0 后缀，再补零到6位
    df['stock_code'] = df[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
    return df

def format_confidence(val):
    """将数值置信度转为 数值 (等级) 的显示格式"""
    if val is None or pd.isna(val):
        return "N/A"
    if val >= 0.8:
        level = "高"
    elif val >= 0.6:
        level = "中"
    else:
        level = "低"
    return f"{val:.4f} ({level})"


def normalize_code(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none", "null"]:
        return np.nan
    return s

def extract_keyword_long(row: pd.Series, level_display_name: str) -> pd.DataFrame:
    rows = []
    for i in range(1, 11):
        kw_col = f"keyword_{i}"
        score_col = f"keyword_{i}_score"
        if kw_col not in row.index or score_col not in row.index:
            continue
        keyword = row[kw_col]
        score = row[score_col]
        if pd.notna(keyword) and str(keyword).strip() != "" and pd.notna(score):
            rows.append({
                "level": level_display_name,
                "keyword": str(keyword).strip(),
                "raw_score": float(score),
                "industry_code": row.get("industry_code", ""),
                "industry_label": row.get("industry_label", ""),
                "industry_company_count": row.get("industry_company_count", np.nan),
            })
    return pd.DataFrame(rows)

def get_company_hierarchy_keywords(
    company_row: pd.Series,
    keyword_df: pd.DataFrame,
    sample_version: str
) -> pd.DataFrame:
    """根据公司一级/二级/三级代码从关键词表提取对应层级的关键词"""
    level_info = [
        {"industry_level": 1, "code": company_row.get("final_level1_code", np.nan),
         "label": company_row.get("final_level1_label", ""), "prefix": "一级行业"},
        {"industry_level": 2, "code": company_row.get("final_level2_code", np.nan),
         "label": company_row.get("final_level2_label", ""), "prefix": "二级行业"},
        {"industry_level": 3, "code": company_row.get("final_level3_code", np.nan),
         "label": company_row.get("final_level3_label", ""), "prefix": "三级行业"}
    ]
    all_dfs = []
    for item in level_info:
        if pd.isna(item["code"]):
            continue
        matched = keyword_df[
            (keyword_df["sample_version"] == sample_version) &
            (keyword_df["industry_level"] == item["industry_level"]) &
            (keyword_df["industry_code"] == item["code"])
        ]
        if matched.empty:
            continue
        row = matched.iloc[0]
        label = row.get("industry_label", item["label"])
        level_display_name = f"{item['prefix']}：{label}"
        all_dfs.append(extract_keyword_long(row, level_display_name))
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

def build_heatmap_matrix(long_df: pd.DataFrame, normalize: bool = True):
    if long_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = long_df.copy()
    if normalize:
        df["score_for_plot"] = df.groupby("level")["raw_score"].transform(
            lambda x: x / x.max() if x.max() and x.max() != 0 else x
        )
    else:
        df["score_for_plot"] = df["raw_score"]
    heatmap_df = df.pivot_table(
        index="level", columns="keyword", values="score_for_plot",
        aggfunc="max", fill_value=0
    )
    raw_df = df.pivot_table(
        index="level", columns="keyword", values="raw_score",
        aggfunc="max", fill_value=0
    )
    level_order = list(df["level"].drop_duplicates())
    keyword_order = list(df["keyword"].drop_duplicates())
    heatmap_df = heatmap_df.reindex(index=level_order, columns=keyword_order)
    raw_df = raw_df.reindex(index=level_order, columns=keyword_order)
    return heatmap_df, raw_df

def plot_hierarchy_keyword_heatmap(long_df: pd.DataFrame, normalize: bool = True, show_text: bool = True):
    heatmap_df, raw_df = build_heatmap_matrix(long_df, normalize=normalize)
    if heatmap_df.empty:
        st.warning("没有可展示的关键词热力图数据。")
        return
    color_title = "归一化得分" if normalize else "原始得分"
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns.tolist(),
            y=heatmap_df.index.tolist(),
            customdata=raw_df.values,
            colorscale="Blues",
            colorbar=dict(title=color_title),
            hovertemplate=(
                "行业层级：%{y}<br>关键词：%{x}<br>" +
                f"{color_title}：%{{z:.4f}}<br>原始得分：%{{customdata:.4f}}<extra></extra>"
            )
        )
    )
    if show_text:
        annotations = []
        for i, y in enumerate(heatmap_df.index):
            for j, x in enumerate(heatmap_df.columns):
                value = heatmap_df.iloc[i, j]
                if value > 0:
                    annotations.append(dict(x=x, y=y, text=f"{value:.2f}", showarrow=False, font=dict(size=10)))
        fig.update_layout(annotations=annotations)
    fig.update_layout(
        title="一级 / 二级 / 三级行业关键词联动比较热力图",
        height=440,
        margin=dict(l=20, r=20, t=70, b=30),
        xaxis=dict(title="关键词", tickangle=45),
        yaxis=dict(title="行业层级"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# 主页面
# ------------------------------
st.set_page_config(layout="wide")
st.title("🔍 行业分类分析")

# 1. 获取当前选中的公司
company_code = st.session_state.get("selected_company", None)
if not company_code:
    st.warning("请先在侧边栏选择一家公司")
    st.stop()

# 2. 加载数据
df = load_industry_mapping()
if df.empty:
    st.error("无法加载行业数据，请检查文件")
    st.stop()

# 3. 标准化匹配
company_code_norm = str(company_code).strip().zfill(6)
matched = df[df['symbol'] == company_code_norm]

if matched.empty:
    st.error(f"未找到股票代码 {company_code} 的行业信息")
    # 调试：打印前几个代码供核对
    st.write("数据中的前5个股票代码：", df['symbol'].head().tolist())
    st.stop()

row = matched.iloc[0]
company_name = row.get('name', company_code)

st.markdown(f"### 当前公司：{company_name}（{company_code_norm}）")

# ------------------------------
# 提取行业信息
# ------------------------------
# 三级行业主名称（优先用 final_level3_label，否则用 industry_label）
industry_name = row.get('final_level3_label', 
                row.get('industry_label', 
                row.get('final_industry', '未提供')))

# 各级行业标签
level1 = row.get('final_level1_label', '')
level2 = row.get('final_level2_label', '')
level3 = row.get('final_level3_label', '')

# 各级置信度数值
conf1 = row.get('new_level1_confidence', None)
conf2 = row.get('new_level2_confidence', None)
conf3 = row.get('new_level3_confidence', None)

# 综合置信度等级（高/中/低）
confidence_level = row.get('final_confidence_group', '未知')

# 跨行业标识
is_cross = row.get('is_cross_industry', False)
secondary_industry = row.get('secondary_industry_label', '')


# ------------------------------
# 展示主行业卡片（含层级置信度）
# ------------------------------
    
st.subheader(f"📌 主行业：{industry_name}")
# 自定义卡片样式
st.markdown("""
<style>
.level-card {
    background-color: #f8f9fa;
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    height: 100%;
}
.level-icon {
    font-size: 2rem;
    margin-bottom: 8px;
}
.level-title {
    font-size: 0.9rem;
    color: #6c757d;
    margin-bottom: 8px;
}
.level-name {
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 12px;
}
.conf-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}
.conf-high { background-color: #d4edda; color: #155724; }
.conf-mid { background-color: #fff3cd; color: #856404; }
.conf-low { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# 定义辅助函数
def get_conf_class(conf):
    if conf >= 0.8: return "conf-high"
    elif conf >= 0.6: return "conf-mid"
    else: return "conf-low"

# 三个层级数据
levels = [
    ("一级行业", level1, conf1),
    ("二级行业", level2, conf2),
    ("三级行业", level3, conf3)
]

cols = st.columns(3)
for col, (title, name, conf) in zip(cols, levels):
    with col:
        if name:
            conf_display = f"{conf:.4f}" if conf else "N/A"
            conf_class = get_conf_class(conf) if conf else "conf-low"
            st.markdown(f"""
            <div class="level-card">
                <div class="level-title">{title}</div>
                <div class="level-name">{name}</div>
                <div><span class="conf-badge {conf_class}">置信度 {conf_display}</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # 空数据占位
            st.markdown(f"""
            <div class="level-card" style="opacity: 0.5;">
                <div class="level-icon">{icon}</div>
                <div class="level-title">{title}</div>
                <div class="level-name">暂无</div>
                <div><span class="conf-badge conf-low">—</span></div>
            </div>
            """, unsafe_allow_html=True)
    
with st.expander("📋 查看行业代码与字典标签", expanded=False):
    # 尝试获取短标签（如果存在）
    short1 = row.get('final_level1_short_label', level1)
    short2 = row.get('final_level2_short_label', level2)
    short3 = row.get('final_level3_short_label', level3)
    hierarchy_view = pd.DataFrame([
        {"层级": "一级行业", "行业代码": row.get('final_level1_code', ''), 
         "行业名称": level1, "短标签": short1},
        {"层级": "二级行业", "行业代码": row.get('final_level2_code', ''), 
         "行业名称": level2, "短标签": short2},
        {"层级": "三级行业", "行业代码": row.get('final_level3_code', ''), 
         "行业名称": level3, "短标签": short3}
    ])
    st.dataframe(hierarchy_view, use_container_width=True)

with st.expander("📊 分类依据（详细置信度数据 - 分级别）"):
    # 定义展示函数
    def show_level_confidence(level_num, level_name, conf_val, centroid=None, neighbor=None, top=None):
        """展示单个级别的置信度数据"""
        st.markdown(f"**{level_name}行业**")
        col1, col2 = st.columns(2)
        with col1:
            if conf_val is not None and not pd.isna(conf_val):
                st.metric(f"综合置信度", f"{conf_val:.4f}" if isinstance(conf_val, float) else str(conf_val))
            else:
                st.metric("综合置信度", "N/A")
        with col2:
            # 显示等级
            if conf_val is not None and not pd.isna(conf_val):
                if conf_val >= 0.8:
                    st.success("等级：高")
                elif conf_val >= 0.6:
                    st.info("等级：中")
                else:
                    st.warning("等级：低")
            else:
                st.metric("等级", "N/A")
        
        # 展示三个细分指标
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("质心相似度", f"{centroid:.4f}" if centroid else "N/A")
            st.metric("最相似公司行业相似度", f"{top:.4f}" if top else "N/A")
        with col_b:
            st.metric("同行业比例", f"{neighbor:.4f}" if neighbor else "N/A")
        st.divider()

    # 一级行业
    conf1 = row.get('new_level1_confidence', None)
    centroid1 = row.get('new_level1_confidence_centroid_similarity', None)
    neighbor1 = row.get('new_level1_confidence_neighbor_same_ratio', None)
    top1 = row.get('new_level1_confidence_top_same_similarity', None)
    show_level_confidence(1, "一级", conf1, centroid1, neighbor1, top1)

    # 二级行业
    conf2 = row.get('new_level2_confidence', None)
    centroid2 = row.get('new_level2_confidence_centroid_similarity', None)
    neighbor2 = row.get('new_level2_confidence_neighbor_same_ratio', None)
    top2 = row.get('new_level2_confidence_top_same_similarity', None)
    show_level_confidence(2, "二级", conf2, centroid2, neighbor2, top2)

    # 三级行业
    conf3 = row.get('new_level3_confidence', None)
    centroid3 = row.get('new_level3_confidence_centroid_similarity', None)
    neighbor3 = row.get('new_level3_confidence_neighbor_same_ratio', None)
    top3 = row.get('new_level3_confidence_top_same_similarity', None)
    show_level_confidence(3, "三级", conf3, centroid3, neighbor3, top3)

    st.caption("""
    **指标说明**：
    - **综合置信度**：该级别行业的最终置信度得分。
    - **质心相似度**：公司特征与行业平均特征的相似度（0~1，越高越典型）。
    - **同行业比例**：最近邻公司中属于该行业的比例（0~1）。
    - **最相似公司行业相似度**：最相似的前几个公司中，行业匹配的强度。
    """)

# AI解读
service = st.session_state.ai_service   # 确保这行在调用之前执行

with st.expander("🤖 AI解读：行业重分类"):
    with st.spinner("AI 分析中..."):
        reclass_text = service.analyze_industry_reclassification(stock_code=company_code_norm)
    st.markdown(reclass_text)

# ------------------------------
# 行业TOP10关键词（表格形式）
# ----------------------------

st.divider()
st.subheader("🏷️ 行业TOP10关键词")

df_kw = load_industry_top10_keywords()
if not df_kw.empty:
    matched_kw = df_kw[df_kw['industry'] == industry_name]
    if not matched_kw.empty:
        row_kw = matched_kw.iloc[0]
        table_data = []
        for i in range(1, 11):
            kw = row_kw.get(f"keyword_{i}", "")
            score = row_kw.get(f"keyword_{i}_score", None)
            if pd.notna(kw) and kw:
                table_data.append({
                    "排名": i,
                    "关键词": kw,
                    "权重": score if pd.notna(score) else None
                })
        if table_data:
            # HTML表格
            html = """
            <style>
            .keyword-table {
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
            }
            .keyword-table th, .keyword-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            .keyword-table th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            </style>
            <table class="keyword-table">
                <thead>
                    <tr><th>排名</th><th>关键词</th><th>权重</th></tr>
                </thead>
                <tbody>
            """
            for row_item in table_data:
                weight_display = f"{row_item['权重']:.3f}" if row_item['权重'] is not None else "—"
                html += f"<tr><td>{row_item['排名']}</td><td>{row_item['关键词']}</td><td>{weight_display}</td></tr>"
            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("无有效关键词数据")
    else:
        st.info(f"未找到行业「{industry_name}」的关键词数据")
else:
    st.info("行业关键词文件未加载")

with st.expander("🤖 AI解读：行业关键词"):
    with st.spinner("AI 分析中..."):
        reclass_text = service.analyze_industry_keywords(stock_code=company_code_norm)
    st.markdown(reclass_text)

st.divider()
st.subheader("📊 一级 / 二级 / 三级行业关键词联动比较")

# 获取可用的样本版本（关键词表中的 sample_version）
if not df_kw.empty and 'sample_version' in df_kw.columns:
    versions = sorted(df_kw['sample_version'].dropna().astype(str).unique())
    default_version = "hard_plus_soft" if "hard_plus_soft" in versions else versions[0] if versions else None
    if default_version:
        # 允许用户选择版本（简洁起见放在 expander 或直接使用默认）
        with st.expander("热力图设置"):
            selected_version = st.selectbox("关键词样本版本", versions, index=versions.index(default_version))
            normalize = st.checkbox("按每个行业层级归一化", value=True)
            show_text = st.checkbox("在热力图中显示数值", value=True)
    else:
        selected_version = None
        normalize = True
        show_text = True
else:
    selected_version = None
    normalize = True
    show_text = True

if selected_version and not df_kw.empty:
    hierarchy_kw_df = get_company_hierarchy_keywords(row, df_kw, selected_version)
    if hierarchy_kw_df.empty:
        st.warning("未能提取到层级关键词数据（可能公司代码未匹配或关键词表中无对应层级）")
    else:
        plot_hierarchy_keyword_heatmap(hierarchy_kw_df, normalize=normalize, show_text=show_text)
        st.info("该热力图展示目标公司所属一级、二级、三级行业的核心关键词结构。颜色越深，关键词在该层级代表性越强。")
else:
    st.info("关键词表缺少 sample_version 列或无可选版本，无法展示热力图。")

st.subheader("📋 关键词明细")
if selected_version and 'hierarchy_kw_df' in locals() and not hierarchy_kw_df.empty:
    detail_df = hierarchy_kw_df.copy()
    # 从 level 列提取行业名称（格式："一级行业：商业银行" -> "商业银行"）
    detail_df['行业名称'] = detail_df['level'].str.split('：').str[-1]
    detail_df = detail_df[[
        "level", "industry_code", "行业名称", "industry_company_count", "keyword", "raw_score"
    ]].rename(columns={
        "level": "行业层级",
        "industry_code": "行业代码",
        "industry_company_count": "行业公司数量",
        "keyword": "关键词",
        "raw_score": "关键词原始得分"
    })
    st.dataframe(detail_df, use_container_width=True)
else:
    st.info("没有关键词明细可展示。")

# ------------------------------
# 同行业相似公司 TOP10
# ------------------------------
st.divider()
st.subheader("🏆 同行业相似公司 TOP10")

df_similar = load_similar_table()

# 筛选当前公司作为源公司的数据
current_similar = df_similar[df_similar['source_stock_code'] == company_code_norm]
if current_similar.empty:
    st.info("未找到当前公司的相似公司数据")
else:
    # 按 rank 排序，取前10
    top10 = current_similar.sort_values('rank').head(10)
    
    # 构建名称映射（用于从主表获取公司名称，如果表格中已有 target_company 则直接用）
    # 但表格中已有 target_company，所以不需要映射
    result = []
    for _, row in top10.iterrows():
        result.append({
            "排名": row['rank'],
            "股票代码": row['target_stock_code'],
            "公司名称": row['target_company'],
            "相似度": round(row['similarity'], 4) if 'similarity' in row else None
        })
    
    if result:
        # 构建 HTML 表格
        html_table = """
        <style>
        .similar-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        .similar-table th, .similar-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .similar-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        </style>
        <table class="similar-table">
            <thead>
                <tr><th>排名</th><th>股票代码</th><th>公司名称</th><th>相似度</th></tr>
            </thead>
            <tbody>
        """
        for row_item in result:
            sim_val = row_item['相似度']
            sim_display = f"{sim_val:.3f}" if sim_val is not None else "—"
            # 可选：为高相似度添加颜色
            if sim_val is not None and sim_val >= 0.8:
                sim_display = f"<span style='color:green;font-weight:bold;'>{sim_display}</span>"
            elif sim_val is not None and sim_val >= 0.6:
                sim_display = f"<span style='color:orange;'>{sim_display}</span>"
            html_table += f"""
            <tr>
                <td>{row_item['排名']}</td>
                <td>{row_item['股票代码']}</td>
                <td>{row_item['公司名称']}</td>
                <td>{sim_display}</td>
            </tr>
            """
        html_table += "</tbody></table>"
        from streamlit.components.v1 import html
        html(html_table, height=400, scrolling=True)
    else:
        st.info("未找到相似公司")

with st.expander("🤖 AI解读：相似公司TOP10"):
    with st.spinner("AI 分析中..."):
        reclass_text = service.analyze_similar_companies(stock_code=company_code_norm)
    st.markdown(reclass_text)

st.divider()
st.subheader("🗂️ 跨行业识别")

# ------------------------------
# 跨行业识别
# ------------------------------
df_cross = load_cross_industry()
cross_row = None
if not df_cross.empty:
    matched = df_cross[df_cross['stock_code'] == company_code_norm]
    if not matched.empty:
        cross_row = matched.iloc[0]

def get_conf_color(conf):
    if conf >= 0.8: return "#28a745"
    elif conf >= 0.6: return "#ffc107"
    else: return "#dc3545"

st.markdown("""
<style>
.cross-section {
    margin-bottom: 24px;
}
.cross-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 12px;
    color: #1f1f1f;
}
.cross-item {
    background-color: #f8f9fa;
    border-radius: 16px;
    padding: 10px 20px;
    margin-bottom: 8px;
    border: 1px solid #e9ecef;
}
.cross-item-text {
    font-size: 1rem;
}
.cross-item-name {
    font-weight: 500;
}
.cross-item-conf {
    font-weight: 600;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

has_cross = False
if cross_row is not None:
    main_industry = row.get('final_level3_label', '')
    levels = [
        ('一级行业', 'mapped_alt_level1_labels', 'mapped_alt_level1_weights'),
        ('二级行业', 'mapped_alt_level2_labels', 'mapped_alt_level2_weights'),
        ('三级行业', 'mapped_alt_level3_labels', 'mapped_alt_level3_weights')
    ]
    for level_name, labels_col, weights_col in levels:
        labels_str = cross_row.get(labels_col, '')
        weights_str = cross_row.get(weights_col, '')
        if pd.notna(labels_str) and isinstance(labels_str, str) and labels_str.strip():
            labels = [l.strip() for l in labels_str.split(';') if l.strip()]
            weights = []
            if pd.notna(weights_str) and isinstance(weights_str, str):
                try:
                    weights = [float(w.strip()) for w in weights_str.split(';') if w.strip()]
                except:
                    weights = []
            while len(weights) < len(labels):
                weights.append(0.5)
            if level_name == '三级行业':
                filtered = [(l, w) for l, w in zip(labels, weights) if l != main_industry]
            else:
                filtered = list(zip(labels, weights))
            if filtered:
                has_cross = True
                st.markdown(f'<div class="cross-section">', unsafe_allow_html=True)
                st.markdown(f'<div class="cross-title">{level_name}</div>', unsafe_allow_html=True)
                for label, weight in sorted(filtered, key=lambda x: x[1], reverse=True):
                    color = get_conf_color(weight)
                    st.markdown(f'''
                    <div class="cross-item">
                        <div class="cross-item-text">
                            <span class="cross-item-name">{label}</span>
                            <span class="cross-item-conf" style="color:{color};">({weight:.1%})</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

if not has_cross:
    st.info("未检测到显著的跨行业特征")