# pages/3_财务分析.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np
import re
from difflib import get_close_matches

st.set_page_config(page_title="财务分析", layout="wide")
st.title("📊 财务分析")

company_code_norm = str(st.session_state.get("selected_company", "")).strip().zfill(6)

# ---------- 辅助函数：标准化公司代码（去除 .0，补零到6位）----------
def normalize_code(code):
    code_str = str(code).strip()
    if code_str.endswith('.0'):
        code_str = code_str[:-2]
    if code_str.isdigit():
        return code_str.zfill(6)
    else:
        return code_str

# ---------- 数据加载函数 ----------
@st.cache_data
def load_company_mapping():
    try:
        df = pd.read_excel("complete_company_industry_mapping_v4_stage16D_checked.xlsx", engine="openpyxl")
    except FileNotFoundError:
        try:
            df = pd.read_excel("complete_company_industry_mapping_v4_stage16D_checked.xls", engine="xlrd")
        except Exception as e:
            st.error(f"找不到公司-行业对照表文件: {e}")
            return pd.DataFrame()
    
    if df.shape[1] < 2:
        st.error("公司-行业对照表至少需要两列")
        return pd.DataFrame()
    
    code_col = df.columns[0]
    name_col = df.columns[1]
    df = df[[code_col, name_col]].copy()
    df.columns = ['stock_code_raw', 'company_name']
    df = df.dropna(subset=['company_name'])
    df['stock_code'] = df['stock_code_raw'].apply(normalize_code)
    return df[['stock_code', 'company_name']]

@st.cache_data
def load_dimension_json():
    try:
        with open("公司各维度雷达图数据（可直接用于streamlit）.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"读取各维度雷达图JSON失败: {e}")
        return {}

@st.cache_data
def load_comprehensive_json():
    try:
        with open("公司综合维度雷达图数据（可直接用于streamlit）.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"读取综合雷达图JSON失败: {e}")
        return {}

@st.cache_data
def load_raw_financial_indicators():
    """
    加载新数据源：company_statistics_with_raw_median_percentile_rank-1.csv
    列名规则：{指标名}_raw_value, {指标名}_raw_median, {指标名}_percentile, {指标名}_rank
    返回长表：company_code, indicator, company_value, industry_median, percentile, rank
    """
    try:
        df = pd.read_csv("company_statistics_with_raw_median_percentile_rank-1.csv", encoding='gbk', low_memory=False)
    except Exception as e:
        st.error(f"加载新数据源失败: {e}")
        return pd.DataFrame()
    
    code_col = None
    for col in df.columns:
        if col == 'stock_code_norm' or '公司代码' in col or 'code' in col.lower():
            code_col = col
            break
    if code_col is None:
        st.error("数据中未找到公司代码列")
        return pd.DataFrame()
    
    df[code_col] = df[code_col].apply(normalize_code)
    
    if 'accper' in df.columns:
        df['accper_date'] = pd.to_datetime(df['accper'], errors='coerce')
        df = df.sort_values([code_col, 'accper_date'], ascending=[True, False])
        df = df.drop_duplicates(subset=[code_col], keep='first')
    
    raw_value_cols = [col for col in df.columns if col.endswith('_raw_value')]
    if not raw_value_cols:
        st.error("未找到以 _raw_value 结尾的指标列")
        return pd.DataFrame()
    
    long_rows = []
    for _, row in df.iterrows():
        company_code = row[code_col]
        if pd.isna(company_code):
            continue
        for vcol in raw_value_cols:
            indicator_base = vcol[:-10]
            company_value = row[vcol]
            median_col = indicator_base + '_raw_median'
            percentile_col = indicator_base + '_percentile'
            rank_col = indicator_base + '_rank'
            industry_median = row[median_col] if median_col in df.columns else np.nan
            percentile = row[percentile_col] if percentile_col in df.columns else np.nan
            rank = row[rank_col] if rank_col in df.columns else np.nan
            if pd.isna(company_value):
                continue
            long_rows.append({
                'company_code': company_code,
                'indicator': indicator_base,
                'company_value': company_value,
                'industry_median': industry_median,
                'percentile': percentile,
                'rank': rank
            })
    result_df = pd.DataFrame(long_rows)
    return result_df

@st.cache_data
def load_dimension_scores():
    try:
        df = pd.read_csv("公司各纬度得分.csv", encoding='utf-8')
    except Exception as e:
        st.warning(f"加载各维度得分文件失败: {e}")
        return pd.DataFrame()
    
    if 'stock_code_norm' in df.columns:
        code_col = 'stock_code_norm'
    else:
        code_col = None
        for col in df.columns:
            if 'code' in col.lower() or '代码' in col:
                code_col = col
                break
    if code_col is None:
        st.warning("各维度得分文件中未找到公司代码列")
        return pd.DataFrame()
    df[code_col] = df[code_col].apply(normalize_code)
    return df

@st.cache_data
def load_yearly_financial():
    try:
        df = pd.read_csv("financial_with_classification.csv", encoding='utf-8')
    except Exception as e:
        st.warning(f"加载年度财务数据失败: {e}")
        return pd.DataFrame()
    
    code_col = None
    for col in df.columns:
        if 'stock_code_norm' in col or 'code' in col.lower():
            code_col = col
            break
    if code_col is None:
        st.warning("年度财务数据中未找到公司代码列")
        return pd.DataFrame()
    df[code_col] = df[code_col].apply(normalize_code)
    
    if 'accper' not in df.columns:
        st.warning("年度财务数据中未找到年份列 'accper'")
        return pd.DataFrame()
    
    def extract_year(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        match = re.search(r'\b(20\d{2}|19\d{2})\b', s)
        if match:
            return int(match.group(1))
        digits = re.findall(r'\d{4}', s)
        if digits:
            return int(digits[0])
        return np.nan
    
    df['year'] = df['accper'].apply(extract_year)
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    return df

# ---------- 辅助函数 ----------
def find_company_key(json_data, company_code, company_name):
    code_clean = str(company_code).lstrip('0').split('.')[0]
    candidates = [
        f"{company_name}_{code_clean}",
        f"{company_name}_{code_clean}.0",
        f"{company_name}_{company_code}",
        company_name,
        code_clean,
        f"{company_name} {code_clean}"
    ]
    for key in json_data.keys():
        for cand in candidates:
            if key == cand:
                return key
    for key in json_data.keys():
        if company_name in key and code_clean in key:
            return key
    for key in json_data.keys():
        if company_name in key:
            return key
    for key in json_data.keys():
        if code_clean in key:
            return key
    matches = get_close_matches(company_name, json_data.keys(), n=1, cutoff=0.6)
    if matches:
        return matches[0]
    return None

def get_company_dimensions(dim_json, company_code, company_name):
    key = find_company_key(dim_json, company_code, company_name)
    if not key:
        return {}
    company_data = dim_json[key]
    dimensions = {}
    for dim_key, dim_val in company_data.items():
        if isinstance(dim_val, dict) and "indicators" in dim_val:
            dim_name = dim_val.get("dimension_name", dim_key.replace("_score", ""))
            dimensions[dim_name] = {
                "indicators": dim_val["indicators"]
            }
    return dimensions

def get_real_values_for_dimensions(dimensions, real_df, company_code):
    company_data = real_df[real_df['company_code'] == normalize_code(company_code)]
    if company_data.empty:
        st.warning(f"未找到公司 {company_code} 的真实财务数据")
        enhanced = {}
        for dim, info in dimensions.items():
            n = len(info['indicators'])
            enhanced[dim] = {
                'indicators': info['indicators'],
                'company_values': [np.nan]*n,
                'industry_medians': [np.nan]*n,
                'percentiles': [np.nan]*n,
                'ranks': [np.nan]*n
            }
        return enhanced
    
    bench_map = {}
    for _, row in company_data.iterrows():
        bench_map[row['indicator']] = {
            'company_value': row['company_value'],
            'industry_median': row['industry_median'],
            'percentile': row['percentile'],
            'rank': row['rank']
        }
    
    enhanced = {}
    unmatched = []
    for dim, info in dimensions.items():
        indicators = info['indicators']
        company_vals = []
        industry_medians = []
        percentiles = []
        ranks = []
        for ind in indicators:
            bench = bench_map.get(ind)
            if bench is None:
                matches = get_close_matches(ind, bench_map.keys(), n=1, cutoff=0.7)
                if matches:
                    bench = bench_map[matches[0]]
                else:
                    unmatched.append(ind)
            if bench:
                company_vals.append(bench['company_value'])
                industry_medians.append(bench['industry_median'])
                percentiles.append(bench['percentile'])
                ranks.append(bench['rank'])
            else:
                company_vals.append(np.nan)
                industry_medians.append(np.nan)
                percentiles.append(np.nan)
                ranks.append(np.nan)
        enhanced[dim] = {
            'indicators': indicators,
            'company_values': company_vals,
            'industry_medians': industry_medians,
            'percentiles': percentiles,
            'ranks': ranks
        }
    if unmatched:
        st.warning(f"以下指标未在真实数据中找到匹配：{', '.join(set(unmatched))}")
    return enhanced

def get_comprehensive_radar_data(dimensions_data):
    dim_names = list(dimensions_data.keys())
    company_scores = []
    for dim in dim_names:
        vals = dimensions_data[dim]['company_values']
        if vals and not np.isnan(vals[0]):
            company_scores.append(vals[0])
        else:
            company_scores.append(0)
    return dim_names, company_scores

def create_radar_chart(indicators, company_values, title="", show_benchmark=False, benchmark_values=None):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=company_values,
        theta=indicators,
        fill='toself',
        name='公司',
        line_color='#2E86AB'
    ))
    if show_benchmark and benchmark_values is not None:
        fig.add_trace(go.Scatterpolar(
            r=benchmark_values,
            theta=indicators,
            fill='none',
            name='行业均值',
            line_color='#A23B72'
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title=title, height=400)
    return fig

def create_bar_chart(dim_names, values_2024, values_5y, title):
    fig = go.Figure()
    fig.add_trace(go.Bar(name='2024年', x=dim_names, y=values_2024, marker_color='#2E86AB'))
    fig.add_trace(go.Bar(name='5年均值', x=dim_names, y=values_5y, marker_color='#A23B72'))
    fig.update_layout(title=title, xaxis_title="维度", yaxis_title="得分", barmode='group', height=400)
    return fig

def create_line_chart(df, x_col, y_col, title, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines+markers',
                             name='公司', line_color='#2E86AB'))
    fig.update_layout(title=title, xaxis_title="年份", yaxis_title=y_label, height=300)
    return fig

# ---------- 主程序 ----------
if 'selected_company' not in st.session_state or st.session_state.selected_company is None:
    st.warning("请先在主页面侧边栏选择公司")
    st.stop()

selected_code_raw = st.session_state.selected_company
selected_code = normalize_code(selected_code_raw)

company_df = load_company_mapping()
if company_df.empty:
    st.stop()

matched = company_df[company_df['stock_code'] == selected_code]
if matched.empty:
    st.error(f"无法从公司-行业对照表中找到股票代码 {selected_code} 对应的公司名称")
    st.stop()

selected_name = matched.iloc[0]['company_name']

st.markdown(f"## 欢迎分析 {selected_name}")
st.markdown(f"**当前选择：{selected_code} - {selected_name}**")
st.markdown("---")

dim_json = load_dimension_json()
if not dim_json:
    st.error("无法加载各维度雷达图JSON数据，请检查文件。")
    st.stop()

dimensions_raw = get_company_dimensions(dim_json, selected_code, selected_name)
if not dimensions_raw:
    st.error(f"未在各维度JSON中找到公司 {selected_name} ({selected_code}) 的数据。")
    st.info(f"JSON中的键名示例：{list(dim_json.keys())[:3]}")
    st.stop()

real_df = load_raw_financial_indicators()
if real_df.empty:
    st.error("无法加载公司真实财务数据文件 'company_statistics_with_raw_median_percentile_rank-1.csv'")
    st.stop()

dimensions_data = get_real_values_for_dimensions(dimensions_raw, real_df, selected_code)

st.markdown("## 综合财务雷达图")
dim_names, comp_scores = get_comprehensive_radar_data(dimensions_data)
if dim_names:
    fig_comp = create_radar_chart(dim_names, comp_scores, title="公司各维度综合得分", show_benchmark=False)
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("无可用维度数据，无法绘制综合雷达图")

service = st.session_state.ai_service   

with st.expander("🤖 AI解读：综合财务雷达图"):
    with st.spinner("AI 分析中..."):
        reclass_text = service.analyze_comprehensive_financial_radar(stock_code=company_code_norm)
    st.markdown(reclass_text)

st.markdown("---")

st.markdown("#### 维度得分趋势对比（2024年 vs 5年均值）")
try:
    score_df = load_dimension_scores()
    if score_df.empty:
        st.info("未加载各维度得分文件")
    else:
        code_col = 'stock_code_norm' if 'stock_code_norm' in score_df.columns else None
        if code_col is None:
            for col in score_df.columns:
                if 'code' in col.lower() or '代码' in col:
                    code_col = col
                    break
        if code_col is None:
            st.warning("各维度得分文件中未找到公司代码列")
        else:
            score_df[code_col] = score_df[code_col].astype(str).str.strip()
            score_df[code_col] = score_df[code_col].apply(
                lambda x: x.zfill(6) if x.replace('.', '').isdigit() else x
            )
            company_scores = score_df[score_df[code_col] == selected_code]
            if company_scores.empty:
                st.info(f"未找到股票代码 {selected_code} 对应的各维度得分数据")
            else:
                dim_score_cols = [col for col in company_scores.columns if col.endswith('_score') and col != 'composite_score']
                dim_names_bar = [col.replace('_score', '') for col in dim_score_cols]
                data_2024 = company_scores[company_scores['year'] == 2024]
                years = [2020, 2021, 2022, 2023, 2024]
                data_5y = company_scores[company_scores['year'].isin(years)]
                values_2024 = []
                values_5y_mean = []
                if not data_2024.empty:
                    row_2024 = data_2024.iloc[0]
                    for col in dim_score_cols:
                        val = row_2024[col]
                        values_2024.append(val if pd.notna(val) else 0)
                else:
                    values_2024 = [0] * len(dim_score_cols)
                if not data_5y.empty:
                    for col in dim_score_cols:
                        mean_val = data_5y[col].mean()
                        values_5y_mean.append(mean_val if pd.notna(mean_val) else 0)
                else:
                    values_5y_mean = [0] * len(dim_score_cols)
                if dim_names_bar:
                    fig_bar = create_bar_chart(dim_names_bar, values_2024, values_5y_mean, "各维度得分对比")
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("未找到可用的维度得分列")
except FileNotFoundError:
    st.info("未找到各维度得分文件 '公司各纬度得分.csv'，将跳过趋势对比")
except Exception as e:
    st.info(f"无法加载各维度得分数据（{e}），将跳过趋势对比")

with st.expander("🤖 AI解读：维度得分趋势对比"):
    with st.spinner("AI 分析中..."):
        reclass_text = service.analyze_dimension_trend(stock_code=company_code_norm)
    st.markdown(reclass_text)

st.markdown("---")

yearly_df = load_yearly_financial()
yearly_company = pd.DataFrame()
if not yearly_df.empty:
    code_col = None
    for col in yearly_df.columns:
        if 'stock_code_norm' in col or 'code' in col.lower():
            code_col = col
            break
    if code_col:
        yearly_company = yearly_df[yearly_df[code_col] == selected_code].sort_values('year')
    else:
        yearly_company = yearly_df

indicator_to_column = {
    '总资产利润率ROA': '总资产利润率ROA',
    '权益资本利润率ROE': '权益资本利润率ROE',
    '营业利润率': '营业利润率',
    '销售利润率': '销售利润率',
    'EBITDA利润率': 'EBITDA利润率',
    '总资产盈利能力': '总资产盈利能力',
    '应收账款周转率': '应收账款周转率',
    '存货周转率': '存货周转率',
    '固定资产周转率': '固定资产周转率',
    '总资产周转率': '总资产周转率',
    '流动比率': '流动比率',
    '速动比率': '速动比率',
    '现金比率': '现金比率',
    '利息保障倍数': '利息保障倍数',
    '资产负债率': '资产负债率',
    '权益资产比': '权益资产比',
    '权益负债比': '权益负债比',
    '销售创现率': '销售创现率',
    '利润创现率_净利润': '利润创现率_净利润',
    '总资产创现率': '总资产创现率',
    '每股收益': '每股收益',
    '每股净资产': '每股净资产',
    '每股经营现金流': '每股经营现金流',
    '息税前利润EBIT': '息税前利润EBIT',
    '息税折旧摊销前收入EBITDA': '息税折旧摊销前收入EBITDA',
    '投入资本利润率ROIC': '投入资本利润率ROIC',
    '应收账款周转天数': '应收账款周转天数',
    '存货周转天数': '存货周转天数',
    '应付账款周转率': '应付账款周转率',
    '营运资本周转率': '营运资本周转率',
    '营运资本': '营运资本',
    '现金流利息保障倍数': '现金流利息保障倍数',
    '长期负债权益比率': '长期负债权益比率',
    '营运资本需求量比率': '营运资本需求量比率',
    '营运资本周转天数': '营运资本周转天数',
    '利润创现率_息税前利润': '利润创现率_息税前利润',
    '每股分红': '每股分红',
    '总资产负债率': '总资产负债率',
    '市盈率': '市盈率',
}

for dim, dim_info in dimensions_data.items():
    indicators = dim_info['indicators']
    company_vals = dim_info['company_values']
    industry_vals = dim_info['industry_medians']
    percentiles = dim_info['percentiles']
    ranks = dim_info['ranks']
    
    with st.expander(f"📈 {dim}分析", expanded=(dim == "盈利能力指标" or dim == "盈利能力")):
        col_left, col_right = st.columns([1, 1.5])
        with col_left:
            radar = create_radar_chart(indicators, company_vals, title=f"{dim} - 公司雷达图", show_benchmark=False)
            st.plotly_chart(radar, use_container_width=True)
        with col_right:
            st.markdown("#### 核心指标对比")
            table_data = []
            for i, ind in enumerate(indicators):
                # 分位数保留三位小数，其他保持原始字符串
                percentile_display = f"{percentiles[i]:.3f}" if not pd.isna(percentiles[i]) else "N/A"
                company_display = str(company_vals[i]) if not pd.isna(company_vals[i]) else "N/A"
                industry_display = str(industry_vals[i]) if not pd.isna(industry_vals[i]) else "N/A"
                rank_display = str(int(ranks[i])) if not pd.isna(ranks[i]) else "N/A"
                table_data.append({
                    "指标": ind,
                    "公司值": company_display,
                    "行业中位数": industry_display,
                    "分位数": percentile_display,
                    "行业排行": rank_display
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
        
        st.markdown("#### 五年趋势变化")
        if not yearly_company.empty:
            available_indicators = []
            for ind in indicators:
                col_name = indicator_to_column.get(ind, ind)
                if col_name in yearly_company.columns:
                    available_indicators.append((ind, col_name))
            if available_indicators:
                selected_ind, selected_col = st.selectbox(
                    f"选择{dim}指标查看历年变化",
                    available_indicators,
                    format_func=lambda x: x[0],
                    key=f"trend_{dim}"
                )
                trend_data = yearly_company[['year', selected_col]].dropna()
                trend_data = trend_data.sort_values('year')
                if not trend_data.empty:
                    fig_line = create_line_chart(trend_data, 'year', selected_col,
                                                f"{selected_ind} 历年变化", selected_ind)
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.info(f"指标 {selected_ind} 无有效年度数据")
            else:
                st.info(f"{dim}维度下无可用年度趋势数据")
        else:
            st.info("未加载年度财务数据文件 'financial_with_classification.csv'")


    
    # 每个维度之间加分隔线（可选）
    st.markdown("---")


st.caption("数据来源：公司各维度雷达图JSON + 新数据源(raw) + 各维度得分CSV + 年度财务数据。分位数已四舍五入保留三位小数，其他数值保持原始精度。")