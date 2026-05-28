import streamlit as st
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.data_loader import (
    load_stock_list,
    get_company_basic_info,
    calculate_industry_rankings,
    clean_business_description,
    get_company_business_keywords,
    get_company_financial_rankings,
    get_company_radar_data,
    get_company_indicator_radar_data
)
from utils.charts import create_radar_chart, create_line_chart

st.set_page_config(page_title="公司概况", page_icon="🏢", layout="wide")

company_code_norm = str(st.session_state.get("selected_company", "")).strip().zfill(6)

if 'selected_company' not in st.session_state:
    st.warning("⚠️ 请先在侧边栏选择一家公司")
    st.stop()

stock_code = st.session_state.selected_company

st.title(f"🏢 {stock_code} - 公司概况")

with st.spinner("加载公司数据..."):
    basic_info = get_company_basic_info(stock_code)

if not basic_info:
    st.error("无法获取公司信息")
    st.stop()

st.header("📋 基本信息")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="股票代码", value=basic_info['股票代码'])

with col2:
    st.metric(label="公司名称", value=basic_info['公司名称'])

with col3:
    st.info(
        f"**所属行业**\n\n"
        f"一级: {basic_info['所属行业一级']}\n\n"
        f"二级: {basic_info['所属行业二级']}\n\n"
        f"三级: {basic_info['所属行业三级']}"
    )


st.divider()

st.header("💼 主营业务")

# 获取行业关键词数据
with st.spinner("加载主营业务数据..."):
    business_keywords = get_company_business_keywords(stock_code)

if business_keywords is not None:
    st.subheader("所属行业")
    st.info(
        f"**行业名称**: {business_keywords.get('industry_label', '未知')}\n\n"
        f"**行业层级**: L{business_keywords.get('industry_level', '未知')}\n\n"
        f"**行业代码**: {business_keywords.get('industry_code', '未知')}\n\n"
        f"**行业公司数量**: {business_keywords.get('industry_company_count', '未知')}家"
    )

    
    st.subheader("主营业务描述")
    top10_keywords = business_keywords.get('top10_keywords', '')
    if top10_keywords:
        description = f"本公司主要从事{top10_keywords}等相关业务领域。"
        st.markdown(description)
    else:
        st.warning("无法获取主营业务数据")

service = st.session_state.ai_service   # 确保这行在调用之前执行

with st.expander("🤖 AI解读：公司概况"):
    with st.spinner("AI 分析中..."):
        reclass_text = service.analyze_company_overview(stock_code=company_code_norm)
    st.markdown(reclass_text)


st.divider()

st.header("📊 行业内关键指标排名")

# 获取财务排名数据
with st.spinner("加载财务排名数据..."):
    financial_rankings = get_company_financial_rankings(stock_code)

if financial_rankings is not None:
    tab1, tab2, tab3 = st.tabs(["ROE排名", "营业利润率", "总资产利润率"])
    
    with tab1:
        roe = financial_rankings['roe']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="公司ROE", value=f"{roe['value']:.2f}")
        
        with col2:
            st.metric(label="行业中位数", value=f"{roe['median']:.2f}")
        
        with col3:
            st.metric(label="行业排名", value=f"{int(roe['rank']) if pd.notna(roe['rank']) else 'N/A'}")
        
        with col4:
            st.metric(label="百分位", value=f"{roe['percentile']*100:.1f}%")

    with st.expander("🤖 AI解读：ROE排名"):
        with st.spinner("AI 分析中..."):
            reclass_text = service.analyze_roe_ranking(stock_code=company_code_norm)
        st.markdown(reclass_text)
    
    with tab2:
        op_margin = financial_rankings['operating_margin']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="营业利润率", value=f"{op_margin['value']:.2f}")
        
        with col2:
            st.metric(label="行业中位数", value=f"{op_margin['median']:.2f}")
        
        with col3:
            st.metric(label="行业排名", value=f"{int(op_margin['rank']) if pd.notna(op_margin['rank']) else 'N/A'}")
        
        with col4:
            st.metric(label="百分位", value=f"{op_margin['percentile']*100:.1f}%")

    with st.expander("🤖 AI解读：营业利润率排名"):
        with st.spinner("AI 分析中..."):
            reclass_text = service.analyze_operating_margin_ranking(stock_code=company_code_norm)
        st.markdown(reclass_text)
    
    with tab3:
        roa = financial_rankings['roa']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="总资产利润率", value=f"{roa['value']:.2f}")
        
        with col2:
            st.metric(label="行业中位数", value=f"{roa['median']:.2f}")
        
        with col3:
            st.metric(label="行业排名", value=f"{int(roa['rank']) if pd.notna(roa['rank']) else 'N/A'}")
        
        with col4:
            st.metric(label="百分位", value=f"{roa['percentile']*100:.1f}%")
    


    with st.expander("🤖 AI解读：ROA排名"):
        with st.spinner("AI 分析中..."):
            reclass_text = service.analyze_roa_ranking(stock_code=company_code_norm)
        st.markdown(reclass_text)


    st.subheader("详细排名数据")

    ranking_data = pd.DataFrame({
        '指标': ['权益资本利润率(ROE)', '营业利润率', '总资产利润率(ROA)', 
                'EBITDA利润率', '总资产周转率', '流动比率', '资产负债率'],
        '公司值': [
            financial_rankings['roe']['value'],
            financial_rankings['operating_margin']['value'],
            financial_rankings['roa']['value'],
            financial_rankings['ebitda_margin']['value'],
            financial_rankings['asset_turnover']['value'],
            financial_rankings['current_ratio']['value'],
            financial_rankings['debt_ratio']['value']
        ],
        '行业中位数': [
            financial_rankings['roe']['median'],
            financial_rankings['operating_margin']['median'],
            financial_rankings['roa']['median'],
            financial_rankings['ebitda_margin']['median'],
            financial_rankings['asset_turnover']['median'],
            financial_rankings['current_ratio']['median'],
            financial_rankings['debt_ratio']['median']
        ],
        '行业排名': [
            int(financial_rankings['roe']['rank']) if pd.notna(financial_rankings['roe']['rank']) else '-',
            int(financial_rankings['operating_margin']['rank']) if pd.notna(financial_rankings['operating_margin']['rank']) else '-',
            int(financial_rankings['roa']['rank']) if pd.notna(financial_rankings['roa']['rank']) else '-',
            int(financial_rankings['ebitda_margin']['rank']) if pd.notna(financial_rankings['ebitda_margin']['rank']) else '-',
            int(financial_rankings['asset_turnover']['rank']) if pd.notna(financial_rankings['asset_turnover']['rank']) else '-',
            int(financial_rankings['current_ratio']['rank']) if pd.notna(financial_rankings['current_ratio']['rank']) else '-',
            int(financial_rankings['debt_ratio']['rank']) if pd.notna(financial_rankings['debt_ratio']['rank']) else '-'
        ],
        '百分位': [
            f"{financial_rankings['roe']['percentile']*100:.1f}%",
            f"{financial_rankings['operating_margin']['percentile']*100:.1f}%",
            f"{financial_rankings['roa']['percentile']*100:.1f}%",
            f"{financial_rankings['ebitda_margin']['percentile']*100:.1f}%",
            f"{financial_rankings['asset_turnover']['percentile']*100:.1f}%",
            f"{financial_rankings['current_ratio']['percentile']*100:.1f}%",
            f"{financial_rankings['debt_ratio']['percentile']*100:.1f}%"
        ]
    })
    
    st.dataframe(ranking_data, use_container_width=True, hide_index=True)

    with st.expander("🤖 AI解读：详细排名总览"):
        with st.spinner("AI 分析中..."):
            reclass_text = service.analyze_financial_rankings_overview(stock_code=company_code_norm)
        st.markdown(reclass_text)
else:
    st.warning("无法获取财务排名数据")


   


st.header("📈 财务指标雷达图")

tab1, tab2 = st.tabs(["综合维度雷达图", "指标级雷达图"])

with tab1:
    # 获取综合维度雷达图数据
    with st.spinner("加载综合维度雷达图数据..."):
        radar_data = get_company_radar_data(stock_code)

    if radar_data is not None:
        dimensions = radar_data.get('dimensions', [])
        scores = radar_data.get('scores', [])
        
        # 将分数转换为百分比（乘以100）
        scores_percent = [score * 100 for score in scores]
        
        company_name = radar_data.get('company_name', stock_code)
        year = radar_data.get('year', '')
        
        title = f"{company_name} 综合能力评估" + (f" ({year})" if year else "")
        
        radar_fig = create_radar_chart(dimensions, scores_percent, title)
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # 显示详细数据
        st.subheader("雷达图详细数据")
        radar_df = pd.DataFrame({
            '维度': dimensions,
            '得分': [f"{score*100:.1f}%" for score in scores]
        })
        st.dataframe(radar_df, use_container_width=True, hide_index=True)
    else:
        st.warning("无法获取综合维度雷达图数据")


with st.expander("🤖 AI解读：综合维度雷达图"):
    with st.spinner("AI 分析中..."):
        reclass_text = service.analyze_comprehensive_radar(stock_code=company_code_norm)
    st.markdown(reclass_text)

with tab2:
    # 获取指标级雷达图数据
    with st.spinner("加载指标级雷达图数据..."):
        indicator_data = get_company_indicator_radar_data(stock_code)

    if indicator_data is not None:
        # 获取所有维度的数据
        all_dimensions = []
        all_scores = []
        
        for key, value in indicator_data.items():
            if '_score' in key:
                dim_name = value.get('dimension_name', key.replace('_score', ''))
                indicators = value.get('indicators', [])
                scores = value.get('scores', [])
                
                # 将子指标分数汇总到维度
                if indicators and scores:
                    avg_score = sum(scores) / len(scores)
                    all_dimensions.append(dim_name)
                    all_scores.append(avg_score * 100)
        
        if all_dimensions:
            title = f"{basic_info.get('公司名称', stock_code)} 指标级能力评估"
            radar_fig = create_radar_chart(all_dimensions, all_scores, title)
            st.plotly_chart(radar_fig, use_container_width=True)
            
            # 显示详细数据
            st.subheader("雷达图详细数据")
            detail_data = []
            for key, value in indicator_data.items():
                if '_score' in key:
                    dim_name = value.get('dimension_name', key.replace('_score', ''))
                    indicators = value.get('indicators', [])
                    scores = value.get('scores', [])
                    
                    for ind, scr in zip(indicators, scores):
                        detail_data.append({
                            '维度': dim_name,
                            '指标': ind,
                            '得分': f"{scr*100:.1f}%"
                        })
            
            detail_df = pd.DataFrame(detail_data)
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
        else:
            st.warning("雷达图数据格式异常")
    else:
        st.warning("无法获取指标级雷达图数据")


with st.expander("🤖 AI解读：指标级雷达图"):
    with st.spinner("AI 分析中..."):
        reclass_text = service.analyze_indicator_radar(stock_code=company_code_norm)
    st.markdown(reclass_text)

st.divider()

st.markdown("---")
st.caption("数据更新时间: 2024年 | 数据来源: 公司公告、Wind数据库")
