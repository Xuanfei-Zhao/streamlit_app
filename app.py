import streamlit as st
import pandas as pd
from pathlib import Path
from backend.api_service import FinancialAIReport


def load_stock_list():
    df = pd.read_excel('complete_company_industry_mapping_v4_stage16D_checked.xlsx', engine='openpyxl')
    # 确保列名包含 'symbol' 和 'name'，必要时重命名
    return df

# 页面配置
st.set_page_config(
    page_title='上市公司分析平台',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# 初始化会话状态
if 'selected_company' not in st.session_state:
    st.session_state.selected_company = None

# 接入AI
# 初始化 AI 服务（只执行一次）
if 'ai_service' not in st.session_state:
    st.session_state.ai_service = FinancialAIReport(
        api_key=st.secrets["DASHSCOPE_API_KEY"]
    )
    
service = st.session_state.ai_service

# -----------------------------------------------------------------------------
# 侧边栏导航

with st.sidebar:
    st.header("🏢 公司选择")
    
    from utils.data_loader import load_stock_list
    
    stocks_df = load_stock_list()  # 使用已标准化列名的数据加载函数
    
    # 验证必要列是否存在
    if 'symbol' not in stocks_df.columns:
        st.error(f"找不到股票代码列，请检查数据文件列名: {list(stocks_df.columns)}")
        st.stop()
    
    if 'name' not in stocks_df.columns:
        st.error(f"找不到公司名称列，请检查数据文件列名: {list(stocks_df.columns)}")
        st.stop()
    
    # 创建显示名称列表（代码 + 名称）
    stock_options = [f"{row['symbol']} - {row['name']}" for _, row in stocks_df.iterrows()] 
    stock_codes = stocks_df['symbol'].tolist()
    
    # 公司选择器
    selected_option = st.selectbox(
        "选择公司",
        options=stock_options,
        index=None,
        placeholder="请选择一家公司..."
    )
    
    if selected_option:
        # 提取股票代码
        selected_code = selected_option.split(' - ')[0]
        st.session_state.selected_company = selected_code
        
        st.success(f"✅ 已选择: {selected_option}")
        
        # 显示快速信息
        company_info = stocks_df[stocks_df['symbol'].astype(str) == selected_code].iloc[0] 
        st.divider() 
        st.markdown(f""" 
        **公司信息** 
        - 代码: `{selected_code}` 
        - 名称: {company_info['name']} 
        """)
    
    st.divider()
    
    # 页面导航说明
    st.header("📑 页面导航")
    st.markdown("""
    - **行业分类**: 查找公司所属行业、同行业相似公司及跨行业特征识别
    - **公司概况**: 获取公司基本信息、主营业务、行业排名
    - **财务分析**: 查看公司盈利能力、资产使用效率等各维度的财务指标得分情况
    - **管理建议**: 获取AI生成的管理建议和风险提示
    """)
    
    st.divider()
    
    # 帮助信息
    with st.expander("❓ 使用说明"):
        st.markdown("""
        1. 在上方选择要分析的公司
        2. 通过左侧菜单切换不同页面
        3. 在"管理建议"页面可点击生成AI建议
        """)

# -----------------------------------------------------------------------------
# 主页内容
# 欢迎页面（未选择公司时显示）
if not st.session_state.selected_company:
    st.markdown("# 📊 欢迎使用上市公司分析平台")
    st.markdown("本平台提供上市公司的深度分析，包括：")
    
    # 自定义CSS：为特定列添加左内边距
    st.markdown("""
    <style>
    .indent-col {
        padding-left: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 第一行：行业分类 + 公司概况
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⭐ 行业分类")
        st.markdown("""
        - 公司所属行业
        - 行业TOP10关键词
        - 同行业相似公司排行
        - 跨行业识别
        """)
    with col2:
        st.markdown("### 🏢 公司概况")
        st.markdown("""
        - 基本信息展示
        - 主营业务分析
        - 关键指标排名（ROE、毛利率、营收增速）
        - 多维度可视化图表
        """)
    
    # 第二行：财务分析 + 管理建议（给财务分析列添加缩进类）
    col3, col4 = st.columns(2)
    with col3:
        # 给这一列的内容加一个带缩进的div
        st.markdown('<div class="indent-col">', unsafe_allow_html=True)
        st.markdown("### 📊 财务分析")
        st.markdown("""
        - 公司综合财务得分
        - 维度得分趋势对比
        - 各维度指标分析
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown("### 💡 管理建议")
        st.markdown("""
        - AI生成的基本结论
        - 针对性管理建议
        - 全面风险提示
        - 可重新生成建议
        """)
    
    st.markdown("## 🚀 开始使用")
    st.markdown("""
    1. 在**侧边栏**选择要分析的公司
    2. 使用左侧菜单切换到不同页面
    3. 查看详细分析报告和建议
    
    ---
    
    *数据更新时间: 2020-2026年*
    """)
    


else:
    # 已选择公司，显示快速概览 
    stock_code = st.session_state.selected_company 
    
    from utils.data_loader import load_stock_list
    
    stocks_df = load_stock_list()  # 使用已标准化列名的数据加载函数
    
    company_row = stocks_df[stocks_df['symbol'].astype(str) == stock_code].iloc[0]
    company_name = company_row['name']
    
    st.markdown(f""" 
    # 👋 欢迎分析 {company_name} 
    
    **当前选择**: `{stock_code} - {company_name}` 
    
    请使用左侧菜单切换到具体页面查看详细分析： 

    - 🔍 **行业分类**：行业分类、行业关键词、同行业相似公司、跨行业识别
    - 📋 **公司概况**: 基本信息、业务分析、行业排名 
    - 📊 **财务分析**: 综合财务得分、维度得分趋势对比、各维度指标分析
    - 💡 **管理建议**: AI智能分析和建议 
    
    --- 
    """)
    
    # 显示快捷操作卡片
    col1, col2, col3,col4 = st.columns(4)

    with col1:
        st.markdown(""" 
        ### 🔍 行业分类
        
        查看公司所属行业： 
        - 行业关键词 
        - 同行业相似公司排行 
        - 跨行业识别  
        """)

        if st.button("前往行业分类 →", key="go_to_industry", type="primary"):
            st.switch_page("pages/1_行业分类.py")
    
    with col2:
        st.markdown(""" 
        ### 📋 公司概况 
        
        查看公司的详细信息： 
        - 基本工商信息 
        - 主营业务分析 
        - 财务指标排名 

        """)
        
        if st.button("前往公司概况 →", key="go_to_overview", type="secondary"):
            st.switch_page("pages/2_公司概况.py")
    
    with col3:
        st.markdown(""" 
        ### 📊 财务分析 

        查看公司的财务数据：
        - 公司综合财务得分 
        - 维度得分趋势对比 
        - 各维度指标分析 

        """)
        
        if st.button("前往财务分析 →", key="go_to_financial", type="secondary"):
            st.switch_page("pages/3_财务分析.py")

    with col4:
        st.markdown(""" 
        ### 💡 管理建议 
        
        获取智能化的管理建议： 
        - AI基本结论 
        - 战略建议 
        - 风险提示 
        """)
        
        if st.button("前往管理建议 →", key="go_to_advice", type="secondary"):
            st.switch_page("pages/4_管理建议.py")

# ----------------------------------------------------------------------------- 
# 页脚 
st.markdown("---") 
st.caption("© 2020-2024 上市公司分析平台 | 数据来源: complete_company_industry_mapping_v4_stage16D_checked.xlsx")
