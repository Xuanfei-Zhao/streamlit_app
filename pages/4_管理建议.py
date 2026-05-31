import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.data_loader import get_company_basic_info

st.set_page_config(page_title="管理建议", page_icon="💡", layout="wide")

if 'selected_company' not in st.session_state:
    st.warning("⚠️ 请先在侧边栏选择一家公司")
    st.stop()

# 确保 AI 服务已初始化
if 'ai_service' not in st.session_state:
    st.error("AI 服务未初始化，请返回主页重新加载")
    st.stop()

service = st.session_state.ai_service
stock_code = st.session_state.selected_company
company_code_norm = str(stock_code).strip().zfill(6)

st.title(f"💡 {stock_code} - 管理建议")

basic_info = get_company_basic_info(stock_code)
company_name = basic_info['公司名称'] if basic_info else stock_code

# 缓存 AI 生成的结果
if 'llm_suggestions' not in st.session_state:
    st.session_state.llm_suggestions = None
if 'last_generated' not in st.session_state:
    st.session_state.last_generated = None

def generate_real_suggestions(stock_code):
    """
    调用真实的 AI 接口生成三个部分的内容
    """
    with st.spinner("正在调用 AI 生成基本结论..."):
        conclusion = service.analyze_basic_conclusion(stock_code)
    with st.spinner("正在调用 AI 生成管理建议..."):
        suggestions = service.analyze_management_suggestions(stock_code)
    with st.spinner("正在调用 AI 生成风险提示..."):
        risks = service.analyze_risk_warnings(stock_code)
    
    return {
        'basic_conclusion': conclusion,
        'management_suggestions': suggestions,
        'risk_warnings': risks
    }

with st.sidebar:
    st.header("⚙️ 建议生成设置")
    
    if st.button("🔄 生成建议", type="primary", use_container_width=True):
        # 调用真实 AI
        st.session_state.llm_suggestions = generate_real_suggestions(stock_code)
        st.session_state.last_generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success("✅ 建议生成成功！")
        st.rerun()  # 刷新页面显示新内容
    
    if st.session_state.last_generated:
        st.caption(f"上次生成时间: {st.session_state.last_generated}")
    
    st.divider()
    
    with st.expander("📋 查看生成依据", expanded=False):
        st.markdown("""
        ### 输入给LLM的关键数据
        
        AI 基于以下数据生成分析：
        - 公司财务指标行业排名（ROE、营业利润率等）
        - 各维度雷达图得分（盈利能力、资产效率、流动性等）
        - 五年维度得分趋势
        - 行业对比数据
        - 跨行业特征（如有）
        """)

# 如果还没有生成过，显示提示信息
if not st.session_state.llm_suggestions:
    st.info("👈 请点击侧边栏的「重新生成建议」按钮，获取AI生成的管理建议")
    
    st.markdown("""
    ---
    ### 💡 功能说明
    
    本页面将基于公司的财务数据、行业信息和市场环境，利用大语言模型生成：
    
    1. **基本结论**: 对公司整体经营状况的评估
    2. **管理建议**: 针对性的战略、运营、人才等方面建议
    3. **风险提示**: 识别潜在风险并提供应对建议
    
    点击侧边栏按钮即可开始生成。
    """)
    st.stop()

# 显示结果
suggestions = st.session_state.llm_suggestions

tab1, tab2, tab3 = st.tabs(["📊 基本结论", "💡 管理建议", "⚠️ 风险提示"])

with tab1:
    st.markdown(suggestions['basic_conclusion'])
    
    # 可选：添加一个简单的反馈按钮
    st.caption("以上分析由 AI 自动生成，仅供参考。")

with tab2:
    st.markdown(suggestions['management_suggestions'])
    
    # 可选：展示建议优先级（如果 AI 返回的内容中不包含，可以不加额外表格，避免虚假数据）
    # 如果需要保留原样的优先级表格，可以注释掉或根据 AI 输出解析。建议保持简洁。

with tab3:
    st.markdown(suggestions['risk_warnings'])

st.markdown("---")
st.caption("⚠️ 免责声明: 以上建议仅供参考，不构成投资决策依据。请结合实际情况谨慎判断。")
