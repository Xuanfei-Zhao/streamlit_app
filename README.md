**上市公司行业重分类智能财务机器人**

本产品为基于主营产品文本和财务指标的上市公司行业重分类平台。主要内容包括：**行业分类** | **跨行业识别**|**公司概况** | **财务分析** |**AI 管理建议**等

📌 功能亮点

- **行业重分类**  
  利用年报文本构建企业相似度网络，通过 Leiden 算法生成 **29 个一级、114 个二级、267 个三级行业**，并提供置信度评估。此外还包含行业/公司关键词、相似公司排行、跨行业识别等丰富内容

- **公司概况**  
  介绍公司的基本信息和业务介绍，提供财务基本面概览。

- **财务分析**  
  六维雷达图、五年趋势对比、核心指标排名（分位数、行业中位数）。

- **管理建议**  
  调用大模型（通义千问）自动生成基本结论、管理建议和风险提示。

📂 项目结构
.
├── pages/                    # 多页面应用
│   ├── 0_产品介绍.py
│   ├── 1_行业分类.py
│   ├── 2_公司概况.py
│   ├── 3_财务分析.py
│   └── 4_管理建议.py
├── utils/                    # 公共工具函数
│   ├── charts.py             # 雷达图、折线图
│   └── data_loader.py        # 数据加载
├── backend/                  # AI 后端服务
│   └── api_service.py        # 大模型调用、数据分析
├── streamlit_app.py          # 主入口
├── requirements.txt          # Python 依赖
└── .streamlit/
    └── secrets.toml          # 密钥（不提交Git）

📂 数据文件

├── complete_company_industry_mapping_v5_qwen_level1_final.xlsx — 公司-行业映射表（含三级标签、置信度）
├── company_statistics_with_raw_median_percentile_rank-1.csv — 财务指标及行业基准数据
├── financial_with_classification.csv — 年度财务数据（含分类）
├── step5_industry_top10_keywords_wide_all.csv — 行业 TOP10 关键词
├── company_5year_top20_tfidf_keywords_v2.csv — 公司TOP20关键词
├── similar_company_top10_5year_tfidf_svd_v2.csv — 同行业相似公司TOP10
├── stage17_final_cross_industry_mapping_t.xlsx — 跨行业备选数据
├── standard_industry_dictionary_v6_qwen_level1_final — 标准行业字典（含一级、二级、三级行业标签）
├── 公司五年数据趋势分析.csv — 趋势分析图
├── 公司各纬度得分.csv — 各维度五年得分
├── 公司各维度雷达图数据（可直接用于streamlit）.json — 维度雷达图 (也即indicator_radars_data.json)
├── 公司综合维度雷达图数据（可直接用于streamlit）.json — 综合雷达图(也即radars_data.json)
├── 趋势分析总结.csv — 各维度标签分布

🛠️ 技术栈

  前端：Streamlit
  数据处理：Pandas, NumPy
  可视化：Plotly
  机器学习：scikit-learn（余弦相似度、标准化）
  AI 模型：通义千问 API (DashScope)

📊 数据流程图

  年报文本 → TF‑IDF → 相似度网络 → Leiden 社群发现 → 三级行业标签
  财务指标 → 标准化 → 余弦相似度 → 同行业相似公司
  多维指标 → 分位数计算 → 行业排名与雷达图