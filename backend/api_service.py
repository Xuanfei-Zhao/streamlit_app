import os
import json
import pandas as pd
import numpy as np
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FinancialAIReport:
    """财务分析AI报告生成器 - 基于前端展示内容，为每个图表模块提供深度AI解读"""

    def __init__(self, api_key: str = "sk-72aef1acb8e342748533da787a6d6c59", model: str = "qwen-turbo"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

        self._stock_list = None
        self._industry_mapping = None
        self._financial_data = None
        self._dimension_json = None
        self._comprehensive_json = None
        self._trend_data = None
        self._benchmark_data = None
        self._cross_industry = None
        self._industry_keywords = None

        self._load_all_data()

    def _normalize_code(self, code: str) -> str:
        code_str = str(code).strip()
        if code_str.endswith('.0'):
            code_str = code_str[:-2]
        return code_str.zfill(6) if code_str.isdigit() else code_str

    def _load_all_data(self):
        print("[后端] 正在加载数据文件...")

        try:
            df = pd.read_excel('complete_company_industry_mapping_v4_stage16D_checked.xlsx', engine='openpyxl')
            if 'stock_code' in df.columns and 'symbol' not in df.columns:
                df.rename(columns={'stock_code': 'symbol'}, inplace=True)
            if 'stock_name' in df.columns and 'name' not in df.columns:
                df.rename(columns={'stock_name': 'name'}, inplace=True)
            df['symbol'] = df['symbol'].astype(str).str.zfill(6)
            self._stock_list = df
            print(f"  公司列表: {len(df)} 家公司")
        except Exception as e:
            print(f"  公司列表加载失败: {e}")

        try:
            df = pd.read_excel('complete_company_industry_mapping_v4_stage16D_checked.xlsx', engine='openpyxl')
            for col in ['stock_code', 'symbol']:
                if col in df.columns:
                    df['symbol'] = df[col].astype(str).str.zfill(6)
                    break
            self._industry_mapping = df
            print(f"  行业映射数据已加载")
        except Exception as e:
            print(f"  行业映射加载失败: {e}")

        try:
            df = pd.read_csv('company_statistics_with_median_percentile_rank.csv', encoding='utf-8', low_memory=False)
            code_col = None
            for col in df.columns:
                if 'code' in col.lower() or '代码' in col:
                    code_col = col
                    break
            if code_col:
                df[code_col] = df[code_col].astype(str).apply(self._normalize_code)
                df.rename(columns={code_col: 'stock_code'}, inplace=True)
            self._financial_data = df
            print(f"  财务指标数据: {len(df)} 条记录")
        except Exception as e:
            print(f"  财务指标加载失败: {e}")

        try:
            with open('公司各维度雷达图数据（可直接用于streamlit）.json', 'r', encoding='utf-8') as f:
                self._dimension_json = json.load(f)
            print(f"  维度雷达图: {len(self._dimension_json)} 家公司")
        except Exception as e:
            print(f"  维度雷达图加载失败: {e}")

        try:
            with open('公司综合维度雷达图数据（可直接用于streamlit）.json', 'r', encoding='utf-8') as f:
                self._comprehensive_json = json.load(f)
            print(f"  综合雷达图: {len(self._comprehensive_json)} 家公司")
        except Exception as e:
            print(f"  综合雷达图加载失败: {e}")

        try:
            df = pd.read_csv('公司五年数据趋势分析.csv', encoding='utf-8')
            code_col = None
            for col in df.columns:
                if 'code' in col.lower() or '代码' in col:
                    code_col = col
                    break
            if code_col:
                df[code_col] = df[code_col].astype(str).apply(self._normalize_code)
            self._trend_data = df
            print(f"  趋势数据: {len(df)} 条记录")
        except Exception as e:
            print(f"  趋势数据加载失败: {e}")

        try:
            df = pd.read_csv('财报分析-数据-分位数-行业中位数-行业排行（1为最优）.csv', encoding='utf-8', low_memory=False)
            code_col = None
            for col in df.columns:
                if col == 'stock_code_norm' or '公司代码' in col or 'code' in col.lower():
                    code_col = col
                    break
            if code_col:
                df[code_col] = df[code_col].astype(str).apply(self._normalize_code)
                df.rename(columns={code_col: 'company_code'}, inplace=True)
            self._benchmark_data = df
            print(f"  行业基准数据: {len(df)} 条记录")
        except Exception as e:
            print(f"  行业基准加载失败: {e}")

        try:
            df = pd.read_excel('stage17_final_cross_industry_mapping_table(2).xlsx', engine='openpyxl')
            if 'stock_code_norm' in df.columns:
                df['stock_code'] = df['stock_code_norm'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
            elif 'stock_code' in df.columns:
                df['stock_code'] = df['stock_code'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
            self._cross_industry = df
            print(f"  跨行业数据: {len(df)} 条记录")
        except Exception as e:
            print(f"  跨行业数据加载失败: {e}")

        try:
            df = pd.read_csv('step5_industry_top10_keywords_wide_all.csv', encoding='utf-8')
            if 'industry_label' in df.columns and 'industry' not in df.columns:
                df.rename(columns={'industry_label': 'industry'}, inplace=True)
            self._industry_keywords = df
            print(f"  行业关键词: {len(df)} 个行业")
        except Exception as e:
            print(f"  行业关键词加载失败: {e}")

        print("[后端] 数据加载完成\n")


    def get_company_name(self, stock_code: str) -> str:
        code = self._normalize_code(stock_code)
        if self._stock_list is not None:
            matched = self._stock_list[self._stock_list['symbol'] == code]
            if not matched.empty:
                return matched.iloc[0].get('name', '未知')
        return "未知"

    def get_industry_info(self, stock_code: str) -> Dict[str, Any]:
        code = self._normalize_code(stock_code)
        info = {
            '一级行业': '未知',
            '二级行业': '未知',
            '三级行业': '未知',
            '置信度': {},
            '是否跨行业': False,
            '副行业': []
        }

        if self._industry_mapping is not None:
            matched = self._industry_mapping[self._industry_mapping['symbol'] == code]
            if not matched.empty:
                row = matched.iloc[0]
                info['一级行业'] = row.get('final_level1_label', row.get('一级行业', '未知'))
                info['二级行业'] = row.get('final_level2_label', row.get('二级行业', '未知'))
                info['三级行业'] = row.get('final_level3_label', row.get('三级行业', '未知'))
                info['置信度'] = {
                    '一级': row.get('new_level1_confidence', None),
                    '二级': row.get('new_level2_confidence', None),
                    '三级': row.get('new_level3_confidence', None)
                }

        if self._cross_industry is not None:
            cross = self._cross_industry[self._cross_industry['stock_code'] == code]
            if not cross.empty:
                row = cross.iloc[0]
                diversified = row.get('is_diversified', False)
                if str(diversified).upper() in ['TRUE', '1', 'YES']:
                    info['是否跨行业'] = True
                    for level in ['1', '2', '3']:
                        labels = row.get(f'mapped_alt_level{level}_labels', '')
                        weights = row.get(f'mapped_alt_level{level}_weights', '')
                        if pd.notna(labels) and labels:
                            lbls = [l.strip() for l in str(labels).split(';') if l.strip()]
                            wts = []
                            if pd.notna(weights):
                                try:
                                    wts = [float(w.strip()) for w in str(weights).split(';') if w.strip()]
                                except:
                                    wts = [0.5] * len(lbls)
                            while len(wts) < len(lbls):
                                wts.append(0.5)
                            for l, w in zip(lbls, wts):
                                info['副行业'].append({'级别': level, '行业': l, '权重': w})

        return info

    # ==================== 数据获取函数（与前端展示对应）====================

    def get_financial_rankings(self, stock_code: str) -> Dict[str, Any]:
        """获取财务指标排名数据（对应2_公司概况.py的财务排名）"""
        code = self._normalize_code(stock_code)
        rankings = {}

        if self._benchmark_data is not None:
            company_data = self._benchmark_data[self._benchmark_data['company_code'] == code]
            if not company_data.empty:
                value_cols = [c for c in company_data.columns if c.endswith('_value')]
                for vcol in value_cols:
                    indicator = vcol[:-6]
                    median_col = indicator + '_median'
                    percentile_col = indicator + '_percentile'
                    rank_col = indicator + '_rank'

                    rankings[indicator] = {
                        '公司值': company_data[vcol].values[0] if vcol in company_data.columns else None,
                        '行业中位数': company_data[median_col].values[0] if median_col in company_data.columns else None,
                        '分位数': company_data[percentile_col].values[0] if percentile_col in company_data.columns else None,
                        '行业排名': company_data[rank_col].values[0] if rank_col in company_data.columns else None
                    }

        return rankings

    def get_radar_data(self, stock_code: str) -> Dict[str, Any]:
        """获取雷达图数据（综合维度 + 指标维度）"""
        code = self._normalize_code(stock_code)
        radar = {'综合维度': {}, '指标维度': {}}

        if self._comprehensive_json:
            for key, data in self._comprehensive_json.items():
                if code in key:
                    radar['综合维度'] = {
                        'dimensions': data.get('dimensions', []),
                        'scores': data.get('scores', [])
                    }
                    break

        if self._dimension_json:
            for key, data in self._dimension_json.items():
                if code in key:
                    indicators = {}
                    for dim_key, dim_val in data.items():
                        if isinstance(dim_val, dict) and 'indicators' in dim_val:
                            dim_name = dim_val.get('dimension_name', dim_key.replace('_score', ''))
                            indicators[dim_name] = {
                                'indicators': dim_val.get('indicators', []),
                                'scores': dim_val.get('scores', [])
                            }
                    radar['指标维度'] = indicators
                    break

        return radar

    def get_dimension_trend(self, stock_code: str) -> Dict[str, Any]:
        """获取各维度五年趋势数据（对应3_财务分析.py的趋势对比）"""
        code = self._normalize_code(stock_code)
        trend = {}

        if self._trend_data is not None:
            code_col = None
            for col in self._trend_data.columns:
                if 'code' in col.lower() or '代码' in col:
                    code_col = col
                    break
            if code_col:
                matched = self._trend_data[self._trend_data[code_col] == code]
                if not matched.empty:
                    row = matched.iloc[0]
                    dim_scores = {}
                    for col in self._trend_data.columns:
                        if col.endswith('_score_2024'):
                            dim_name = col.replace('_score_2024', '')
                            dim_scores[dim_name] = {
                                '2024': row.get(col, None),
                                '5年均值': row.get(col.replace('_score_2024', '_score_5y_mean'), None)
                            }
                    trend['维度趋势'] = dim_scores

        return trend

    def get_dimension_detail(self, stock_code: str) -> Dict[str, Any]:
        """获取各能力维度的详细数据（含指标级对比，对应3_财务分析.py的折叠卡片）"""
        code = self._normalize_code(stock_code)
        detail = {}

        # 从JSON获取维度指标和得分
        if self._dimension_json:
            for key, data in self._dimension_json.items():
                if code in key:
                    for dim_key, dim_val in data.items():
                        if isinstance(dim_val, dict) and 'indicators' in dim_val:
                            dim_name = dim_val.get('dimension_name', dim_key.replace('_score', ''))
                            detail[dim_name] = {
                                'indicators': dim_val.get('indicators', []),
                                'scores': dim_val.get('scores', [])
                            }
                    break

        # 合并行业基准数据
        if self._benchmark_data is not None:
            company_data = self._benchmark_data[self._benchmark_data['company_code'] == code]
            if not company_data.empty:
                for dim_name, dim_info in detail.items():
                    indicators = dim_info['indicators']
                    industry_medians = []
                    percentiles = []
                    ranks = []
                    for ind in indicators:
                        vcol = ind + '_value'
                        mcol = ind + '_median'
                        pcol = ind + '_percentile'
                        rcol = ind + '_rank'
                        industry_medians.append(company_data[mcol].values[0] if mcol in company_data.columns else None)
                        percentiles.append(company_data[pcol].values[0] if pcol in company_data.columns else None)
                        ranks.append(company_data[rcol].values[0] if rcol in company_data.columns else None)
                    detail[dim_name]['industry_medians'] = industry_medians
                    detail[dim_name]['percentiles'] = percentiles
                    detail[dim_name]['ranks'] = ranks

        return detail

    def get_industry_keywords(self, industry_name: str) -> List[Dict]:
        keywords = []
        if self._industry_keywords is not None:
            matched = self._industry_keywords[self._industry_keywords['industry'] == industry_name]
            if not matched.empty:
                row = matched.iloc[0]
                for i in range(1, 11):
                    kw = row.get(f'keyword_{i}', '')
                    score = row.get(f'keyword_{i}_score', None)
                    if pd.notna(kw) and kw:
                        keywords.append({'排名': i, '关键词': kw, '权重': score})
        return keywords

    def get_similar_companies(self, stock_code: str, top_n: int = 10) -> List[Dict]:
        code = self._normalize_code(stock_code)
        similar = []
        industry_info = self.get_industry_info(code)
        industry = industry_info.get('三级行业', '')

        if not industry or self._industry_mapping is None:
            return similar

        same_industry = self._industry_mapping[
            (self._industry_mapping['final_level3_label'] == industry) &
            (self._industry_mapping['symbol'] != code)
        ]

        for _, row in same_industry.head(top_n).iterrows():
            similar.append({
                '代码': row['symbol'],
                '名称': row.get('name', '未知'),
                '行业': industry
            })

        return similar

    # ==================== 模块化AI分析函数 ====================

    # ---------- 1. 行业分类页 ----------

    def analyze_company_overview(self, stock_code: str) -> str:
        """模块1：公司概况AI解读（对应1_行业分类.py顶部公司信息）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        industry_info = self.get_industry_info(code)

        data_lines = [
            "公司名称: " + company_name,
            "股票代码: " + code,
            "所属新一级行业: " + industry_info['一级行业'],
            "所属新二级行业: " + industry_info['二级行业'],
            "所属新三级行业: " + industry_info['三级行业'],
            "是否跨行业: " + ('是' if industry_info['是否跨行业'] else '否')
        ]
        data = "\n".join(data_lines)


        prompt_lines = [
            "你是一位资深财务分析师。请基于以下公司基本信息，生成一段专业的公司概况AI解读。",
            "要求：",
            "1. 介绍公司行业定位和主营业务特征",
            "2. 说明公司在新行业体系下的分类意义",
            "3. 如果存在跨行业特征，分析其业务多元化程度",
            "4. 语言专业、客观，像金融研究报告的口吻",
            "5. 不限制字数，根据信息量充分分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)


        return self.call_qwen(prompt, max_tokens=2000, temperature=0.3)

    def analyze_industry_reclassification(self, stock_code: str) -> str:
        """模块2：行业重分类AI解读（对应1_行业分类.py行业身份卡）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        industry_info = self.get_industry_info(code)

        conf = industry_info.get('置信度', {})
        conf_lines = []
        if conf.get('一级') is not None:
            conf_lines.append("一级置信度: " + str(round(conf['一级'], 4)))
        if conf.get('二级') is not None:
            conf_lines.append("二级置信度: " + str(round(conf['二级'], 4)))
        if conf.get('三级') is not None:
            conf_lines.append("三级置信度: " + str(round(conf['三级'], 4)))
        conf_text = "\n".join(conf_lines)

        data_lines = [
            "公司名称: " + company_name,
            "新一级行业: " + industry_info['一级行业'],
            "新二级行业: " + industry_info['二级行业'],
            "新三级行业: " + industry_info['三级行业'],
            conf_text,
            "是否跨行业: " + ('是' if industry_info['是否跨行业'] else '否')
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深行业研究专家。请基于以下行业重分类数据，生成一段专业的AI解读。",
            "要求：",
            "1. 说明公司在新行业体系下的精准定位",
            "2. 解释新分类相比传统分类的优势和意义",
            "3. 分析分类置信度的含义（高/中/低分别代表什么）",
            "4. 如果存在跨行业特征，分析其业务边界模糊的原因",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2000, temperature=0.3)

    def analyze_industry_keywords(self, stock_code: str) -> str:
        """模块3：行业关键词AI解读（对应1_行业分类.py行业TOP10关键词）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        industry_info = self.get_industry_info(code)
        keywords = self.get_industry_keywords(industry_info.get('三级行业', ''))

        keywords_lines = []
        if keywords:
            for kw in keywords[:10]:
                weight_str = str(round(kw['权重'], 3)) if kw['权重'] is not None else "N/A"
                keywords_lines.append("排名" + str(kw['排名']) + ": " + kw['关键词'] + " (权重" + weight_str + ")")
        else:
            keywords_lines.append("数据缺失")
        keywords_text = "\n".join(keywords_lines)

        data_lines = [
            "公司名称: " + company_name,
            "所属三级行业: " + industry_info['三级行业'],
            "行业TOP10关键词:",
            keywords_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深行业分析师。请基于以下行业关键词数据，生成一段专业的AI解读。",
            "要求：",
            "1. 深入分析关键词反映的行业核心特征和商业模式",
            "2. 说明这些关键词如何定义公司在行业中的角色和定位",
            "3. 从关键词权重分布看行业竞争焦点和发展趋势",
            "4. 结合公司实际情况，分析其在行业中的差异化特征",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2000, temperature=0.3)

    def analyze_similar_companies(self, stock_code: str) -> str:
        """模块4：同行业相似公司AI解读（对应1_行业分类.py相似公司TOP10）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        similar = self.get_similar_companies(code, top_n=10)

        similar_lines = []
        if similar:
            for i, comp in enumerate(similar[:10], 1):
                similar_lines.append("排名" + str(i) + ": " + comp['名称'] + " (" + comp['代码'] + ")")
        else:
            similar_lines.append("数据缺失")
        similar_text = "\n".join(similar_lines)

        data_lines = [
            "公司名称: " + company_name,
            "同行业相似公司TOP10:",
            similar_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深投资分析师。请基于以下同行业相似公司数据，生成一段专业的AI解读。",
            "要求：",
            "1. 分析目标公司与相似公司的竞争关系和竞争格局",
            "2. 说明这些相似公司在业务模式和财务特征上的共同点",
            "3. 评价目标公司在同行中的相对位置和竞争优势",
            "4. 从同行对比角度，指出目标公司的独特价值和潜在风险",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2000, temperature=0.3)

    # ---------- 2. 公司概况页 - 行业内关键指标排名（三方面+总览） ----------

    def analyze_roe_ranking(self, stock_code: str) -> str:
        """模块5a：ROE排名AI解读（对应2_公司概况.py Tab1）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        rankings = self.get_financial_rankings(code)

        roe_data = rankings.get('权益资本利润率ROE', {})
        if not roe_data or roe_data.get('公司值') is None:
            return "ROE数据缺失，无法生成分析。"

        median_str = str(roe_data['行业中位数']) if roe_data['行业中位数'] is not None else 'N/A'
        rank_str = str(int(roe_data['行业排名'])) if roe_data['行业排名'] is not None and not pd.isna(roe_data['行业排名']) else 'N/A'
        pct_str = str(round(roe_data['分位数']*100, 1)) + "%" if roe_data['分位数'] is not None else 'N/A'

        data_lines = [
            "公司名称: " + company_name,
            "权益资本利润率(ROE):",
            "  公司值: " + str(roe_data['公司值']),
            "  行业中位数: " + median_str,
            "  行业排名: " + rank_str + " (1为最优)",
            "  百分位: " + pct_str + " (越高越好)"
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下ROE排名数据，生成一段专业的AI解读。",
            "要求：",
            "1. 分析公司ROE水平的绝对值和相对行业位置",
            "2. 对比行业中位数，评价公司股东回报能力的强弱",
            "3. 结合行业排名和百分位，判断公司在同行中的竞争地位",
            "4. 从杜邦分析角度（利润率、资产周转率、杠杆），推测ROE的驱动因素",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2000, temperature=0.3)

    def analyze_operating_margin_ranking(self, stock_code: str) -> str:
        """模块5b：营业利润率排名AI解读（对应2_公司概况.py Tab2）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        rankings = self.get_financial_rankings(code)

        margin_data = rankings.get('营业利润率', {})
        if not margin_data or margin_data.get('公司值') is None:
            return "营业利润率数据缺失，无法生成分析。"

        median_str = str(margin_data['行业中位数']) if margin_data['行业中位数'] is not None else 'N/A'
        rank_str = str(int(margin_data['行业排名'])) if margin_data['行业排名'] is not None and not pd.isna(margin_data['行业排名']) else 'N/A'
        pct_str = str(round(margin_data['分位数']*100, 1)) + "%" if margin_data['分位数'] is not None else 'N/A'

        data_lines = [
            "公司名称: " + company_name,
            "营业利润率:",
            "  公司值: " + str(margin_data['公司值']),
            "  行业中位数: " + median_str,
            "  行业排名: " + rank_str + " (1为最优)",
            "  百分位: " + pct_str + " (越高越好)"
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下营业利润率排名数据，生成一段专业的AI解读。",
            "要求：",
            "1. 分析公司营业利润率水平的绝对值和相对行业位置",
            "2. 对比行业中位数，评价公司核心业务盈利能力的强弱",
            "3. 结合行业排名和百分位，判断公司在同行中的盈利地位",
            "4. 分析营业利润率反映的成本控制能力和定价权",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2000, temperature=0.3)

    def analyze_roa_ranking(self, stock_code: str) -> str:
        """模块5c：总资产利润率排名AI解读（对应2_公司概况.py Tab3）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        rankings = self.get_financial_rankings(code)

        roa_data = rankings.get('总资产利润率ROA', {})
        if not roa_data or roa_data.get('公司值') is None:
            return "ROA数据缺失，无法生成分析。"

        median_str = str(roa_data['行业中位数']) if roa_data['行业中位数'] is not None else 'N/A'
        rank_str = str(int(roa_data['行业排名'])) if roa_data['行业排名'] is not None and not pd.isna(roa_data['行业排名']) else 'N/A'
        pct_str = str(round(roa_data['分位数']*100, 1)) + "%" if roa_data['分位数'] is not None else 'N/A'

        data_lines = [
            "公司名称: " + company_name,
            "总资产利润率(ROA):",
            "  公司值: " + str(roa_data['公司值']),
            "  行业中位数: " + median_str,
            "  行业排名: " + rank_str + " (1为最优)",
            "  百分位: " + pct_str + " (越高越好)"
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下ROA排名数据，生成一段专业的AI解读。",
            "要求：",
            "1. 分析公司ROA水平的绝对值和相对行业位置",
            "2. 对比行业中位数，评价公司资产运用效率的强弱",
            "3. 结合行业排名和百分位，判断公司在同行中的资产回报地位",
            "4. 分析ROA与ROE的差异，推断公司的杠杆策略",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2000, temperature=0.3)

    def analyze_financial_rankings_overview(self, stock_code: str) -> str:
        """模块5d：财务指标排名总览AI解读（对应2_公司概况.py详细排名数据表）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        rankings = self.get_financial_rankings(code)

        rankings_lines = []
        key_metrics = ['权益资本利润率ROE', '营业利润率', '总资产利润率ROA',
                       '总资产周转率', '流动比率', '总资产负债率',
                       '总资产创现率', '销售创现率', 'EBITDA利润率']
        for metric in key_metrics:
            if metric in rankings:
                r = rankings[metric]
                val = r['公司值']
                median = r['行业中位数']
                pct = r['分位数']
                rank = r['行业排名']
                if val is not None and not pd.isna(val):
                    pct_str = str(round(pct * 100, 1)) + "%" if pct is not None and not pd.isna(pct) else "N/A"
                    rank_str = str(int(rank)) if rank is not None and not pd.isna(rank) else "N/A"
                    vs_median = "高于" if val > median else "低于"
                    line = metric + ": 公司值" + str(round(val, 4)) + " | 行业中位数" + str(round(median, 4) if median is not None else 'N/A') + " | " + vs_median + "中位数 | 分位数" + pct_str + " | 排名" + rank_str
                    rankings_lines.append(line)

        if not rankings_lines:
            rankings_lines.append("数据缺失")
        rankings_text = "\n".join(rankings_lines)

        data_lines = [
            "公司名称: " + company_name,
            "核心财务指标行业对比总览:",
            rankings_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下财务指标排名总览数据，生成一段全面的AI解读。",
            "要求：",
            "1. 总结公司财务表现的整体特征和核心竞争力",
            "2. 系统分析盈利能力（ROE、营业利润率、ROA）、运营效率（资产周转率）、偿债能力（资产负债率、流动比率）、现金流能力（创现率）等维度",
            "3. 指出最强和最弱的指标，并用数据支撑结论",
            "4. 综合评价公司在行业中的财务竞争地位",
            "5. 不限制字数，充分展开分析，像一份专业的财务点评",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=3000, temperature=0.3)

    # ---------- 3. 公司概况页 - 雷达图（综合维度 + 指标级） ----------

    def analyze_comprehensive_radar(self, stock_code: str) -> str:
        """模块6a：综合维度雷达图AI解读（对应2_公司概况.py Tab1）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        radar = self.get_radar_data(code)

        radar_lines = []
        if radar['综合维度'].get('dimensions'):
            for dim, score in zip(radar['综合维度']['dimensions'], radar['综合维度']['scores']):
                radar_lines.append(dim + ": " + str(round(score * 100, 1)) + "分")
        else:
            radar_lines.append("数据缺失")
        radar_text = "\n".join(radar_lines)

        data_lines = [
            "公司名称: " + company_name,
            "综合维度能力评估:",
            radar_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下综合维度雷达图数据，生成一段专业的AI解读。",
            "要求：",
            "1. 描述公司六维能力的整体画像和特征",
            "2. 指出最强和最弱的维度，分析原因",
            "3. 评价公司属于什么类型（高成长型/稳健型/成熟型/问题型等）",
            "4. 分析各维度之间的协同或矛盾关系",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2000, temperature=0.3)

    def analyze_indicator_radar(self, stock_code: str) -> str:
        """模块6b：指标级雷达图AI解读（对应2_公司概况.py Tab2）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        radar = self.get_radar_data(code)

        indicator_lines = []
        if radar['指标维度']:
            for dim_name, dim_data in radar['指标维度'].items():
                indicator_lines.append("")
                indicator_lines.append("【" + dim_name + "】")
                indicators = dim_data.get('indicators', [])
                scores = dim_data.get('scores', [])
                for ind, scr in zip(indicators, scores):
                    indicator_lines.append("  " + ind + ": " + str(round(scr * 100, 1)) + "分")
        else:
            indicator_lines.append("数据缺失")
        indicator_text = "\n".join(indicator_lines)

        data_lines = [
            "公司名称: " + company_name,
            "指标级能力评估（各维度下的具体指标得分）:",
            indicator_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下指标级雷达图数据，生成一段专业的AI解读。",
            "要求：",
            "1. 分析每个能力维度下的具体指标表现",
            "2. 指出各维度内部的强项和弱项指标",
            "3. 分析指标得分的内在逻辑和关联性",
            "4. 从指标层面评价公司的精细化运营能力",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2500, temperature=0.3)

    # ---------- 4. 财务分析页 ----------

    def analyze_comprehensive_financial_radar(self, stock_code: str) -> str:
        """模块7a：综合财务雷达图分析（对应3_财务分析.py顶部综合得分区域）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        radar = self.get_radar_data(code)
        trend = self.get_dimension_trend(code)

        radar_lines = []
        if radar['综合维度'].get('dimensions'):
            for dim, score in zip(radar['综合维度']['dimensions'], radar['综合维度']['scores']):
                radar_lines.append(dim + ": " + str(round(score * 100, 1)) + "分")
        else:
            radar_lines.append("数据缺失")
        radar_text = "\n".join(radar_lines)

        trend_lines = []
        if '维度趋势' in trend:
            for dim, vals in trend['维度趋势'].items():
                v2024 = vals.get('2024')
                v5y = vals.get('5年均值')
                if v2024 is not None and v5y is not None:
                    change = v2024 - v5y
                    trend_dir = "提升" if change > 0 else "下降" if change < 0 else "持平"
                    trend_lines.append(dim + ": 2024年" + str(round(v2024, 2)) + " vs 5年均值" + str(round(v5y, 2)) + " (" + trend_dir + " " + str(round(abs(change), 2)) + ")")
        trend_text = "\n".join(trend_lines)

        data_lines = [
            "公司名称: " + company_name,
            "综合财务雷达图数据:",
            radar_text,
            "",
            "维度得分趋势对比:",
            trend_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下综合财务雷达图和趋势数据，生成一段全面的AI解读。",
            "要求：",
            "1. 分析公司当前六维能力的整体画像",
            "2. 结合五年趋势，评价各维度的发展轨迹和改善/恶化情况",
            "3. 指出哪些维度在持续改善，哪些在退步，分析可能原因",
            "4. 综合评价公司的财务健康度和成长性",
            "5. 不限制字数，充分展开分析，像一份专业的财务诊断报告",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=3000, temperature=0.3)

    def analyze_dimension_trend(self, stock_code: str) -> str:
        """模块7b：维度得分趋势对比分析（对应3_财务分析.py柱状图区域）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        trend = self.get_dimension_trend(code)

        trend_lines = []
        if '维度趋势' in trend:
            for dim, vals in trend['维度趋势'].items():
                v2024 = vals.get('2024')
                v5y = vals.get('5年均值')
                if v2024 is not None and v5y is not None:
                    change = v2024 - v5y
                    trend_dir = "提升" if change > 0 else "下降" if change < 0 else "持平"
                    trend_lines.append(dim + ": 2024年" + str(round(v2024, 2)) + " vs 5年均值" + str(round(v5y, 2)) + " (" + trend_dir + " " + str(round(abs(change), 2)) + ")")
                elif v2024 is not None:
                    trend_lines.append(dim + ": 2024年" + str(round(v2024, 2)) + " (5年均值: 数据缺失)")
                elif v5y is not None:
                    trend_lines.append(dim + ": 5年均值" + str(round(v5y, 2)) + " (2024年: 数据缺失)")
        else:
            trend_lines.append("趋势数据缺失")
        trend_text = "\n".join(trend_lines)

        data_lines = [
            "公司名称: " + company_name,
            "各维度五年得分趋势对比:",
            trend_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下维度得分趋势数据，生成一段专业的AI解读。",
            "要求：",
            "1. 分析公司五年来的整体发展轨迹",
            "2. 指出改善最明显的维度和退步最明显的维度",
            "3. 分析趋势变化背后的经营原因（如战略调整、行业周期、竞争格局变化等）",
            "4. 评价公司的发展轨迹类型（上升型/下降型/波动型/稳定型）",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2500, temperature=0.3)

    def _format_dimension_detail(self, dim_data: Dict) -> str:
        """辅助函数：格式化维度详细数据"""
        indicators = dim_data.get('indicators', [])
        scores = dim_data.get('scores', [])
        medians = dim_data.get('industry_medians', [])
        percentiles = dim_data.get('percentiles', [])
        ranks = dim_data.get('ranks', [])

        detail_lines = []
        for i, ind in enumerate(indicators):
            score = scores[i] if i < len(scores) else 'N/A'
            median = medians[i] if i < len(medians) and medians[i] is not None else 'N/A'
            pct = percentiles[i] if i < len(percentiles) and percentiles[i] is not None else 'N/A'
            rank = ranks[i] if i < len(ranks) and ranks[i] is not None else 'N/A'

            score_str = str(round(score*100, 1)) + "%" if isinstance(score, (int, float)) else str(score)
            median_str = str(round(median*100, 1)) + "%" if isinstance(median, (int, float)) else str(median)
            pct_str = str(round(pct*100, 1)) + "%" if isinstance(pct, (int, float)) else str(pct)
            rank_str = str(int(rank)) if isinstance(rank, (int, float)) and not pd.isna(rank) else str(rank)

            line = "  " + ind + ": 得分" + score_str + " | 行业中位数" + median_str + " | 分位数" + pct_str + " | 排名" + rank_str
            detail_lines.append(line)
        return "\n".join(detail_lines)


    def analyze_profitability(self, stock_code: str) -> str:
        """模块7c：盈利能力指标分析（对应3_财务分析.py盈利能力折叠卡片）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        detail = self.get_dimension_detail(code)

        profitability_dim = None
        for dim_name in detail.keys():
            if '盈利' in dim_name or 'profit' in dim_name.lower():
                profitability_dim = dim_name
                break

        if not profitability_dim:
            return "盈利能力维度数据缺失，无法生成分析。"

        dim_data = detail[profitability_dim]
        detail_text = self._format_dimension_detail(dim_data)

        data_lines = [
            "公司名称: " + company_name,
            "盈利能力维度分析:",
            "维度名称: " + profitability_dim,
            "指标详情:",
            detail_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下盈利能力指标数据，生成一段专业的AI解读。",
            "要求：",
            "1. 分析公司盈利能力的整体水平和结构特征",
            "2. 对比行业均值，评价各盈利指标的相对强弱",
            "3. 从毛利率、净利率、ROE、ROA等多角度分析盈利质量",
            "4. 分析盈利能力的可持续性和增长潜力",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2500, temperature=0.3)

    def analyze_asset_efficiency(self, stock_code: str) -> str:
        """模块7d：资产使用效率指标分析（对应3_财务分析.py资产效率折叠卡片）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        detail = self.get_dimension_detail(code)

        efficiency_dim = None
        for dim_name in detail.keys():
            if '资产' in dim_name or '效率' in dim_name or '周转' in dim_name or 'efficiency' in dim_name.lower():
                efficiency_dim = dim_name
                break

        if not efficiency_dim:
            return "资产效率维度数据缺失，无法生成分析。"

        dim_data = detail[efficiency_dim]
        detail_text = self._format_dimension_detail(dim_data)

        data_lines = [
            "公司名称: " + company_name,
            "资产使用效率维度分析:",
            "维度名称: " + efficiency_dim,
            "指标详情:",
            detail_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下资产使用效率指标数据，生成一段专业的AI解读。",
            "要求：",
            "1. 分析公司资产运营效率的整体水平",
            "2. 对比行业均值，评价各效率指标的相对强弱",
            "3. 从总资产周转率、存货周转率、应收账款周转率等角度分析资产利用效率",
            "4. 分析资产效率对ROE的驱动作用",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2500, temperature=0.3)

    def analyze_liquidity(self, stock_code: str) -> str:
        """模块7e：流动性指标分析（对应3_财务分析.py流动性折叠卡片）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        detail = self.get_dimension_detail(code)

        liquidity_dim = None
        for dim_name in detail.keys():
            if '流动' in dim_name or 'liquidity' in dim_name.lower():
                liquidity_dim = dim_name
                break

        if not liquidity_dim:
            return "流动性维度数据缺失，无法生成分析。"

        dim_data = detail[liquidity_dim]
        detail_text = self._format_dimension_detail(dim_data)

        data_lines = [
            "公司名称: " + company_name,
            "流动性维度分析:",
            "维度名称: " + liquidity_dim,
            "指标详情:",
            detail_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下流动性指标数据，生成一段专业的AI解读。",
            "要求：",
            "1. 分析公司流动性的整体水平和安全边际",
            "2. 对比行业均值，评价各流动性指标的相对强弱",
            "3. 从流动比率、速动比率、现金比率等角度分析短期偿债能力",
            "4. 分析流动性风险及其对经营稳定性的影响",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2500, temperature=0.3)

    def analyze_cash_creation(self, stock_code: str) -> str:
        """模块7f：现金创造能力指标分析（对应3_财务分析.py现金创造折叠卡片）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        detail = self.get_dimension_detail(code)

        cash_dim = None
        for dim_name in detail.keys():
            if '现金' in dim_name or '创现' in dim_name or 'cash' in dim_name.lower():
                cash_dim = dim_name
                break

        if not cash_dim:
            return "现金创造维度数据缺失，无法生成分析。"

        dim_data = detail[cash_dim]
        detail_text = self._format_dimension_detail(dim_data)

        data_lines = [
            "公司名称: " + company_name,
            "现金创造能力维度分析:",
            "维度名称: " + cash_dim,
            "指标详情:",
            detail_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下现金创造能力指标数据，生成一段专业的AI解读。",
            "要求：",
            "1. 分析公司现金创造能力的整体水平",
            "2. 对比行业均值，评价各现金指标的相对强弱",
            "3. 从经营现金流、自由现金流、现金转换周期等角度分析现金生成效率",
            "4. 分析现金创造能力对分红、投资和偿债的支撑作用",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2500, temperature=0.3)

    def analyze_solvency(self, stock_code: str) -> str:
        """模块7g：偿债能力指标分析（对应3_财务分析.py偿债能力折叠卡片）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        detail = self.get_dimension_detail(code)

        solvency_dim = None
        for dim_name in detail.keys():
            if '偿债' in dim_name or '负债' in dim_name or 'solvency' in dim_name.lower() or 'debt' in dim_name.lower():
                solvency_dim = dim_name
                break

        if not solvency_dim:
            return "偿债能力维度数据缺失，无法生成分析。"

        dim_data = detail[solvency_dim]
        detail_text = self._format_dimension_detail(dim_data)

        data_lines = [
            "公司名称: " + company_name,
            "偿债能力维度分析:",
            "维度名称: " + solvency_dim,
            "指标详情:",
            detail_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下偿债能力指标数据，生成一段专业的AI解读。",
            "要求：",
            "1. 分析公司偿债能力的整体水平和财务安全性",
            "2. 对比行业均值，评价各偿债指标的相对强弱",
            "3. 从资产负债率、利息保障倍数、长期负债比率等角度分析偿债风险",
            "4. 分析资本结构是否合理，是否存在过度杠杆或杠杆不足",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2500, temperature=0.3)

    def analyze_shareholder_return(self, stock_code: str) -> str:
        """模块7h：股东收益指标分析（对应3_财务分析.py股东收益折叠卡片）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        detail = self.get_dimension_detail(code)

        shareholder_dim = None
        for dim_name in detail.keys():
            if '股东' in dim_name or '收益' in dim_name or 'return' in dim_name.lower() or 'shareholder' in dim_name.lower():
                shareholder_dim = dim_name
                break

        if not shareholder_dim:
            return "股东收益维度数据缺失，无法生成分析。"

        dim_data = detail[shareholder_dim]
        detail_text = self._format_dimension_detail(dim_data)

        data_lines = [
            "公司名称: " + company_name,
            "股东收益维度分析:",
            "维度名称: " + shareholder_dim,
            "指标详情:",
            detail_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深投资分析师。请基于以下股东收益指标数据，生成一段专业的AI解读。",
            "要求：",
            "1. 分析公司为股东创造价值的能力",
            "2. 对比行业均值，评价各股东回报指标的相对强弱",
            "3. 从ROE、股息率、每股收益增长率等角度分析股东回报质量",
            "4. 分析公司的分红政策和资本回报策略",
            "5. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2500, temperature=0.3)

    # ---------- 5. 管理建议部分 ----------

    def analyze_basic_conclusion(self, stock_code: str) -> str:
        """模块8a：基本结论（经营状况、财务健康度、行业地位、发展潜力）"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        industry_info = self.get_industry_info(code)
        rankings = self.get_financial_rankings(code)
        radar = self.get_radar_data(code)
        trend = self.get_dimension_trend(code)

        # 计算各维度的综合判断
        advantages = []
        weaknesses = []
        for ind in ['权益资本利润率ROE', '营业利润率', '总资产利润率ROA',
                    '总资产周转率', '流动比率', '总资产负债率']:
            if ind in rankings:
                r = rankings[ind]
                pct = r['分位数']
                if pct is not None and not pd.isna(pct):
                    if pct > 0.7:
                        advantages.append(ind + " (分位数" + str(round(pct * 100, 1)) + "%)")
                    elif pct < 0.3:
                        weaknesses.append(ind + " (分位数" + str(round(pct * 100, 1)) + "%)")

        radar_strong = []
        radar_weak = []
        if radar['综合维度'].get('dimensions'):
            for dim, score in zip(radar['综合维度']['dimensions'], radar['综合维度']['scores']):
                if score > 0.7:
                    radar_strong.append(dim + " (" + str(round(score * 100, 1)) + "分)")
                elif score < 0.4:
                    radar_weak.append(dim + " (" + str(round(score * 100, 1)) + "分)")

        trend_lines = []
        if '维度趋势' in trend:
            improving = []
            declining = []
            for dim, vals in trend['维度趋势'].items():
                v2024 = vals.get('2024')
                v5y = vals.get('5年均值')
                if v2024 is not None and v5y is not None:
                    change = v2024 - v5y
                    if change > 0.05:
                        improving.append(dim + " (+" + str(round(change, 2)) + ")")
                    elif change < -0.05:
                        declining.append(dim + " (" + str(round(change, 2)) + ")")
            if improving:
                trend_lines.append("改善维度: " + "; ".join(improving))
            if declining:
                trend_lines.append("退步维度: " + "; ".join(declining))
        trend_summary = "\n".join(trend_lines)

        data_lines = [
            "公司名称: " + company_name,
            "所属行业: " + industry_info['三级行业'],
            "",
            "核心优势指标:",
            "; ".join(advantages) if advantages else "数据不足",
            "",
            "主要短板指标:",
            "; ".join(weaknesses) if weaknesses else "数据不足",
            "",
            "雷达图强项:",
            "; ".join(radar_strong) if radar_strong else "数据不足",
            "",
            "雷达图弱项:",
            "; ".join(radar_weak) if radar_weak else "数据不足",
            "",
            "五年趋势:",
            trend_summary
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下公司数据，生成一段全面的基本结论AI解读。",
            "要求从以下四个维度分别给出判断：",
            "1. **经营状况**：分析公司当前经营的健康程度和稳定性",
            "2. **财务健康度**：综合评价公司的财务结构、盈利质量、现金流和偿债能力",
            "3. **行业地位**：评价公司在所属行业中的竞争位置和相对优势",
            "4. **发展潜力**：分析公司未来的成长空间和改善潜力",
            "每个维度都要基于提供的数据给出明确的判断（优秀/良好/一般/较差/需关注），并说明理由。",
            "不限制字数，充分展开分析，像一份专业的投资评级报告。",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=4000, temperature=0.3)

    def analyze_management_suggestions(self, stock_code: str) -> str:
        """模块8b：管理建议"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        industry_info = self.get_industry_info(code)
        rankings = self.get_financial_rankings(code)
        radar = self.get_radar_data(code)
        trend = self.get_dimension_trend(code)

        # 自动识别优势和短板
        advantages = []
        weaknesses = []
        for ind in ['权益资本利润率ROE', '营业利润率', '总资产利润率ROA',
                    '总资产周转率', '流动比率', '总资产负债率',
                    '总资产创现率', '销售创现率']:
            if ind in rankings:
                r = rankings[ind]
                pct = r['分位数']
                if pct is not None and not pd.isna(pct):
                    if pct > 0.7:
                        advantages.append(ind + " (分位数" + str(round(pct * 100, 1)) + "%)")
                    elif pct < 0.3:
                        weaknesses.append(ind + " (分位数" + str(round(pct * 100, 1)) + "%)")

        radar_strong = []
        radar_weak = []
        if radar['综合维度'].get('dimensions'):
            for dim, score in zip(radar['综合维度']['dimensions'], radar['综合维度']['scores']):
                if score > 0.7:
                    radar_strong.append(dim + " (" + str(round(score * 100, 1)) + "分)")
                elif score < 0.4:
                    radar_weak.append(dim + " (" + str(round(score * 100, 1)) + "分)")

        trend_issues = []
        if '维度趋势' in trend:
            for dim, vals in trend['维度趋势'].items():
                v2024 = vals.get('2024')
                v5y = vals.get('5年均值')
                if v2024 is not None and v5y is not None:
                    change = v2024 - v5y
                    if change < -0.1:
                        trend_issues.append(dim + "显著退步 (" + str(round(change, 2)) + ")")
        trend_issues_text = "\n".join(trend_issues) if trend_issues else "无显著恶化趋势"


        data_lines = [
            "公司名称: " + company_name,
            "所属行业: " + industry_info['三级行业'],
            "",
            "核心优势:",
            "; ".join(advantages) if advantages else "数据不足",
            "",
            "主要短板:",
            "; ".join(weaknesses) if weaknesses else "数据不足",
            "",
            "雷达图强项:",
            "; ".join(radar_strong) if radar_strong else "数据不足",
            "",
            "雷达图弱项:",
            "; ".join(radar_weak) if radar_weak else "数据不足",
            "",
            "趋势恶化:",
            trend_issues_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深企业管理顾问。请基于以下公司财务数据，生成一段全面的管理建议AI解读。",
            "要求：",
            "1. 基于识别的优势和短板，给出3-5条具体、可操作的管理建议",
            "2. 建议分为：战略层面（长期方向）、运营层面（日常改进）、财务层面（资本结构优化）",
            "3. 每条建议必须针对具体问题，不得空泛",
            "4. 考虑行业特征和公司发展阶段",
            "5. 不限制字数，充分展开分析，像一份专业的管理咨询报告",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=4000, temperature=0.3)

    def analyze_risk_warnings(self, stock_code: str) -> str:
        """模块8c：风险提示"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        rankings = self.get_financial_rankings(code)
        radar = self.get_radar_data(code)
        trend = self.get_dimension_trend(code)

        # 识别风险点
        risks = []
        risk_details = []

        for ind in ['总资产负债率', '流动比率']:
            if ind in rankings:
                r = rankings[ind]
                pct = r['分位数']
                if pct is not None and not pd.isna(pct):
                    if ind == '总资产负债率' and pct > 0.7:
                        risks.append("资产负债率偏高 (分位数" + str(round(pct * 100, 1)) + "%)")
                        rd = "资产负债率: 公司值" + str(round(r['公司值'], 4))
                        rd += " | 行业中位数" + str(round(r['行业中位数'], 4) if r['行业中位数'] is not None else 'N/A')
                        rd += " | 排名" + str(int(r['行业排名']) if r['行业排名'] is not None and not pd.isna(r['行业排名']) else 'N/A')
                        risk_details.append(rd)
                    elif ind == '流动比率' and pct < 0.3:
                        risks.append("流动性不足 (分位数" + str(round(pct * 100, 1)) + "%)")
                        rd = "流动比率: 公司值" + str(round(r['公司值'], 4))
                        rd += " | 行业中位数" + str(round(r['行业中位数'], 4) if r['行业中位数'] is not None else 'N/A')
                        rd += " | 排名" + str(int(r['行业排名']) if r['行业排名'] is not None and not pd.isna(r['行业排名']) else 'N/A')
                        risk_details.append(rd)

        # 雷达图弱项风险
        radar_risks = []
        if radar['综合维度'].get('dimensions'):
            for dim, score in zip(radar['综合维度']['dimensions'], radar['综合维度']['scores']):
                if score < 0.3:
                    radar_risks.append(dim + "能力严重不足 (" + str(round(score * 100, 1)) + "分)")

        # 趋势风险
        trend_risks = []
        if '维度趋势' in trend:
            for dim, vals in trend['维度趋势'].items():
                v2024 = vals.get('2024')
                v5y = vals.get('5年均值')
                if v2024 is not None and v5y is not None:
                    change = v2024 - v5y
                    if change < -0.15:
                        trend_risks.append(dim + "能力持续恶化 (5年下降" + str(round(abs(change), 2)) + ")")

        roe_pct = str(round(rankings.get('权益资本利润率ROE', {}).get('分位数', 0) * 100, 1)) if '权益资本利润率ROE' in rankings else 'N/A'
        op_pct = str(round(rankings.get('营业利润率', {}).get('分位数', 0) * 100, 1)) if '营业利润率' in rankings else 'N/A'
        debt_pct = str(round(rankings.get('总资产负债率', {}).get('分位数', 0) * 100, 1)) if '总资产负债率' in rankings else 'N/A'

        data_lines = [
            "公司名称: " + company_name,
            "",
            "识别的财务风险点:",
            "; ".join(risks) if risks else "基于现有数据未发现显著财务风险",
            "",
            "风险指标详情:" + ("\n".join(risk_details) if risk_details else "无"),

            "",
            "雷达图能力弱项:",
            "; ".join(radar_risks) if radar_risks else "无",
            "",
            "趋势恶化风险:",
            "; ".join(trend_risks) if trend_risks else "无",
            "",
            "关键财务指标:",
            "- ROE分位数: " + roe_pct + "%",
            "- 营业利润率分位数: " + op_pct + "%",
            "- 资产负债率分位数: " + debt_pct + "%"
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深风险管理专家。请基于以下风险数据，生成一段全面的风险提示AI解读。",
            "要求：",
            "1. 分类说明风险类型：经营风险、财务风险、行业位置风险、趋势风险",
            "2. 每个风险必须有具体数据支撑，不得泛泛而谈",
            "3. 给出每条风险的简要应对建议",
            "4. 评估整体风险等级（低/中/高/极高）",
            "5. 不限制字数，充分展开分析，像一份专业的风险评级报告",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=3000, temperature=0.3)

    # ==================== API调用 ====================

    def call_qwen(self, prompt: str, max_tokens: int = 4000, temperature: float = 0.3) -> str:
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一位资深的财务分析专家，拥有20年上市公司研究经验。擅长从财务数据中发现企业经营问题，提出切实可行的管理建议。分析风格专业、客观、数据驱动，绝不编造数据。输出专业的AI解读文字，不限制字数，根据信息量充分展开分析。"
                    },
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {
                "result_format": "message",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.8
            }
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

            if "output" in result and "choices" in result["output"]:
                content = result["output"]["choices"][0]["message"]["content"]
                return content
            else:
                return "API返回异常: " + json.dumps(result, ensure_ascii=False, indent=2)

        except requests.exceptions.RequestException as e:
            return "API调用失败: " + str(e) + "\n请检查API Key和网络连接。"

        except Exception as e:
            return "处理响应时出错: " + str(e)

    def get_company_data_summary(self, stock_code: str) -> Dict[str, Any]:
        code = self._normalize_code(stock_code)
        return {
            '基本信息': {
                '股票代码': code,
                '公司名称': self.get_company_name(code),
                '行业信息': self.get_industry_info(code)
            },
            '财务排名': self.get_financial_rankings(code),
            '雷达图数据': self.get_radar_data(code),
            '维度趋势': self.get_dimension_trend(code),
            '维度详情': self.get_dimension_detail(code)
        }


# ==================== 便捷函数 ====================

def analyze_module(stock_code: str, module: str, api_key: str, model: str = "qwen-turbo") -> str:
    """
    快速分析指定模块

    Args:
        stock_code: 股票代码
        module: 模块名称
            # 行业分类页
            - 'overview' -> 公司概况
            - 'industry' -> 行业重分类
            - 'keywords' -> 行业关键词
            - 'similar' -> 相似公司
            # 公司概况页 - 财务排名
            - 'roe_ranking' -> ROE排名
            - 'margin_ranking' -> 营业利润率排名
            - 'roa_ranking' -> ROA排名
            - 'rankings_overview' -> 财务排名总览
            # 公司概况页 - 雷达图
            - 'comprehensive_radar' -> 综合维度雷达图
            - 'indicator_radar' -> 指标级雷达图
            # 财务分析页
            - 'financial_radar' -> 综合财务雷达图
            - 'dimension_trend' -> 维度得分趋势
            - 'profitability' -> 盈利能力
            - 'asset_efficiency' -> 资产使用效率
            - 'liquidity' -> 流动性指标
            - 'cash_creation' -> 现金创造能力
            - 'solvency' -> 偿债能力
            - 'shareholder_return' -> 股东收益
            # 管理建议页
            - 'conclusion' -> 基本结论
            - 'suggestions' -> 管理建议
            - 'risk' -> 风险提示
        api_key: API Key
        model: 模型名称

    Returns:
        AI解读文字
    """
    service = FinancialAIReport(api_key=api_key, model=model)

    module_map = {
        # 行业分类页
        'overview': service.analyze_company_overview,
        'industry': service.analyze_industry_reclassification,
        'keywords': service.analyze_industry_keywords,
        'similar': service.analyze_similar_companies,
        # 公司概况页 - 财务排名
        'roe_ranking': service.analyze_roe_ranking,
        'margin_ranking': service.analyze_operating_margin_ranking,
        'roa_ranking': service.analyze_roa_ranking,
        'rankings_overview': service.analyze_financial_rankings_overview,
        # 公司概况页 - 雷达图
        'comprehensive_radar': service.analyze_comprehensive_radar,
        'indicator_radar': service.analyze_indicator_radar,
        # 财务分析页
        'financial_radar': service.analyze_comprehensive_financial_radar,
        'dimension_trend': service.analyze_dimension_trend,
        'profitability': service.analyze_profitability,
        'asset_efficiency': service.analyze_asset_efficiency,
        'liquidity': service.analyze_liquidity,
        'cash_creation': service.analyze_cash_creation,
        'solvency': service.analyze_solvency,
        'shareholder_return': service.analyze_shareholder_return,
        # 管理建议页
        'conclusion': service.analyze_basic_conclusion,
        'suggestions': service.analyze_management_suggestions,
        'risk': service.analyze_risk_warnings,
    }

    if module in module_map:
        return module_map[module](stock_code)
    else:
        return "【模块名称错误，可选模块列表见文档】"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_code = sys.argv[1]
    else:
        test_code = "000001"

    api_key = os.environ.get("DASHSCOPE_API_KEY", "your-api-key-here")

    if api_key == "your-api-key-here":
        print("请设置环境变量 DASHSCOPE_API_KEY 或修改代码中的 api_key")
        print("示例: export DASHSCOPE_API_KEY=sk-xxxxx")
    else:
        service = FinancialAIReport(api_key=api_key, model="qwen-turbo")

        # 测试所有模块
        modules = [
            ('overview', '公司概况'),
            ('industry', '行业重分类'),
            ('keywords', '行业关键词'),
            ('similar', '相似公司'),
            ('roe_ranking', 'ROE排名'),
            ('margin_ranking', '营业利润率排名'),
            ('roa_ranking', 'ROA排名'),
            ('rankings_overview', '财务排名总览'),
            ('comprehensive_radar', '综合维度雷达图'),
            ('indicator_radar', '指标级雷达图'),
            ('financial_radar', '综合财务雷达图'),
            ('dimension_trend', '维度得分趋势'),
            ('profitability', '盈利能力'),
            ('asset_efficiency', '资产使用效率'),
            ('liquidity', '流动性指标'),
            ('cash_creation', '现金创造能力'),
            ('solvency', '偿债能力'),
            ('shareholder_return', '股东收益'),
            ('conclusion', '基本结论'),
            ('suggestions', '管理建议'),
            ('risk', '风险提示'),
        ]

        for module_key, module_name in modules:
            print("\n" + "="*60)

            print("测试模块: " + module_name + " (" + module_key + ")")
            print("="*60)
            result = analyze_module(test_code, module_key, api_key)
            print(result[:500] + "..." if len(result) > 500 else result)