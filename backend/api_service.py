import os
import json
import pandas as pd
import numpy as np
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
import warnings
import re
from difflib import get_close_matches
warnings.filterwarnings('ignore')


class FinancialAIReport:
    """财务分析AI报告生成器 - 基于前端展示内容，为每个图表模块提供深度AI解读"""

    def __init__(self, api_key: str = "sk-0d7e63627bd044e59e984e7062519a0c", model: str = "qwen-turbo"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

        self._stock_list = None
        self._industry_mapping = None
        self._financial_data = None
        self._dimension_json = None
        self._comprehensive_json = None
        self._dimension_scores = None          
        self._yearly_financial = None          
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

        # ===== 1. 公司列表 & 行业映射 =====
        try:
            df = pd.read_excel('complete_company_industry_mapping_v5_qwen_level1_final.xlsx', engine="openpyxl")        
            code_col = None
            candidates = ['stock_code', 'stock_code_norm', 'symbol', 'code', '证券代码']
            for col in candidates:
                if col in df.columns:
                    code_col = col
                    break
            if code_col is None:
                for col in df.columns:
                    if 'code' in col.lower():
                        code_col = col
                        break
            if code_col is None:
                raise KeyError(f"未找到股票代码列，可用列: {df.columns.tolist()}")
        
            df['symbol'] = df[code_col].apply(self._normalize_code)
        
            name_col = None
            name_candidates = ['company_name', 'stock_name', 'name', '公司名称']
            for col in name_candidates:
                if col in df.columns:
                    name_col = col
                    break
            if name_col is None:
                for col in df.columns:
                    if 'name' in col.lower() or '公司' in col:
                        name_col = col
                        break
            if name_col:
                df['name'] = df[name_col]
            else:
                df['name'] = '未知'
        
            self._stock_list = df
            self._industry_mapping = df
            print(f"  公司列表: {len(df)} 家公司")
        except Exception as e:
            print(f"  数据加载失败: {e}")
            self._stock_list = pd.DataFrame()
            self._industry_mapping = pd.DataFrame()

        # ===== 2. 财务指标数据（保留原始多行，不去重）=====
        try:
            df = pd.read_csv('company_statistics_with_raw_median_percentile_rank-1.csv', encoding='utf-8', low_memory=False)
            if 'stock_code_norm' in df.columns:
                df['stock_code_norm'] = df['stock_code_norm'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
            # 转换数值列
            numeric_cols = [col for col in df.columns if any(x in col for x in ['_raw_value', '_raw_median', '_percentile', '_rank'])]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            self._financial_data = df
            self._benchmark_data = df  # 复用同一份数据
            print(f"  财务指标数据: {len(df)} 条记录")
        except Exception as e:
            print(f"  财务指标加载失败: {e}")
            self._financial_data = pd.DataFrame()
            self._benchmark_data = pd.DataFrame()

        # ===== 3. 维度雷达图 JSON =====
        try:
            with open('公司各维度雷达图数据（可直接用于streamlit）.json', 'r', encoding='utf-8') as f:
                self._dimension_json = json.load(f)
            print(f"  维度雷达图: {len(self._dimension_json)} 家公司")
        except Exception as e:
            print(f"  维度雷达图加载失败: {e}")

        # ===== 4. 综合雷达图 JSON =====
        try:
            with open('公司综合维度雷达图数据（可直接用于streamlit）.json', 'r', encoding='utf-8') as f:
                self._comprehensive_json = json.load(f)
            print(f"  综合雷达图: {len(self._comprehensive_json)} 家公司")
        except Exception as e:
            print(f"  综合雷达图加载失败: {e}")

        # ===== 5. 各维度得分 CSV（多行格式：每年一行）=====
        try:
            df = pd.read_csv('公司各纬度得分.csv', encoding='utf-8')
            if 'stock_code_norm' in df.columns:
                df['stock_code_norm'] = df['stock_code_norm'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
            self._dimension_scores = df
            print(f"  各维度得分数据: {len(df)} 条记录")
        except Exception as e:
            print(f"  各维度得分加载失败: {e}")

        # ===== 6. 年度财务数据 =====
        try:
            df = pd.read_csv('financial_with_classification.csv', encoding='utf-8')
            code_col = None
            for col in df.columns:
                if 'stock_code_norm' in col or 'code' in col.lower():
                    code_col = col
                    break
            if code_col:
                df[code_col] = df[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
            if 'accper' in df.columns:
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
            self._yearly_financial = df
            print(f"  年度财务数据: {len(df)} 条记录")
        except Exception as e:
            print(f"  年度财务数据加载失败: {e}")

        # ===== 7. 跨行业数据 =====
        try:
            df = pd.read_excel('stage17_final_cross_industry_mapping_table.xlsx', engine="openpyxl")
            if 'stock_code_norm' in df.columns:
                df['stock_code'] = df['stock_code_norm'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
            elif 'stock_code' in df.columns:
                df['stock_code'] = df['stock_code'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
            self._cross_industry = df
            print(f"  跨行业数据: {len(df)} 条记录")
        except Exception as e:
            print(f"  跨行业数据加载失败: {e}")

        # ===== 8. 行业关键词 =====
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

    def get_financial_rankings(self, stock_code: str, year: int = None) -> Dict[str, Any]:
        """获取财务指标排名数据 - 与前端 get_company_financial_rankings 完全一致"""
        code = self._normalize_code(stock_code)
        
        if self._financial_data is None or self._financial_data.empty:
            return {}
        
        stats_df = self._financial_data
        filtered = stats_df[stats_df['stock_code_norm'].astype(str).str.zfill(6) == code]
        
        if filtered.empty:
            return {}
        
        if year is not None:
            filtered = filtered[filtered['accper'] == year]
        
        if filtered.empty:
            # 取最新年份
            filtered = stats_df[stats_df['stock_code_norm'].astype(str).str.zfill(6) == code]
            filtered = filtered.sort_values('accper', ascending=False).head(1)
        
        if filtered.empty:
            return {}
        
        row = filtered.iloc[0]
        
        def safe_float(val):
            try:
                if pd.isna(val) or val in ('', 'N/A', 'None', 'null'):
                    return None
                return float(val)
            except (ValueError, TypeError):
                return None
        
        def safe_int(val):
            f = safe_float(val)
            return int(f) if f is not None else None
        
        # 与前端 data_loader.py 列名完全一致
        rankings = {
            'roe': {
                'value': safe_float(row.get('权益资本利润率ROE_raw_value')),
                'median': safe_float(row.get('权益资本利润率ROE_raw_median')),
                'percentile': safe_float(row.get('权益资本利润率ROE_percentile')),
                'rank': safe_int(row.get('权益资本利润率ROE_rank'))
            },
            'operating_margin': {
                'value': safe_float(row.get('营业利润率_raw_value')),
                'median': safe_float(row.get('营业利润率_raw_median')),
                'percentile': safe_float(row.get('营业利润率_percentile')),
                'rank': safe_int(row.get('营业利润率_rank'))
            },
            'roa': {
                'value': safe_float(row.get('总资产利润率ROA_raw_value')),
                'median': safe_float(row.get('总资产利润率ROA_raw_median')),
                'percentile': safe_float(row.get('总资产利润率ROA_percentile')),
                'rank': safe_int(row.get('总资产利润率ROA_rank'))
            },
            'ebitda_margin': {
                'value': safe_float(row.get('EBITDA利润率_raw_value')),
                'median': safe_float(row.get('EBITDA利润率_raw_median')),
                'percentile': safe_float(row.get('EBITDA利润率_percentile')),
                'rank': safe_int(row.get('EBITDA利润率_rank'))
            },
            'asset_turnover': {
                'value': safe_float(row.get('总资产周转率_raw_value')),
                'median': safe_float(row.get('总资产周转率_raw_median')),
                'percentile': safe_float(row.get('总资产周转率_percentile')),
                'rank': safe_int(row.get('总资产周转率_rank'))
            },
            'current_ratio': {
                'value': safe_float(row.get('流动比率_raw_value')),
                'median': safe_float(row.get('流动比率_raw_median')),
                'percentile': safe_float(row.get('流动比率_percentile')),
                'rank': safe_int(row.get('流动比率_rank'))
            },
            'debt_ratio': {
                'value': safe_float(row.get('资产负债率_raw_value')),
                'median': safe_float(row.get('资产负债率_raw_median')),
                'percentile': safe_float(row.get('资产负债率_percentile')),
                'rank': safe_int(row.get('资产负债率_rank'))
            }
        }
        return rankings

    def get_radar_data(self, stock_code: str) -> Dict[str, Any]:
        """获取雷达图数据 - 与前端 get_company_radar_data / get_company_indicator_radar_data 一致"""
        code = self._normalize_code(stock_code)
        radar = {'综合维度': {}, '指标维度': {}}
        
        stock_code_str = code
        try:
            stock_code_int = str(int(float(stock_code_str)))
        except:
            stock_code_int = stock_code_str
        
        # ----- 综合维度 -----
        if self._comprehensive_json:
            matched_key = None
            for key in self._comprehensive_json.keys():
                if key.endswith(f'_{stock_code_str}') or key.endswith(f'_{stock_code_str}.0') or \
                   key.endswith(f'_{stock_code_int}') or key.endswith(f'_{stock_code_int}.0'):
                    matched_key = key
                    break
                data_stock_code = self._comprehensive_json[key].get('stock_code', '')
                if str(data_stock_code) == stock_code_str or str(data_stock_code) == stock_code_int:
                    matched_key = key
                    break
                try:
                    data_stock_code_int = str(int(float(str(data_stock_code))))
                    if data_stock_code_int == stock_code_int:
                        matched_key = key
                        break
                except:
                    pass
            
            if matched_key:
                data = self._comprehensive_json[matched_key]
                radar['综合维度'] = {
                    'dimensions': data.get('dimensions', []),
                    'scores': data.get('scores', [])
                }
            else:
                print(f"[警告] 综合雷达图未找到公司 {code}")
        
        # ----- 指标维度 -----
        if self._dimension_json:
            matched_key = None
            for key in self._dimension_json.keys():
                if key.endswith(f'_{stock_code_str}') or key.endswith(f'_{stock_code_str}.0') or \
                   key.endswith(f'_{stock_code_int}') or key.endswith(f'_{stock_code_int}.0'):
                    matched_key = key
                    break
            
            if matched_key:
                data = self._dimension_json[matched_key]
                indicators = {}
                for dim_key, dim_val in data.items():
                    if isinstance(dim_val, dict) and 'indicators' in dim_val:
                        dim_name = dim_val.get('dimension_name', dim_key.replace('_score', ''))
                        indicators[dim_name] = {
                            'indicators': dim_val.get('indicators', []),
                            'scores': dim_val.get('scores', [])
                        }
                radar['指标维度'] = indicators
            else:
                print(f"[警告] 维度雷达图未找到公司 {code}")
        
        return radar

    def get_dimension_trend(self, stock_code: str) -> Dict[str, Any]:
        """获取各维度五年趋势数据 - 与前端条形图逻辑一致（多行格式）"""
        code = self._normalize_code(stock_code)
        trend = {}
        
        if self._dimension_scores is None or self._dimension_scores.empty:
            return trend
        
        score_df = self._dimension_scores
        company_scores = score_df[score_df['stock_code_norm'].astype(str).str.zfill(6) == code]
        
        if company_scores.empty:
            print(f"[警告] 维度得分表未找到公司 {code}")
            return trend
        
        # 查找 _score 列（排除 composite_score）
        dim_score_cols = [col for col in company_scores.columns 
                          if col.endswith('_score') and col != 'composite_score']
        
        dim_scores = {}
        for col in dim_score_cols:
            dim_name = col.replace('_score', '')
            
            # 2024年数据
            data_2024 = company_scores[company_scores['year'] == 2024]
            val_2024 = data_2024[col].iloc[0] if not data_2024.empty else None
            
            # 5年均值 (2020-2024)
            years = [2020, 2021, 2022, 2023, 2024]
            data_5y = company_scores[company_scores['year'].isin(years)]
            val_5y = data_5y[col].mean() if not data_5y.empty else None
            
            if val_2024 is not None or val_5y is not None:
                dim_scores[dim_name] = {
                    '2024': val_2024,
                    '5年均值': val_5y
                }
        
        if dim_scores:
            trend['维度趋势'] = dim_scores
        
        return trend

    def get_dimension_detail(self, stock_code: str) -> Dict[str, Any]:
        """获取各能力维度的详细数据 - 与前端 load_raw_financial_indicators 一致"""
        code = self._normalize_code(stock_code)
        detail = {}
        
        # 1. 从JSON获取维度指标列表和归一化得分
        if self._dimension_json:
            stock_code_str = code
            try:
                stock_code_int = str(int(float(stock_code_str)))
            except:
                stock_code_int = stock_code_str
            
            matched_key = None
            for key in self._dimension_json.keys():
                if key.endswith(f'_{stock_code_str}') or key.endswith(f'_{stock_code_str}.0') or \
                   key.endswith(f'_{stock_code_int}') or key.endswith(f'_{stock_code_int}.0'):
                    matched_key = key
                    break
            
            if matched_key:
                data = self._dimension_json[matched_key]
                for dim_key, dim_val in data.items():
                    if isinstance(dim_val, dict) and 'indicators' in dim_val:
                        dim_name = dim_val.get('dimension_name', dim_key.replace('_score', ''))
                        detail[dim_name] = {
                            'indicators': dim_val.get('indicators', []),
                            'scores': dim_val.get('scores', [])  # 归一化得分 0-1
                        }
        
        if not detail:
            return detail
        
        # 2. 从CSV获取原始值和行业对比数据（宽表→长表逻辑）
        if self._benchmark_data is not None and not self._benchmark_data.empty:
            company_data = self._benchmark_data[self._benchmark_data['stock_code_norm'].astype(str).str.zfill(6) == code]
            if not company_data.empty:
                # 取最新期
                if 'accper' in company_data.columns:
                    company_data = company_data.sort_values('accper', ascending=False).head(1)
                row = company_data.iloc[0]
                
                for dim_name, dim_info in detail.items():
                    indicators = dim_info['indicators']
                    company_vals = []
                    industry_medians = []
                    percentiles = []
                    ranks = []
                    
                    for ind in indicators:
                        # 直接匹配列名
                        vcol = ind + '_raw_value'
                        mcol = ind + '_raw_median'
                        pcol = ind + '_percentile'
                        rcol = ind + '_rank'
                        
                        # 如果直接匹配失败，尝试模糊匹配（与前端一致）
                        if vcol not in row.index:
                            all_cols = [c for c in row.index if c.endswith('_raw_value')]
                            matches = get_close_matches(ind, [c[:-10] for c in all_cols], n=1, cutoff=0.7)
                            if matches:
                                base = matches[0]
                                vcol = base + '_raw_value'
                                mcol = base + '_raw_median'
                                pcol = base + '_percentile'
                                rcol = base + '_rank'
                        
                        company_vals.append(row.get(vcol) if vcol in row.index else None)
                        industry_medians.append(row.get(mcol) if mcol in row.index else None)
                        percentiles.append(row.get(pcol) if pcol in row.index else None)
                        ranks.append(row.get(rcol) if rcol in row.index else None)
                    
                    detail[dim_name]['company_values'] = company_vals
                    detail[dim_name]['industry_medians'] = industry_medians
                    detail[dim_name]['percentiles'] = percentiles
                    detail[dim_name]['ranks'] = ranks
        
        return detail

    def get_yearly_trend(self, stock_code: str, indicator: str) -> pd.DataFrame:
        """获取指定指标的年度趋势数据"""
        code = self._normalize_code(stock_code)
        if self._yearly_financial is None or self._yearly_financial.empty:
            return pd.DataFrame()

        code_col = None
        for col in self._yearly_financial.columns:
            if 'stock_code_norm' in col or 'code' in col.lower():
                code_col = col
                break
        if code_col is None:
            return pd.DataFrame()

        company_data = self._yearly_financial[self._yearly_financial[code_col] == code]
        if company_data.empty:
            return pd.DataFrame()

        if indicator not in company_data.columns:
            return pd.DataFrame()

        result = company_data[['year', indicator]].dropna().sort_values('year')
        return result

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
        """模块1：公司概况AI解读"""
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
            "请基于以下公司基本信息，生成一段专业的公司概况AI解读。"
            "严格要求："
            "直接输出最终分析内容，"
            "不要出现任何前言、身份说明或解释"
            "禁止使用以下表达："
            "1. 作为一位……"
            "2. 根据您提供的信息……"
            "3. 我认为……"
            "4. 我将从……分析"
            "5. 以下是分析……",
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
        """模块2：行业重分类AI解读"""
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
            "请基于以下行业重分类数据，生成一段专业的AI解读。"
            "严格要求："
            "直接输出最终分析内容，"
            "不要出现任何前言、身份说明或解释"
            "禁止使用以下表达："
            "1. 作为一位……"
            "2. 根据您提供的信息……"
            "3. 我认为……"
            "4. 我将从……分析"
            "5. 以下是分析……",
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
        """模块3：行业关键词AI解读"""
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
            "请基于以下行业关键词数据，生成一段专业的AI解读。"
            "严格要求："
            "直接输出最终分析内容，"
            "不要出现任何前言、身份说明或解释"
            "禁止使用以下表达："
            "1. 作为一位……"
            "2. 根据您提供的信息……"
            "3. 我认为……"
            "4. 我将从……分析"
            "5. 以下是分析……",
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
        """模块4：同行业相似公司AI解读"""
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
            "请基于以下同行业相似公司数据，生成一段专业的AI解读。",
            "严格要求："
            "直接输出最终分析内容，"
            "不要出现任何前言、身份说明或解释"
            "禁止使用以下表达："
            "1. 作为一位……"
            "2. 根据您提供的信息……"
            "3. 我认为……"
            "4. 我将从……分析"
            "5. 以下是分析……",
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

    # ---------- 2. 公司概况页 - 行业内关键指标排名 ----------

    def analyze_roe_ranking(self, stock_code: str) -> str:
        """模块5a：ROE排名AI解读"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        rankings = self.get_financial_rankings(code, year=2024)

        roe_data = rankings.get('roe', {})
        if not roe_data or roe_data.get('value') is None:
            return "ROE数据缺失，无法生成分析。"

        median_str = str(roe_data['median']) if roe_data['median'] is not None else 'N/A'
        rank_str = str(int(roe_data['rank'])) if roe_data['rank'] is not None and not pd.isna(roe_data['rank']) else 'N/A'
        pct_str = str(round(roe_data['percentile']*100, 1)) + "%" if roe_data['percentile'] is not None else 'N/A'

        data_lines = [
            "公司名称: " + company_name,
            "权益资本利润率(ROE):",
            "  公司值: " + str(roe_data['value']),
            "  行业中位数: " + median_str,
            "  行业排名: " + rank_str + " (1为最优)",
            "  百分位: " + pct_str + " (越高越好)"
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "请基于以下ROE排名数据，生成一段专业的AI解读。",
            "严格要求："
            "直接输出最终分析内容，"
            "不要出现任何前言、身份说明或解释"
            "禁止使用以下表达："
            "1. 作为一位……"
            "2. 根据您提供的信息……"
            "3. 我认为……"
            "4. 我将从……分析"
            "5. 以下是分析……",
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
        """模块5b：营业利润率排名AI解读"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        rankings = self.get_financial_rankings(code, year=2024)

        margin_data = rankings.get('operating_margin', {})
        if not margin_data or margin_data.get('value') is None:
            return "营业利润率数据缺失，无法生成分析。"

        median_str = str(margin_data['median']) if margin_data['median'] is not None else 'N/A'
        rank_str = str(int(margin_data['rank'])) if margin_data['rank'] is not None and not pd.isna(margin_data['rank']) else 'N/A'
        pct_str = str(round(margin_data['percentile']*100, 1)) + "%" if margin_data['percentile'] is not None else 'N/A'

        data_lines = [
            "公司名称: " + company_name,
            "营业利润率:",
            "  公司值: " + str(margin_data['value']),
            "  行业中位数: " + median_str,
            "  行业排名: " + rank_str + " (1为最优)",
            "  百分位: " + pct_str + " (越高越好)"
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "请基于以下营业利润率排名数据，生成一段专业的AI解读。",
            "严格要求："
            "直接输出最终分析内容，"
            "不要出现任何前言、身份说明或解释"
            "禁止使用以下表达："
            "1. 作为一位……"
            "2. 根据您提供的信息……"
            "3. 我认为……"
            "4. 我将从……分析"
            "5. 以下是分析……",
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
        """模块5c：总资产利润率排名AI解读"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        rankings = self.get_financial_rankings(code, year=2024)

        roa_data = rankings.get('roa', {})
        if not roa_data or roa_data.get('value') is None:
            return "ROA数据缺失，无法生成分析。"

        median_str = str(roa_data['median']) if roa_data['median'] is not None else 'N/A'
        rank_str = str(int(roa_data['rank'])) if roa_data['rank'] is not None and not pd.isna(roa_data['rank']) else 'N/A'
        pct_str = str(round(roa_data['percentile']*100, 1)) + "%" if roa_data['percentile'] is not None else 'N/A'

        data_lines = [
            "公司名称: " + company_name,
            "总资产利润率(ROA):",
            "  公司值: " + str(roa_data['value']),
            "  行业中位数: " + median_str,
            "  行业排名: " + rank_str + " (1为最优)",
            "  百分位: " + pct_str + " (越高越好)"
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "请基于以下ROA排名数据，生成一段专业的AI解读。",
            "严格要求："
            "直接输出最终分析内容，"
            "不要出现任何前言、身份说明或解释"
            "禁止使用以下表达："
            "1. 作为一位……"
            "2. 根据您提供的信息……"
            "3. 我认为……"
            "4. 我将从……分析"
            "5. 以下是分析……",
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
        """模块5d：财务指标排名总览AI解读"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        rankings = self.get_financial_rankings(code, year=2024)

        rankings_lines = []
        metric_display_map = {
            'roe': '权益资本利润率ROE',
            'operating_margin': '营业利润率',
            'roa': '总资产利润率ROA',
            'asset_turnover': '总资产周转率',
            'current_ratio': '流动比率',
            'debt_ratio': '资产负债率',
            'cash_creation_total': '总资产创现率',
            'cash_creation_sales': '销售创现率',
            'ebitda_margin': 'EBITDA利润率',
        }
        for en_key, display_name in metric_display_map.items():
            if en_key in rankings:
                r = rankings[en_key]
                val = r['value']
                median = r['median']
                pct = r['percentile']
                rank = r['rank']
                if val is not None and not pd.isna(val):
                    pct_str = str(round(pct * 100, 1)) + "%" if pct is not None and not pd.isna(pct) else "N/A"
                    rank_str = str(int(rank)) if rank is not None and not pd.isna(rank) else "N/A"
                    vs_median = "高于" if median is not None and val > median else "低于" if median is not None else "无法对比"
                    line = display_name + ": 公司值" + (str(round(val, 4)) if val is not None else 'N/A') + " | 行业中位数" + (str(round(median, 4)) if median is not None else 'N/A') + " | " + vs_median + "中位数 | 分位数" + pct_str + " | 排名" + rank_str
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
            "请基于以下财务指标排名总览数据，生成一段全面的AI解读。",
            "严格要求："
            "直接输出最终分析内容，"
            "不要出现任何前言、身份说明或解释"
            "禁止使用以下表达："
            "1. 作为一位……"
            "2. 根据您提供的信息……"
            "3. 我认为……"
            "4. 我将从……分析"
            "5. 以下是分析……",
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

    # ---------- 3. 公司概况页 - 雷达图 ----------

    def analyze_comprehensive_radar(self, stock_code: str) -> str:
        """模块6a：综合维度雷达图AI解读"""
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
            "请基于以下综合维度雷达图数据，生成一段专业的AI解读。",
            "严格要求："
            "直接输出最终分析内容，"
            "不要出现任何前言、身份说明或解释"
            "禁止使用以下表达："
            "1. 作为一位……"
            "2. 根据您提供的信息……"
            "3. 我认为……"
            "4. 我将从……分析"
            "5. 以下是分析……",
            "要求：",
            "1. 描述公司六维能力的整体画像和特征",
            "2. 指出最强和最弱的维度，分析原因",
            "3. 评价公司属于什么类型（高成长型/稳健型/成熟型/问题型等）",
            "4. 分析各维度之间的协同或矛盾关系",
            "5. 不限制字数，充分展开分析",
            "6. 【重要】所有维度得分均为百分制（满分100分，绝非10分制）。禁止将得分理解为满分10分或满分5分",
            "7. 分析公司当前六维能力的整体画像，结合具体得分",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2000, temperature=0.3)

    def analyze_indicator_radar(self, stock_code: str) -> str:
        """模块6b：指标级雷达图AI解读"""
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
            "请基于以下指标级雷达图数据，生成一段专业的AI解读。",
            "严格要求："
            "直接输出最终分析内容，"
            "不要出现任何前言、身份说明或解释"
            "禁止使用以下表达："
            "1. 作为一位……"
            "2. 根据您提供的信息……"
            "3. 我认为……"
            "4. 我将从……分析"
            "5. 以下是分析……",
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
        """模块7a：综合财务雷达图分析 - 增强数据输入"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        radar = self.get_radar_data(code)
        trend = self.get_dimension_trend(code)
        rankings = self.get_financial_rankings(code, year=2024)
        detail = self.get_dimension_detail(code)
        industry_info = self.get_industry_info(code)

        # 格式化雷达图数据
        radar_lines = []
        if radar['综合维度'].get('dimensions'):
            for dim, score in zip(radar['综合维度']['dimensions'], radar['综合维度']['scores']):
                radar_lines.append(f"{dim}: {round(score * 100, 1)}分")
        else:
            radar_lines.append("雷达图数据缺失")

        # 格式化趋势数据
        trend_lines = []
        if '维度趋势' in trend:
            for dim, vals in trend['维度趋势'].items():
                v2024 = vals.get('2024')
                v5y = vals.get('5年均值')
                if v2024 is not None and v5y is not None:
                    change = v2024 - v5y
                    trend_dir = "提升" if change > 0 else "下降" if change < 0 else "持平"
                    trend_lines.append(f"{dim}: 2024年{round(v2024, 2)} vs 5年均值{round(v5y, 2)} ({trend_dir} {round(abs(change), 2)})")
                elif v2024 is not None:
                    trend_lines.append(f"{dim}: 2024年{round(v2024, 2)} (5年均值: 数据缺失)")
                elif v5y is not None:
                    trend_lines.append(f"{dim}: 5年均值{round(v5y, 2)} (2024年: 数据缺失)")
        else:
            trend_lines.append("趋势数据缺失")

        # 格式化详细排名数据
        rankings_lines = []
        metric_display_map = {
            'roe': '权益资本利润率ROE',
            'operating_margin': '营业利润率',
            'roa': '总资产利润率ROA',
            'asset_turnover': '总资产周转率',
            'current_ratio': '流动比率',
            'debt_ratio': '资产负债率',
            'cash_creation_total': '总资产创现率',
            'cash_creation_sales': '销售创现率',
            'ebitda_margin': 'EBITDA利润率',
        }
        for en_key, display_name in metric_display_map.items():
            if en_key in rankings:
                r = rankings[en_key]
                val = r['value']
                median = r['median']
                pct = r['percentile']
                rank = r['rank']
                if val is not None and not pd.isna(val):
                    pct_str = f"{round(pct * 100, 1)}%" if pct is not None and not pd.isna(pct) else "N/A"
                    rank_str = str(int(rank)) if rank is not None and not pd.isna(rank) else "N/A"
                    vs_median = "高于" if median is not None and val > median else "低于" if median is not None else "无法对比"
                    line = f"{display_name}: 公司值{round(val, 4)} | 行业中位数{round(median, 4) if median is not None else 'N/A'} | {vs_median}中位数 | 分位数{pct_str} | 排名{rank_str}"
                    rankings_lines.append(line)

        rankings_text = "\n".join(rankings_lines) if rankings_lines else "排名数据缺失"

        # 格式化各维度详细指标数据（同时显示归一化得分+原始值）
        detail_lines = []
        if detail:
            for dim_name, dim_data in detail.items():
                detail_lines.append(f"\n【{dim_name}】")
                indicators = dim_data.get('indicators', [])
                scores = dim_data.get('scores', [])  # 归一化得分
                company_values = dim_data.get('company_values', [])  # 原始值
                medians = dim_data.get('industry_medians', [])
                percentiles = dim_data.get('percentiles', [])
                ranks = dim_data.get('ranks', [])

                for i, ind in enumerate(indicators):
                    score = scores[i] if i < len(scores) else 'N/A'
                    val = company_values[i] if i < len(company_values) else 'N/A'
                    median = medians[i] if i < len(medians) and medians[i] is not None else 'N/A'
                    pct = percentiles[i] if i < len(percentiles) and percentiles[i] is not None else 'N/A'
                    rank = ranks[i] if i < len(ranks) and ranks[i] is not None else 'N/A'

                    score_str = f"{round(score * 100, 1)}%" if isinstance(score, (int, float)) else str(score)
                    val_str = f"{val:.5f}" if isinstance(val, (int, float)) and not pd.isna(val) else str(val)
                    median_str = f"{median:.5f}" if isinstance(median, (int, float)) and not pd.isna(median) else str(median)
                    pct_str = f"{pct:.3f}" if isinstance(pct, (int, float)) and not pd.isna(pct) else str(pct)
                    rank_str = str(int(rank)) if isinstance(rank, (int, float)) and not pd.isna(rank) else str(rank)

                    detail_lines.append(f"  {ind}: 归一化得分{score_str} | 公司值{val_str} | 行业中位数{median_str} | 分位数{pct_str} | 排名{rank_str}")
        else:
            detail_lines.append("详细指标数据缺失")

        detail_text = "\n".join(detail_lines)

        data_lines = [
            f"公司名称: {company_name}",
            f"所属行业: {industry_info['三级行业']}",
            "",
            "=== 综合财务雷达图数据 ===",
            "\n".join(radar_lines),
            "",
            "=== 维度得分趋势对比 ===",
            "\n".join(trend_lines),
            "",
            "=== 核心财务指标行业排名 ===",
            rankings_text,
            "",
            "=== 各维度详细指标数据 ===",
            detail_text
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "请基于以下全面的财务数据，生成一段专业的综合财务雷达图AI解读。",
            "严格要求：",
            "直接输出最终分析内容，",
            "不要出现任何前言、身份说明或解释",
            "禁止使用以下表达：",
            "1. 作为一位……",
            "2. 根据您提供的信息……",
            "3. 我认为……",
            "4. 我将从……分析",
            "5. 以下是分析……",
            "要求：",
            "1. 分析公司当前六维能力的整体画像，结合具体得分",
            "2. 结合五年趋势，评价各维度的发展轨迹和改善/恶化情况",
            "3. 结合核心财务指标排名，分析各维度的行业相对位置",
            "4. 结合各维度详细指标数据，深入分析具体指标的强弱",
            "5. 指出哪些维度在持续改善，哪些在退步，分析可能原因",
            "6. 综合评价公司的财务健康度和成长性",
            "7. 不限制字数，充分展开分析，像一份专业的财务诊断报告",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=4000, temperature=0.3)

    def analyze_dimension_trend(self, stock_code: str) -> str:
        """模块7b：维度得分趋势对比分析"""
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
            "请基于以下维度得分趋势数据，生成一段专业的AI解读。",
            "严格要求：",
            "直接输出最终分析内容，",
            "不要出现任何前言、身份说明或解释",
            "禁止使用以下表达：",
            "1. 作为一位……",
            "2. 根据您提供的信息……",
            "3. 我认为……",
            "4. 我将从……分析",
            "5. 以下是分析……",
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
        """辅助函数：格式化维度详细数据 - 与前端一致（5位小数）"""
        indicators = dim_data.get('indicators', [])
        company_values = dim_data.get('company_values', [])  # 原始值
        medians = dim_data.get('industry_medians', [])
        percentiles = dim_data.get('percentiles', [])
        ranks = dim_data.get('ranks', [])

        detail_lines = []
        for i, ind in enumerate(indicators):
            val = company_values[i] if i < len(company_values) else 'N/A'
            median = medians[i] if i < len(medians) and medians[i] is not None else 'N/A'
            pct = percentiles[i] if i < len(percentiles) and percentiles[i] is not None else 'N/A'
            rank = ranks[i] if i < len(ranks) and ranks[i] is not None else 'N/A'

            val_str = f"{val:.5f}" if isinstance(val, (int, float)) and not pd.isna(val) else str(val)
            median_str = f"{median:.5f}" if isinstance(median, (int, float)) and not pd.isna(median) else str(median)
            pct_str = f"{pct:.3f}" if isinstance(pct, (int, float)) and not pd.isna(pct) else str(pct)
            rank_str = str(int(rank)) if isinstance(rank, (int, float)) and not pd.isna(rank) else str(rank)

            line = f"  {ind}: 公司值{val_str} | 行业中位数{median_str} | 分位数{pct_str} | 排名{rank_str}"
            detail_lines.append(line)
        return "\n".join(detail_lines)

    def analyze_profitability(self, stock_code: str) -> str:
        """模块7c：盈利能力指标分析"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        detail = self.get_dimension_detail(code)
        rankings = self.get_financial_rankings(code, year=2024)

        profitability_dim = None
        for dim_name in detail.keys():
            if '盈利' in dim_name or 'profit' in dim_name.lower():
                profitability_dim = dim_name
                break

        if not profitability_dim:
            return "盈利能力维度数据缺失，无法生成分析。"

        dim_data = detail[profitability_dim]
        detail_text = self._format_dimension_detail(dim_data)

        related_rankings = []
        for en_key, display_name in [('roe', 'ROE'), ('operating_margin', '营业利润率'), ('roa', 'ROA'), ('ebitda_margin', 'EBITDA利润率')]:
            if en_key in rankings:
                r = rankings[en_key]
                pct_str = f"{round(r['percentile']*100, 1)}%" if r['percentile'] is not None else 'N/A'
                rank_str = str(int(r['rank'])) if r['rank'] is not None and not pd.isna(r['rank']) else 'N/A'
                val_str = f"{round(r['value'], 4)}" if r['value'] is not None else "N/A"
                related_rankings.append(f"{display_name}: 公司值{val_str} | 分位数{pct_str} | 排名{rank_str}")

        data_lines = [
            "公司名称: " + company_name,
            "盈利能力维度分析:",
            "维度名称: " + profitability_dim,
            "指标详情:",
            detail_text,
            "",
            "相关核心指标排名:",
            "\n".join(related_rankings) if related_rankings else "数据缺失"
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "你是一位资深财务分析师。请基于以下盈利能力指标数据，生成一段专业的AI解读。",
            "要求：",
            "1. 分析公司盈利能力的整体水平和结构特征",
            "2. 对比行业均值，评价各盈利指标的相对强弱",
            "3. 从毛利率、净利率、ROE、ROA等多角度分析盈利质量",
            "4. 结合核心指标排名，分析公司在行业中的盈利地位",
            "5. 分析盈利能力的可持续性和增长潜力",
            "6. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2500, temperature=0.3)

    def analyze_asset_efficiency(self, stock_code: str) -> str:
        """模块7d：资产使用效率指标分析"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        detail = self.get_dimension_detail(code)
        rankings = self.get_financial_rankings(code, year=2024)

        efficiency_dim = None
        for dim_name in detail.keys():
            if '资产' in dim_name or '效率' in dim_name or '周转' in dim_name or 'efficiency' in dim_name.lower():
                efficiency_dim = dim_name
                break

        if not efficiency_dim:
            return "资产效率维度数据缺失，无法生成分析。"

        dim_data = detail[efficiency_dim]
        detail_text = self._format_dimension_detail(dim_data)

        related_rankings = []
        for en_key, display_name in [('asset_turnover', '总资产周转率'), ('inventory_turnover', '存货周转率')]:
            if en_key in rankings:
                r = rankings[en_key]
                pct_str = f"{round(r['percentile']*100, 1)}%" if r['percentile'] is not None else 'N/A'
                rank_str = str(int(r['rank'])) if r['rank'] is not None and not pd.isna(r['rank']) else 'N/A'
                val_str = f"{round(r['value'], 4)}" if r['value'] is not None else "N/A"
                related_rankings.append(f"{display_name}: 公司值{val_str} | 分位数{pct_str} | 排名{rank_str}")
        data_lines = [
            "公司名称: " + company_name,
            "资产使用效率维度分析:",
            "维度名称: " + efficiency_dim,
            "指标详情:",
            detail_text,
            "",
            "相关核心指标排名:",
            "\n".join(related_rankings) if related_rankings else "数据缺失"
        ]
        data = "\n".join(data_lines)

        prompt_lines = [
            "请基于以下资产使用效率指标数据，生成一段专业的AI解读。",
            "严格要求：",
            "直接输出最终分析内容，",
            "不要出现任何前言、身份说明或解释",
            "禁止使用以下表达：",
            "1. 作为一位……",
            "2. 根据您提供的信息……",
            "3. 我认为……",
            "4. 我将从……分析",
            "5. 以下是分析……",
            "",
            "要求：",
            "1. 分析公司资产运营效率的整体水平",
            "2. 对比行业均值，评价各效率指标的相对强弱",
            "3. 从总资产周转率、存货周转率、应收账款周转率等角度分析资产利用效率",
            "4. 结合核心指标排名，分析公司在行业中的效率地位",
            "5. 分析资产效率对ROE的驱动作用",
            "6. 不限制字数，充分展开分析",
            "",
            "数据：",
            data
        ]
        prompt = "\n".join(prompt_lines)

        return self.call_qwen(prompt, max_tokens=2500, temperature=0.3)

    def analyze_liquidity(self, stock_code: str) -> str:
        """模块7e：流动性指标分析"""
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
            "请基于以下流动性指标数据，生成一段专业的AI解读。",
            "严格要求：",
            "直接输出最终分析内容，",
            "不要出现任何前言、身份说明或解释",
            "禁止使用以下表达：",
            "1. 作为一位……",
            "2. 根据您提供的信息……",
            "3. 我认为……",
            "4. 我将从……分析",
            "5. 以下是分析……",
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
        """模块7f：现金创造能力指标分析"""
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
            "请基于以下现金创造能力指标数据，生成一段专业的AI解读。",
            "严格要求：",
            "直接输出最终分析内容，",
            "不要出现任何前言、身份说明或解释",
            "禁止使用以下表达：",
            "1. 作为一位……",
            "2. 根据您提供的信息……",
            "3. 我认为……",
            "4. 我将从……分析",
            "5. 以下是分析……",
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
        """模块7g：偿债能力指标分析"""
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
            "请基于以下偿债能力指标数据，生成一段专业的AI解读。",
            "严格要求：",
            "直接输出最终分析内容，",
            "不要出现任何前言、身份说明或解释",
            "禁止使用以下表达：",
            "1. 作为一位……",
            "2. 根据您提供的信息……",
            "3. 我认为……",
            "4. 我将从……分析",
            "5. 以下是分析……",
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
        """模块7h：股东收益指标分析"""
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
            "请基于以下股东收益指标数据，生成一段专业的AI解读。",
            "严格要求：",
            "直接输出最终分析内容，",
            "不要出现任何前言、身份说明或解释",
            "禁止使用以下表达：",
            "1. 作为一位……",
            "2. 根据您提供的信息……",
            "3. 我认为……",
            "4. 我将从……分析",
            "5. 以下是分析……",
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
        """模块8a：基本结论"""
        code = self._normalize_code(stock_code)
        company_name = self.get_company_name(code)
        industry_info = self.get_industry_info(code)
        rankings = self.get_financial_rankings(code, year=2024)
        radar = self.get_radar_data(code)
        trend = self.get_dimension_trend(code)

        advantages = []
        weaknesses = []
        metric_display_map = {
            'roe': '权益资本利润率ROE',
            'operating_margin': '营业利润率',
            'roa': '总资产利润率ROA',
            'asset_turnover': '总资产周转率',
            'current_ratio': '流动比率',
            'debt_ratio': '资产负债率',
        }
        for en_key, display_name in metric_display_map.items():
            if en_key in rankings:
                r = rankings[en_key]
                pct = r['percentile']
                if pct is not None and not pd.isna(pct):
                    if pct > 0.7:
                        advantages.append(display_name + " (分位数" + str(round(pct * 100, 1)) + "%)")
                    elif pct < 0.3:
                        weaknesses.append(display_name + " (分位数" + str(round(pct * 100, 1)) + "%)")

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
            "请基于以下公司数据，生成一段全面的基本结论AI解读。"
            "严格要求："
            "直接输出最终分析内容，"
            "不要出现任何前言、身份说明或解释"
            "禁止使用以下表达："
            "1. 作为一位……"
            "2. 根据您提供的信息……"
            "3. 我认为……"
            "4. 我将从……分析"
            "5. 以下是分析……",
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
        rankings = self.get_financial_rankings(code, year=2024)
        radar = self.get_radar_data(code)
        trend = self.get_dimension_trend(code)

        advantages = []
        weaknesses = []
        metric_display_map = {
            'roe': '权益资本利润率ROE',
            'operating_margin': '营业利润率',
            'roa': '总资产利润率ROA',
            'asset_turnover': '总资产周转率',
            'current_ratio': '流动比率',
            'debt_ratio': '资产负债率',
            'cash_creation_total': '总资产创现率',
            'cash_creation_sales': '销售创现率',
        }
        for en_key, display_name in metric_display_map.items():
            if en_key in rankings:
                r = rankings[en_key]
                pct = r['percentile']
                if pct is not None and not pd.isna(pct):
                    if pct > 0.7:
                        advantages.append(display_name + " (分位数" + str(round(pct * 100, 1)) + "%)")
                    elif pct < 0.3:
                        weaknesses.append(display_name + " (分位数" + str(round(pct * 100, 1)) + "%)")

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
            "请基于以下公司财务数据，生成一段全面的管理建议AI解读。",
            "严格要求："
            "直接输出最终分析内容，"
            "不要出现任何前言、身份说明或解释"
            "禁止使用以下表达："
            "1. 作为一位……"
            "2. 根据您提供的信息……"
            "3. 我认为……"
            "4. 我将从……分析"
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
        rankings = self.get_financial_rankings(code, year=2024)
        radar = self.get_radar_data(code)
        trend = self.get_dimension_trend(code)

        risks = []
        risk_details = []

        risk_metrics = {
            'debt_ratio': ('资产负债率', lambda pct: pct > 0.7, '资产负债率偏高'),
            'current_ratio': ('流动比率', lambda pct: pct < 0.3, '流动性不足'),
        }
        for en_key, (display_name, check_risk, risk_label) in risk_metrics.items():
            if en_key in rankings:
                r = rankings[en_key]
                pct = r['percentile']
                if pct is not None and not pd.isna(pct):
                    if check_risk(pct):
                        risks.append(risk_label + " (分位数" + str(round(pct * 100, 1)) + "%)")
                        rd = display_name + ": 公司值" + str(round(r['value'], 4))
                        rd += " | 行业中位数" + str(round(r['median'], 4) if r['median'] is not None else 'N/A')
                        rd += " | 排名" + str(int(r['rank']) if r['rank'] is not None and not pd.isna(r['rank']) else 'N/A')
                        risk_details.append(rd)

        radar_risks = []
        if radar['综合维度'].get('dimensions'):
            for dim, score in zip(radar['综合维度']['dimensions'], radar['综合维度']['scores']):
                if score < 0.3:
                    radar_risks.append(dim + "能力严重不足 (" + str(round(score * 100, 1)) + "分)")

        trend_risks = []
        if '维度趋势' in trend:
            for dim, vals in trend['维度趋势'].items():
                v2024 = vals.get('2024')
                v5y = vals.get('5年均值')
                if v2024 is not None and v5y is not None:
                    change = v2024 - v5y
                    if change < -0.15:
                        trend_risks.append(dim + "能力持续恶化 (5年下降" + str(round(abs(change), 2)) + ")")

        roe_pct = str(round(rankings.get('roe', {}).get('percentile', 0) * 100, 1)) if 'roe' in rankings else 'N/A'
        op_pct = str(round(rankings.get('operating_margin', {}).get('percentile', 0) * 100, 1)) if 'operating_margin' in rankings else 'N/A'
        debt_pct = str(round(rankings.get('debt_ratio', {}).get('percentile', 0) * 100, 1)) if 'debt_ratio' in rankings else 'N/A' 

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
            "请基于以下风险数据，生成一段全面的风险提示AI解读。"
            "严格要求："
            "直接输出最终分析内容，"
            "不要出现任何前言、身份说明或解释"
            "禁止使用以下表达："
            "1. 作为一位……"
            "2. 根据您提供的信息……"
            "3. 我认为……"
            "4. 我将从……分析"
            "5. 以下是分析……",
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
            '财务排名': self.get_financial_rankings(code, year=2024),
            '雷达图数据': self.get_radar_data(code),
            '维度趋势': self.get_dimension_trend(code),
            '维度详情': self.get_dimension_detail(code)
        }


# ==================== 便捷函数 ====================

def analyze_module(stock_code: str, module: str, api_key: str, model: str = "qwen-turbo") -> str:
    service = FinancialAIReport(api_key=api_key, model=model)

    module_map = {
        'overview': service.analyze_company_overview,
        'industry': service.analyze_industry_reclassification,
        'keywords': service.analyze_industry_keywords,
        'similar': service.analyze_similar_companies,
        'roe_ranking': service.analyze_roe_ranking,
        'margin_ranking': service.analyze_operating_margin_ranking,
        'roa_ranking': service.analyze_roa_ranking,
        'rankings_overview': service.analyze_financial_rankings_overview,
        'comprehensive_radar': service.analyze_comprehensive_radar,
        'indicator_radar': service.analyze_indicator_radar,
        'financial_radar': service.analyze_comprehensive_financial_radar,
        'dimension_trend': service.analyze_dimension_trend,
        'profitability': service.analyze_profitability,
        'asset_efficiency': service.analyze_asset_efficiency,
        'liquidity': service.analyze_liquidity,
        'cash_creation': service.analyze_cash_creation,
        'solvency': service.analyze_solvency,
        'shareholder_return': service.analyze_shareholder_return,
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