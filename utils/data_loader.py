import pandas as pd
import os
from pathlib import Path
import streamlit as st
import json

@st.cache_data
def load_main_data():
    """加载主数据文件"""
    df = pd.read_excel('complete_company_industry_mapping_v5_qwen_level1_final.xlsx')
    # 标准化列名，处理可能的命名差异
    column_mapping = {}
    
    # 跟踪已经映射的列，避免重复映射
    mapped_symbol = False
    mapped_name = False
    
    for col in df.columns:
        normalized_col = str(col).strip().lower()
        
        if not mapped_symbol and normalized_col in ['股票代码', '证券代码', 'code', 'symbol', '股票CODE', 'code_num', 'stock_code']:
            column_mapping[col] = 'symbol'
            mapped_symbol = True
        elif not mapped_name and normalized_col in ['公司名称', '股票名称', 'name', '股票简称', 'sec_name', 'short_name', 'stock_name', 'company_name']:
            column_mapping[col] = 'name'
            mapped_name = True
        elif normalized_col in ['行业一级', '行业分类一级', '一级行业', 'industry_level1', 'industry_l1', 'sw_l1', '申万一级', 'first_industry', 'industry1', 'final_level1_label']:
            if 'industry_level1' not in column_mapping.values():
                column_mapping[col] = 'industry_level1'
        elif normalized_col in ['行业二级', '行业分类二级', '二级行业', 'industry_level2', 'industry_l2', 'sw_l2', '申万二级', 'second_industry', 'industry2', 'final_level2_label']:
            if 'industry_level2' not in column_mapping.values():
                column_mapping[col] = 'industry_level2'
        elif normalized_col in ['行业三级', '行业分类三级', '三级行业', 'industry_level3', 'industry_l3', 'sw_l3', '申万三级', 'third_industry', 'industry3', 'final_level3_label']:
            if 'industry_level3' not in column_mapping.values():
                column_mapping[col] = 'industry_level3'
        elif normalized_col in ['roe', '净资产收益率', 'return_on_equity', 'roa', '资产收益率', 'net_asset_yield']:
            if 'roe' not in column_mapping.values():
                column_mapping[col] = 'roe'
        elif normalized_col in ['毛利率', 'gross_margin', '销售毛利率', 'gross_profit_margin']:
            if 'gross_margin' not in column_mapping.values():
                column_mapping[col] = 'gross_margin'
        elif normalized_col in ['营收增长率', 'revenue_growth', '营业收入增长率', 'income_growth']:
            if 'revenue_growth' not in column_mapping.values():
                column_mapping[col] = 'revenue_growth'
        elif normalized_col in ['上市日期', 'listing_date', 'ipo_date', '上市时间', '挂牌日期', 'first_list_date']:
            if 'listing_date' not in column_mapping.values():
                column_mapping[col] = 'listing_date'
        elif normalized_col in ['year', '上市年份', 'ipo_year', 'first_list_year']:
            if 'year' not in column_mapping.values():
                column_mapping[col] = 'year'
    
    df.rename(columns=column_mapping, inplace=True)
    
    # 如果存在重复的列（如多个名称列），保留第一个
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 确保股票代码以字符串形式显示，保留前导零
    if 'symbol' in df.columns:
        # 将股票代码转换为字符串，确保保留6位格式（如000001）
        df['symbol'] = df['symbol'].apply(lambda x: str(int(x)).zfill(6) if pd.notna(x) else '')
    
    return df

@st.cache_data
def load_stock_list():
    """加载股票列表"""
    df = load_main_data()
    return df

@st.cache_data
def load_industry_classification():
    """加载行业分类数据"""
    df = load_main_data()
    return df

@st.cache_data
def load_financial_data(stock_code):
    """
    加载指定股票的财务数据
    这里需要根据实际数据结构调整
    """
    # 由于缺少专门的财务数据文件，暂时返回空字典
    # 如果 complete_company_industry_mapping_v4_stage16D_checked(3).xlsx 包含所需财务数据，可以在此处添加相应逻辑
    financial_data = {}
    
    # 尝试从主数据文件加载财务数据
    try:
        main_df = load_main_data()  # 使用缓存的主数据函数
        # 根据实际数据结构提取财务数据
        # 这里需要根据实际的列名和数据结构进行调整
        stock_data = main_df[main_df['symbol'].astype(str) == str(stock_code)]
        if not stock_data.empty:
            # 假设有一些财务指标列，根据实际数据结构修改
            financial_data['latest'] = stock_data.iloc[0].to_dict()  # 将行转换为字典格式，便于后续使用
    except Exception:
        pass
    
    return financial_data

@st.cache_data
def get_company_basic_info(stock_code):
    """获取公司基本信息"""
    stocks_df = load_stock_list()
    industry_df = load_industry_classification()
    
    company_info = stocks_df[stocks_df['symbol'].astype(str) == str(stock_code)]
    
    if company_info.empty:
        return None
    
    company = company_info.iloc[0]
    
    industry_info = industry_df[industry_df['symbol'].astype(str) == str(stock_code)]
    
    # 获取行业信息时添加安全检查
    industry_row = industry_info.iloc[0] if not industry_info.empty else None
    
    # 获取上市日期 - 优先从 IPO_Cobasic.csv 获取（Listdt列）
    listing_date = ''
    
    # 先尝试从 IPO_Cobasic.csv 获取上市日期
    listing_dates_df = load_listing_dates()
    stock_code_str = str(company['symbol'])
    
    # 尝试多种格式匹配（6位完整代码和去掉前导零的代码）
    matched = listing_dates_df[
        (listing_dates_df['stock_code'] == stock_code_str) |
        (listing_dates_df['stock_code'] == stock_code_str.zfill(6)) |
        (listing_dates_df['stock_code'] == str(int(float(stock_code_str))))
    ]
    
    if not matched.empty:
        listing_date = str(matched.iloc[0]['listing_date'])
        # 清理日期格式
        if ' 00:00:00' in listing_date:
            listing_date = listing_date.replace(' 00:00:00', '')
        elif 'T00:00:00' in listing_date:
            listing_date = listing_date.replace('T00:00:00', '')
    
    # 如果没有找到，尝试从主数据文件的 year 列获取
    if not listing_date and 'year' in company.index and pd.notna(company['year']):
        year_value = company['year']
        if isinstance(year_value, (int, float)):
            listing_date = f"{int(year_value)}-01-01"
        else:
            listing_date = str(year_value)
    
    # 如果仍没有，尝试从主数据文件的 listing_date 列获取
    if not listing_date and 'listing_date' in company.index and pd.notna(company['listing_date']):
        listing_date = str(company['listing_date'])
    
    basic_info = {
        '股票代码': str(company['symbol']),
        '公司名称': company['name'],
        '上市日期': listing_date if listing_date else '待补充',
        '所属行业一级': industry_row['industry_level1'] if industry_row is not None and 'industry_level1' in industry_row.index else '未知',
        '所属行业二级': industry_row['industry_level2'] if industry_row is not None and 'industry_level2' in industry_row.index else '未知',
        '所属行业三级': industry_row['industry_level3'] if industry_row is not None and 'industry_level3' in industry_row.index else '未知',
    }
    
    return basic_info

@st.cache_data
def calculate_industry_rankings(stock_code, metric='roe'):
    """
    计算公司在行业内的排名和分位数
    metric: 'roe', 'gross_margin', 'revenue_growth'
    """
    # 加载主数据
    main_df = load_main_data()
    
    # 获取目标公司的数据
    company_data = main_df[main_df['symbol'].astype(str) == str(stock_code)]
    
    if company_data.empty:
        return {
            'metric_name': metric,
            'company_value': 0.0,
            'industry_average': 0.0,
            'percentile': 0.0,
            'rank': 0,
            'total_companies': 0
        }
    
    # 检查指标列是否存在
    if metric not in company_data.columns:
        return {
            'metric_name': metric,
            'company_value': 0.0,
            'industry_average': 0.0,
            'percentile': 0.0,
            'rank': 0,
            'total_companies': 0
        }
    
    # 获取公司当前指标值
    company_value = company_data.iloc[0][metric]
    
    # 获取同行业的其他公司
    company_industry = company_data.iloc[0].get('industry_level1', '')
    industry_companies = main_df[main_df['industry_level1'] == company_industry] if 'industry_level1' in main_df.columns else main_df
    
    # 筛选非空指标值的公司
    valid_companies = industry_companies[industry_companies[metric].notna()]
    
    if len(valid_companies) == 0:
        return {
            'metric_name': metric,
            'company_value': float(company_value) if pd.notna(company_value) else 0.0,
            'industry_average': 0.0,
            'percentile': 0.0,
            'rank': 0,
            'total_companies': 0
        }
    
    # 计算行业平均值
    industry_average = valid_companies[metric].mean()
    
    # 计算公司在行业中的排名
    valid_values = sorted(valid_companies[metric].dropna(), reverse=True)  # 降序排列
    rank = next((i + 1 for i, val in enumerate(valid_values) if val <= company_value), len(valid_values))
    
    # 计算百分位数
    percentile = (len(valid_values) - rank + 1) / len(valid_values) * 100 if len(valid_values) > 0 else 0
    
    rankings = {
        'metric_name': metric,
        'company_value': float(company_value) if pd.notna(company_value) else 0.0,
        'industry_average': float(industry_average),
        'percentile': round(percentile, 2),
        'rank': rank,
        'total_companies': len(valid_values)
    }
    
    return rankings

def clean_business_description(text):
    """清洗主营业务描述文本"""
    if not text or pd.isna(text):
        return "暂无数据"
    
    cleaned = str(text).strip()
    return cleaned

@st.cache_data
def load_industry_keywords():
    """加载行业关键词数据"""
    df = pd.read_csv('step5_industry_top10_keywords_wide_all.csv')
    return df

@st.cache_data
def load_company_statistics():
    """加载公司统计数据（包含排名信息）"""
    df = pd.read_csv('company_statistics_with_median_percentile_rank.csv')
    return df

@st.cache_data
def load_radar_data():
    """加载综合维度雷达图数据"""
    with open('radars_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

@st.cache_data
def load_indicator_radar_data():
    """加载指标级雷达图数据"""
    with open('indicator_radars_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

@st.cache_data
def load_composite_scores():
    """加载公司综合得分数据"""
    df = pd.read_csv('company_composite_scores_fixed.csv')
    return df

@st.cache_data
def load_listing_dates():
    """加载公司上市日期数据（从IPO_Cobasic.csv）"""
    try:
        df = pd.read_csv('IPO_Cobasic.csv')
        
        # 列名映射，匹配IPO_Cobasic.csv的格式
        column_mapping = {
            'Stkcd': 'stock_code',
            'Stknme': 'company_name',
            'Listdt': 'listing_date'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # 确保股票代码为字符串格式，保留前导零
        if 'stock_code' in df.columns:
            df['stock_code'] = df['stock_code'].astype(str)
            # 如果股票代码被读取为数字（丢失前导零），则补齐6位
            df['stock_code'] = df['stock_code'].apply(
                lambda x: str(int(float(x))).zfill(6) if '.' in x else x.zfill(6)
            )
        
        # 确保上市日期格式正确
        if 'listing_date' in df.columns:
            df['listing_date'] = df['listing_date'].astype(str)
        
        return df
    except Exception as e:
        print(f"加载上市日期文件失败: {e}")
        return pd.DataFrame(columns=['stock_code', 'company_name', 'listing_date'])

def get_company_business_keywords(stock_code):
    """获取公司所属行业的主营业务关键词"""
    # 先获取公司的行业信息
    basic_info = get_company_basic_info(stock_code)
    if not basic_info:
        return None
    
    # 获取行业关键词
    keywords_df = load_industry_keywords()
    
    # 根据行业名称查找关键词
    industry_level1 = basic_info.get('所属行业一级', '')
    industry_level2 = basic_info.get('所属行业二级', '')
    industry_level3 = basic_info.get('所属行业三级', '')
    
    # 优先匹配更细粒度的行业
    for industry_label in [industry_level3, industry_level2, industry_level1]:
        if industry_label and industry_label != '未知':
            # 尝试精确匹配
            matched = keywords_df[keywords_df['industry_label'] == industry_label]
            if not matched.empty:
                return matched.iloc[0]
            # 尝试模糊匹配
            matched = keywords_df[keywords_df['industry_label'].str.contains(industry_label, na=False)]
            if not matched.empty:
                return matched.iloc[0]
    
    return None

def get_company_financial_rankings(stock_code, year=None):
    """
    获取公司的财务指标排名数据
    返回包含ROE、营业利润率等关键指标的排名信息
    """
    stats_df = load_company_statistics()
    
    # 尝试匹配股票代码
    stock_code_str = str(stock_code)
    
    # 先尝试精确匹配
    filtered = stats_df[stats_df['stock_code_norm'].astype(str) == stock_code_str]
    
    # 如果没有匹配，尝试处理可能的格式差异
    if filtered.empty:
        # 尝试移除前导零
        try:
            filtered = stats_df[stats_df['stock_code_norm'].astype(int) == int(stock_code_str)]
        except:
            pass
    
    if filtered.empty:
        return None
    
    # 如果指定年份，筛选对应年份数据
    if year:
        filtered = filtered[filtered['accper'] == year]
    
    if filtered.empty:
        # 返回最新年份数据
        filtered = stats_df[stats_df['stock_code_norm'].astype(str) == stock_code_str]
        if not filtered.empty:
            filtered = filtered[filtered['accper'] == filtered['accper'].max()]
    
    if filtered.empty:
        return None
    
    row = filtered.iloc[0]
    
    # 提取关键指标的排名信息
    rankings = {
        'stock_code': stock_code,
        'company_name': row.get('company_name', ''),
        'year': row.get('accper', ''),
        'roe': {
            'value': row.get('权益资本利润率ROE_value', ''),
            'median': row.get('权益资本利润率ROE_median', ''),
            'percentile': row.get('权益资本利润率ROE_percentile', ''),
            'rank': row.get('权益资本利润率ROE_rank', '')
        },
        'operating_margin': {
            'value': row.get('营业利润率_value', ''),
            'median': row.get('营业利润率_median', ''),
            'percentile': row.get('营业利润率_percentile', ''),
            'rank': row.get('营业利润率_rank', '')
        },
        'roa': {
            'value': row.get('总资产利润率ROA_value', ''),
            'median': row.get('总资产利润率ROA_median', ''),
            'percentile': row.get('总资产利润率ROA_percentile', ''),
            'rank': row.get('总资产利润率ROA_rank', '')
        },
        'ebitda_margin': {
            'value': row.get('EBITDA利润率_value', ''),
            'median': row.get('EBITDA利润率_median', ''),
            'percentile': row.get('EBITDA利润率_percentile', ''),
            'rank': row.get('EBITDA利润率_rank', '')
        },
        'asset_turnover': {
            'value': row.get('总资产周转率_value', ''),
            'median': row.get('总资产周转率_median', ''),
            'percentile': row.get('总资产周转率_percentile', ''),
            'rank': row.get('总资产周转率_rank', '')
        },
        'current_ratio': {
            'value': row.get('流动比率_value', ''),
            'median': row.get('流动比率_median', ''),
            'percentile': row.get('流动比率_percentile', ''),
            'rank': row.get('流动比率_rank', '')
        },
        'debt_ratio': {
            'value': row.get('资产负债率_value', ''),
            'median': row.get('资产负债率_median', ''),
            'percentile': row.get('资产负债率_percentile', ''),
            'rank': row.get('资产负债率_rank', '')
        }
    }
    
    return rankings

def get_company_radar_data(stock_code):
    """
    获取公司的综合维度雷达图数据
    """
    radar_data = load_radar_data()
    
    # 尝试多种格式的股票代码匹配
    stock_code_str = str(stock_code)
    
    # 处理6位股票代码（如 000001）转换为整数格式（如 1）
    try:
        stock_code_int = str(int(float(stock_code_str)))
    except:
        stock_code_int = stock_code_str
    
    # 尝试多种匹配方式
    for key in radar_data.keys():
        # 格式1: company_name_stock_code（完整6位）
        if key.endswith(f'_{stock_code_str}'):
            return radar_data[key]
        # 格式2: company_name_stock_code.0（完整6位加.0）
        if key.endswith(f'_{stock_code_str}.0'):
            return radar_data[key]
        # 格式3: company_name_stock_code_int（去掉前导零的整数）
        if key.endswith(f'_{stock_code_int}'):
            return radar_data[key]
        # 格式4: company_name_stock_code_int.0（去掉前导零的整数加.0）
        if key.endswith(f'_{stock_code_int}.0'):
            return radar_data[key]
        # 通过stock_code字段匹配
        data_stock_code = radar_data[key].get('stock_code', '')
        if str(data_stock_code) == stock_code_str or str(data_stock_code) == stock_code_int:
            return radar_data[key]
        # 尝试将stock_code字段转换为整数后匹配
        try:
            data_stock_code_int = str(int(float(str(data_stock_code))))
            if data_stock_code_int == stock_code_int:
                return radar_data[key]
        except:
            pass
    
    return None

def get_company_indicator_radar_data(stock_code):
    """
    获取公司的指标级雷达图数据
    """
    radar_data = load_indicator_radar_data()
    
    # 尝试多种格式的股票代码匹配
    stock_code_str = str(stock_code)
    
    # 处理6位股票代码（如 000001）转换为整数格式（如 1）
    try:
        stock_code_int = str(int(float(stock_code_str)))
    except:
        stock_code_int = stock_code_str
    
    # 尝试多种匹配方式
    for key in radar_data.keys():
        # 格式1: company_name_stock_code（完整6位）
        if key.endswith(f'_{stock_code_str}'):
            return radar_data[key]
        # 格式2: company_name_stock_code.0（完整6位加.0）
        if key.endswith(f'_{stock_code_str}.0'):
            return radar_data[key]
        # 格式3: company_name_stock_code_int（去掉前导零的整数）
        if key.endswith(f'_{stock_code_int}'):
            return radar_data[key]
        # 格式4: company_name_stock_code_int.0（去掉前导零的整数加.0）
        if key.endswith(f'_{stock_code_int}.0'):
            return radar_data[key]
    
    return None