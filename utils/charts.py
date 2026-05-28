import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import streamlit as st

def create_radar_chart(categories, values, title="雷达图"):
    """
    创建雷达图
    categories: 维度名称列表
    values: 对应的数值列表
    """
    fig = go.Figure()
    
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name=title,
        line_color='rgb(0,100,200)',
        fillcolor='rgba(0,100,200,0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min(values)*0.8, max(values)*1.2]
            )
        ),
        showlegend=True,
        title=title
    )
    
    return fig

def create_line_chart(df, x_col, y_cols, title="趋势图", x_label=None, y_label=None):
    """
    创建折线图
    df: DataFrame
    x_col: X轴列名
    y_cols: Y轴列名列表
    """
    fig = go.Figure()
    
    for col in y_cols:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[col],
            name=col,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label or x_col,
        yaxis_title=y_label or '数值',
        hovermode='x unified',
        legend_title='指标'
    )
    
    return fig

def create_wordcloud_from_text(text, title="词云图"):
    """
    从文本生成词云图
    返回matplotlib图形对象
    """
    if not text or len(str(text).strip()) < 10:
        return None
    
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            font_path='simhei.ttf',
            max_words=100,
            colormap='viridis'
        ).generate(str(text))
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"生成词云失败: {str(e)}")
        return None

def create_bar_chart(categories, values, title="柱状图", color='steelblue'):
    """
    创建柱状图
    """
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=color
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title='类别',
        yaxis_title='数值',
        showlegend=False
    )
    
    return fig

def create_comparison_chart(df, x_col, y_cols, company_code, title="对比分析"):
    """
    创建公司与行业平均的对比图
    """
    fig = go.Figure()
    
    for col in y_cols:
        if col in df.columns:
            fig.add_trace(go.Bar(
                x=[f'{company_code}', '行业平均'],
                y=[df[col].iloc[0], df[f'{col}_industry_avg'].iloc[0] if f'{col}_industry_avg' in df.columns else 0],
                name=col
            ))
    
    fig.update_layout(
        title=title,
        barmode='group',
        xaxis_title='主体',
        yaxis_title='数值'
    )
    
    return fig