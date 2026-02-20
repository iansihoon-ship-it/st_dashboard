import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. 페이지 설정 (Wide Mode)
st.set_page_config(page_title="Naver API 유형별 통합 분석 대시보드 v3", layout="wide")

# 2. 데이터 로드 및 전처리 (캐싱 처리)
@st.cache_data
def load_and_preprocess_data():
    data_dir = 'data'
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    blog_list, shop_list, trend_list, news_list = [], [], [], []
    
    for f in files:
        filename = os.path.basename(f)
        
        # 키워드 추출 로직 (v3: rsplit을 사용하여 날짜 접미사 제거)
        keyword = ""
        if 'blog_' in filename:
            keyword = filename.replace('blog_', '').rsplit('_', 1)[0]
        elif 'shopping_trend_' in filename:
            keyword = filename.replace('shopping_trend_', '').rsplit('_', 1)[0]
        elif 'shop_' in filename:
            keyword = filename.replace('shop_', '').rsplit('_', 1)[0]
        elif 'news_' in filename:
            keyword = filename.replace('news_', '').rsplit('_', 1)[0]
        else:
            continue
            
        try:
            df = pd.read_csv(f)
            df['target_keyword'] = keyword
            
            if 'blog_' in filename:
                df['postdate'] = pd.to_datetime(df['postdate'], format='%Y%m%d', errors='coerce')
                blog_list.append(df)
            elif 'shopping_trend' in filename:
                df['period'] = pd.to_datetime(df['period'], errors='coerce')
                trend_list.append(df)
            elif 'shop_' in filename:
                shop_list.append(df)
            elif 'news_' in filename:
                df['pubDate'] = pd.to_datetime(df['pubDate'], errors='coerce')
                news_list.append(df)
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            
    return (pd.concat(blog_list, ignore_index=True) if blog_list else pd.DataFrame(),
            pd.concat(shop_list, ignore_index=True) if shop_list else pd.DataFrame(),
            pd.concat(trend_list, ignore_index=True) if trend_list else pd.DataFrame(),
            pd.concat(news_list, ignore_index=True) if news_list else pd.DataFrame())

# 데이터 로딩
blog_df, shop_df, trend_df, news_df = load_and_preprocess_data()

# 3. 사이드바 (Sidebar Filters)
st.sidebar.title("🔍 분석 설정")
all_keywords = sorted(trend_df['target_keyword'].unique().tolist()) if not trend_df.empty else []
# 기본적으로 처음 3개 키워드 선택 (너무 많으면 그래프가 복잡함)
selected_keywords = st.sidebar.multiselect("분석 키워드 선택", all_keywords, default=all_keywords[:3] if len(all_keywords) > 3 else all_keywords)

if not trend_df.empty:
    min_date = trend_df['period'].min()
    max_date = trend_df['period'].max()
    date_range = st.sidebar.date_input("분석 기간", [min_date, max_date], min_value=min_date, max_value=max_date)
else:
    date_range = []

# 데이터 필터링 로직
if len(date_range) == 2:
    start_date, end_date = date_range
    f_trend = trend_df[(trend_df['target_keyword'].isin(selected_keywords)) & 
                       (trend_df['period'].dt.date >= start_date) & 
                       (trend_df['period'].dt.date <= end_date)]
    f_blog = blog_df[(blog_df['target_keyword'].isin(selected_keywords)) & 
                     (blog_df['postdate'].dt.date >= start_date) & 
                     (blog_df['postdate'].dt.date <= end_date)]
    f_news = news_df[news_df['target_keyword'].isin(selected_keywords)]
    f_shop = shop_df[shop_df['target_keyword'].isin(selected_keywords)]
else:
    f_trend, f_blog, f_shop, f_news = trend_df, blog_df, shop_df, news_df

# 4. 메인 화면 구성
st.title("🛡️ Naver API 유형별 통합 분석 대시보드 v3")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📉 트렌드 탐색기", "✍️ 블로그 인사이트", "🛒 쇼핑 마켓 분석", "📰 뉴스 분석"])

# --- 탭 1: 트렌드 분석 ---
with tab1:
    st.header("1. 유형별 검색 트렌드 비교")
    if not f_trend.empty:
        fig1 = px.line(f_trend, x='period', y='ratio', color='target_keyword',
                       title="선글라스 유형별 검색 비율 추이", labels={'ratio': '클릭 지수', 'period': '날짜'},
                       template="plotly_white")
        st.plotly_chart(fig1, key='fig1_trend', width='stretch')
        
        st.subheader("표 1: 키워드별 기술 통계 요약")
        trend_desc = f_trend.groupby('target_keyword')['ratio'].agg(['mean', 'std', 'min', 'max']).reset_index()
        st.dataframe(trend_desc, width='stretch')
    else:
        st.info("비교할 키워드를 선택해 주세요.")

# --- 탭 2: 블로그 분석 ---
with tab2:
    st.header("2. 블로그 인사이트 및 키워드 분석")
    if not f_blog.empty:
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.subheader("그래프 2: 포스팅 빈도 상위 블로거")
            blogger_top = f_blog['bloggername'].value_counts().head(20).reset_index()
            fig2 = px.bar(blogger_top, x='count', y='bloggername', orientation='h', 
                          title="포스팅 빈도 상위 블로거", color='count')
            fig2.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig2, key='fig2_blog', width='stretch')
            
        with col_b2:
            st.subheader("그래프 3: 핵심 키워드 트리맵 (TF-IDF)")
            try:
                tfidf_vec = TfidfVectorizer(max_features=30)
                tfidf_mat = tfidf_vec.fit_transform(f_blog['description'].fillna(''))
                ranking = pd.DataFrame({'keyword': tfidf_vec.get_feature_names_out(), 'score': np.asarray(tfidf_mat.sum(axis=0)).flatten()})
                fig3 = px.treemap(ranking, path=['keyword'], values='score', color='score', 
                                  title="블로그 데이터 핵심 키워드")
                st.plotly_chart(fig3, key='fig3_tfidf', width='stretch')
            except:
                st.write("키워드 분석을 위한 충분한 데이터가 없습니다.")

        st.markdown("---")
        st.subheader("표 2: 최신 블로그 포스팅 목록 (20건)")
        st.dataframe(f_blog[['postdate', 'bloggername', 'title', 'target_keyword']].sort_values('postdate', ascending=False).head(20), width='stretch')
    else:
        st.info("데이터가 없습니다.")

# --- 탭 3: 쇼핑 분석 ---
with tab3:
    st.header("3. 마켓플레이스 및 브랜드 분석")
    if not f_shop.empty:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.subheader("그래프 4: 유형별 가격 분포 히스토그램")
            fig4 = px.histogram(f_shop, x='lprice', color='target_keyword', barmode='overlay', 
                                title="가격대별 상품 분포 (Overlaid)", marginal="rug")
            st.plotly_chart(fig4, key='fig4_shop', width='stretch')
        with col_s2:
            st.subheader("그래프 5: 주요 브랜드별 가격 범위")
            top_brands = f_shop['brand'].value_counts().head(10).index
            f_brand = f_shop[f_shop['brand'].isin(top_brands)]
            fig5 = px.box(f_brand, x='brand', y='lprice', color='target_keyword', 
                          title="상위 10개 브랜드 가격 편차")
            st.plotly_chart(fig5, key='fig5_shop_box', width='stretch')
            
        st.markdown("---")
        st.subheader("표 3: 브랜드별 마켓 지표 요약")
        brand_summary = f_shop.groupby(['brand', 'target_keyword'])['lprice'].agg(['mean', 'min', 'max', 'count']).reset_index()
        st.dataframe(brand_summary.sort_values('count', ascending=False).head(50), width='stretch')
    else:
        st.info("데이터가 없습니다.")

# --- 탭 4: 뉴스 분석 ---
with tab4:
    st.header("4. 최신 뉴스 트렌드 분석")
    if not f_news.empty:
        col_n1, col_n2 = st.columns([1, 2])
        with col_n1:
            st.subheader("뉴스 키워드 중요도")
            try:
                tfidf_vec_n = TfidfVectorizer(max_features=25)
                tfidf_mat_n = tfidf_vec_n.fit_transform(f_news['title'].fillna(''))
                ranking_n = pd.DataFrame({'keyword': tfidf_vec_n.get_feature_names_out(), 'score': np.asarray(tfidf_mat_n.sum(axis=0)).flatten()})
                fig6 = px.bar(ranking_n.sort_values('score', ascending=True), x='score', y='keyword', orientation='h', 
                              title="뉴스 헤드라인 주요 키워드", color='score')
                st.plotly_chart(fig6, key='fig6_news', width='stretch')
            except:
                st.write("키워드 분석 데이터 부족")
        with col_n2:
            st.subheader("표 4: 최신 뉴스 헤드라인 목록")
            st.dataframe(f_news[['target_keyword', 'title', 'pubDate']].sort_values('pubDate', ascending=False).head(30), width='stretch')
    else:
        st.info("뉴스 데이터가 없습니다.")

st.markdown("---")
st.caption("Naver API Multi-Keyword Analysis Dashboard v3 - Created by Antigravity")
