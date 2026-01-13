import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
import urllib.request
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Movie Recommender Pro", layout="wide")

# --- HÀM TẢI VÀ XỬ LÝ DỮ LIỆU ---
@st.cache_resource
def load_and_train():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    if not os.path.exists("ml"):
        urllib.request.urlretrieve(url, "ml.zip")
        with zipfile.ZipFile("ml.zip", 'r') as zip_ref:
            zip_ref.extractall("ml")

    movies = pd.read_csv("ml/ml-latest-small/movies.csv")
    ratings = pd.read_csv("ml/ml-latest-small/ratings.csv")
    data = pd.merge(ratings, movies, on="movieId")

    # Tạo ma trận User-Movie
    user_movie_matrix = data.pivot_table(index="userId", columns="title", values="rating").fillna(0).astype("float32")

    # Huấn luyện SVD
    svd = TruncatedSVD(n_components=50, random_state=42)
    latent_matrix = svd.fit_transform(user_movie_matrix)
    similarity_matrix = cosine_similarity(latent_matrix)

    # Lấy danh sách thể loại duy nhất
    all_genres = set()
    movies['genres'].str.split('|').apply(lambda x: [all_genres.add(g) for g in x])
    
    return user_movie_matrix, similarity_matrix, movies, sorted(list(all_genres))

user_movie_matrix, similarity_matrix, movies_df, genre_list = load_and_train()

# --- HÀM LẤY ẢNH POSTER ---
def get_poster(movie_title):
    api_key = "8265bd1679663a7ea12ac168da84d2e8"
    clean_title = movie_title.split(' (')[0]
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_title}"
    try:
        data = requests.get(url).json()
        return f"https://image.tmdb.org/t/p/w500/{data['results'][0]['poster_path']}"
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

# --- GIAO DIỆN THANH BÊN (SIDEBAR INPUTS) ---
st.sidebar.header("🔍 Cài đặt gợi ý")

user_id_input = st.sidebar.number_input("Nhập User ID (1-610):", min_value=1, max_value=610, value=1)

# Input mới: Chọn thể loại
selected_genre = st.sidebar.selectbox("Chọn thể loại yêu thích:", ["Tất cả"] + genre_list)

num_recs_input = st.sidebar.slider("Số lượng phim hiển thị:", 4, 16, 8)

predict_button = st.sidebar.button("🚀 Khám phá phim")

# --- HIỂN THỊ CHÍNH ---
st.title("🎬 Movie Recommender Pro")
st.markdown(f"Hệ thống đang phân tích sở thích của **Người dùng {user_id_input}**...")

if predict_button:
    # 1. Thuật toán Collaborative Filtering (SVD)
    user_index = int(user_id_input) - 1
    similarity_scores = similarity_matrix[user_index]
    similar_users = np.argsort(similarity_scores)[::-1][1:11]
    
    recommended_series = (
        user_movie_matrix.iloc[similar_users]
        .mean(axis=0)
        .sort_values(ascending=False)
    )

    already_watched = user_movie_matrix.iloc[user_index]
    recs_all = recommended_series[already_watched == 0]

    # 2. Lọc theo thể loại nếu người dùng yêu cầu
    final_recs = []
    if selected_genre != "Tất cả":
        for title, score in recs_all.items():
            # Kiểm tra thể loại của phim trong movies_df
            movie_info = movies_df[movies_df['title'] == title]
            if not movie_info.empty and selected_genre in movie_info.iloc[0]['genres']:
                final_recs.append((title, score))
            if len(final_recs) >= num_recs_input:
                break
    else:
        final_recs = list(recs_all.head(num_recs_input).items())

    # 3. Hiển thị kết quả
    if not final_recs:
        st.warning(f"Rất tiếc, không tìm thấy phim nào thuộc thể loại '{selected_genre}' phù hợp với bạn.")
    else:
        cols = st.columns(4)
        for i, (title, score) in enumerate(final_recs):
            with cols[i % 4]:
                st.image(get_poster(title), use_container_width=True)
                st.markdown(f"**{title}**")
                # Lấy thể loại để hiển thị tag
                g = movies_df[movies_df['title'] == title].iloc[0]['genres'].replace('|', ', ')
                st.caption(f"🎭 {g}")
                st.caption(f"⭐ Độ phù hợp: {score:.1f}")
else:
    st.info("Nhấn 'Khám phá phim' để nhận danh sách đề xuất cá nhân hóa!")
