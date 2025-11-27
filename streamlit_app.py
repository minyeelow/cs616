import os
import streamlit as st
import pandas as pd
import replicate

# Replicate API token input (sidebar)
replicate_token = st.sidebar.text_input('Enter your Replicate API token:', type='password', key='replicate_token')
if replicate_token:
    os.environ['REPLICATE_API_TOKEN'] = replicate_token

# Load anime dataset
@st.cache_data
def load_data():
    df = pd.read_csv("anime-dataset-2023.csv")
    return df

df = load_data()

st.title("[CS616 Assignment 2 - Min Yee] Anime Recommendation App")
st.markdown("Ever wished you had Spotify Rewind and recommendations for Crunchyroll? This app takes the 2023 MyAnimeList database and does exactly that. Simply input your favourite animes, or things you'd like to see and get back your Anime Taste Profile and another anime for you to watch!")

# Search bar

st.header("1. Find Some Animes That You Like")
with st.container():
    search_cols = st.columns([8, 2])
    search_query = search_cols[0].text_input(
        "Search by Anime Name",
        key="search_anime_name",
        placeholder="Type a name or click '🏆' for some award winning suggestions"
    )
    show_more_clicked = search_cols[1].button("🏆")
    search_cols[1].markdown(
        """
        <style>
        .stButton button {
            font-size: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Show possible results as soon as user starts typing (no enter required)
if search_query:
    filtered_df = df[df['Name'].str.contains(search_query, case=False, na=False)]
    st.write(f"Found {len(filtered_df)} anime(s):")
# Show up to 50 random "Award Winning" anime from the dataset
if not search_query:
    if 'award_sample_seed' not in st.session_state:
        st.session_state.award_sample_seed = 42
    if show_more_clicked:
        st.session_state.award_sample_seed += 1

    _award_df = df[df['Genres'].str.contains('Award Winning', case=False, na=False)]
    sample_n = min(9, len(_award_df))
    if sample_n > 0:
        filtered_df = _award_df.sample(n=sample_n, random_state=st.session_state.award_sample_seed)
    else:
        filtered_df = _award_df

# Selection container
# Layout: anime options on left, selected anime on right
cols_main = st.columns([5, 2])
with st.container():
    anime_options = filtered_df['Name'].tolist()
    # Use session state to persist selection
    if 'selected_anime' not in st.session_state:
        st.session_state.selected_anime = []
    selected_anime = st.session_state.selected_anime
    # Display API response if submit is clicked
    if 'show_api_response' not in st.session_state:
        st.session_state.show_api_response = False
    else:
        # Display anime options as a grid of buttons
        if anime_options:
            st.caption("ℹ️ Click to add the anime to your selection")
            cols = st.columns(3)
            for idx, name in enumerate(anime_options):
                col = cols[idx % 3]
                if col.button(f"{name}", key=f"add_{name}"):
                    if name not in st.session_state.selected_anime:
                        st.session_state.selected_anime.append(name)

    st.header("2. Review Your Anime List")

    with st.container():
        st.caption("ℹ️ Your selected anime goes here")
        if selected_anime:
            for idx, name in enumerate(selected_anime):
                row = st.container()
                with row:
                    col1, col2 = st.columns([0.85, 0.15])
                    col1.markdown(f"<span style='font-size:16px;line-height:32px;'>{name}</span>", unsafe_allow_html=True)
                    if col2.button("❌", key=f"delete_{name}"):
                        st.session_state.selected_anime.remove(name)
                        st.experimental_rerun()
        else:
            st.empty()
        # ...existing code...

st.header("3. Finetune Your Recommendation")
additional_prompt = st.text_input(label= '(Optional) Include any special requests!',placeholder= 'Comedy, strong female protagonist etc')

# Move the submit button and its logic here
submit = st.button("Generate my anime preferences and new recommendation!")
if submit and selected_anime:
    st.session_state.show_api_response = True

st.header("4. Here's Your Taste Profile and Recommendation!")

def generate_recommendation(selected_anime, df):
    anime_list = df['Name'].tolist()
    anime_list_str = str(anime_list)
    prompt = (
        f"Describe the user's preferences like Spotify rewind, based on these selections: {', '.join(selected_anime)}. "
        f"Only recommend an anime that can be found in this Python list: {anime_list_str}. "
        f"Ensure that the reccomendation fits the description of {additional_prompt} and is not already in {', '.join(selected_anime)}."
        "The last line of the output must be Recommendation: (anime name)"
    )
    input_payload = {
        "prompt": prompt,
        "system_prompt": "You are a helpful assistant."
    }
    st.markdown("**Common Description:**")
    placeholder = st.empty()
    placeholder.info("Generating your anime preferences and recommendation... Please wait.")
    try:
        output = replicate.run(
            "openai/gpt-4.1-nano",
            input=input_payload
        )
        response_text = "".join(output)
        placeholder.success("Done!")
        st.write(response_text)
        # Extract recommended anime name from output
        import re
        match = re.search(r"(?:recommendation)[^:]*: ([^\n]+)", str(response_text), re.IGNORECASE)
        recommended_name = match.group(1).strip() if match else None

        if recommended_name:
            # Normalize names for robust matching
            def normalize(name):
                return str(name).strip().lower()
            # Try exact match first
            rec_df = df[df['Name'].apply(normalize) == normalize(recommended_name)]
            # If not found, try partial match
            if rec_df.empty:
                rec_df = df[df['Name'].apply(lambda x: normalize(recommended_name) in normalize(x))]
            if not rec_df.empty:
                st.markdown("**Anime Details:**")
                anime_info = rec_df.iloc[0]
                # Display image if available
                if 'Image URL' in anime_info and pd.notna(anime_info['Image URL']):
                    st.image(anime_info['Image URL'], caption=anime_info['Name'])
                for col in rec_df.columns:
                    st.write(f"**{col}:** {anime_info[col]}")
            else:
                st.warning(f"Recommended anime '{recommended_name}' not found in dataset.")
    except Exception as e:
        placeholder.error(f"Replicate API error: {e}")

# Show recommendation after submit
show_api_response = st.session_state.get('show_api_response', False)
if submit and selected_anime:
    st.session_state.show_api_response = True
    show_api_response = True
    generate_recommendation(selected_anime, df)

# Show 'Generate Another Recommendation' button only after first response
if show_api_response and st.button("Generate Another Recommendation!", key="another_recommendation"):
    generate_recommendation(selected_anime, df)
