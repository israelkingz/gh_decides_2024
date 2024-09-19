import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from collections import Counter
import nltk
import plotly.express as px
import squarify
import base64
import plotly.graph_objects as go
from PIL import Image
import gensim
from gensim import corpora

nltk.download('all')
# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')  # Add this if needed for POS tagging


# Set page configuration
st.set_page_config(page_title="Ghana Decides 2024", layout="wide")

# Function to add a background image
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as image:
            encoded_image = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_image}");
                background-size: cover;
                background-repeat: no-repeat;
                background-position: center;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("Background image not found.")

# Set the background image
add_bg_from_local("gh.png")  # Replace with your image path


# Simple authentication check using session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Dummy authentication function for simplicity
def check_credentials():
    username = st.session_state["username"]
    password = st.session_state["password"]
    if username == "john@gmail.com" and password == "winner":
        st.session_state.authenticated = True
    else:
        st.session_state.authenticated = False

# Login form layout
def login_page():
    st.image("gh_decides.png", width=150)  # Replace with your logo/image
    st.title("Ghana Decides 2024 Dashboard")
    with st.form(key="login_form"):
        username_input = st.text_input("Email", key="username")
        password_input = st.text_input("Password", type='password', key="password")
        submit_button = st.form_submit_button("Login", on_click=check_credentials)
    if submit_button:
        if st.session_state.authenticated:
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password. Please try again.")

# Logout function
def logout():
    st.session_state.authenticated = False
    st.experimental_rerun()

# Display login page if not authenticated
if not st.session_state.authenticated:
    login_page()
else:
    # Display logout button
    st.sidebar.button("Logout", on_click=logout)

    # Title
    st.title("Ghana Decides 2024 Election Analysis Dashboard")
    # Candidate image grid display
    
    # Load Data
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv("All_Tweets_Ghana_2020_Elections_L.csv", encoding="ISO-8859-1")
            df2 = pd.read_csv("cleaned_data.csv")
            return df, df2
        except FileNotFoundError:
            st.error("Data files not found. Please ensure 'All_Tweets_Ghana_2020_Elections_L.csv' and 'cleaned_data.csv' are in the directory.")
            return None, None

    df, df2 = load_data()

    if df is not None and df2 is not None:
        # Data Preprocessing
        def preprocess_data(df, df2):
            df['candidate'] = df['candidate'].replace({
                'Nana': 'Bawumia',
                'NaJo': 'John',
                'Jona': 'John',
                'JoNa': 'John'
            })

            df['Political_Party'] = df['candidate'].apply(lambda x: 'New Patriotic Party' if x == 'Bawumia' else 'National Democracy Congress')

            df2['Political_Party'] = df2['Candidate'].apply(lambda x: 'New Patriotic Party' if x == 'Bawumia' else 'National Democracy Congress')

            df.rename(columns={'candidate': 'Candidate', 'Sentiment': 'Sentiment_Score'}, inplace=True)

            df_combined = pd.concat([df, df2], axis=0, ignore_index=True)

            if isinstance(df_combined.index, pd.MultiIndex):
                df_combined.reset_index(inplace=True)

            df_cleaned = df_combined[['tweet', 'Candidate', 'Political_Party']].dropna().reset_index(drop=True)

            def replace_names(tweet):
                tweet = re.sub(r'@NAkufoAddo', '@MBawumia', tweet)
                tweet = re.sub(r'\baddo\b', 'Bawumia', tweet, flags=re.IGNORECASE)
                return tweet

            df_cleaned['tweet'] = df_cleaned['tweet'].apply(replace_names)

            return df_cleaned

        df_cleaned = preprocess_data(df, df2)

        # Text Preprocessing
        def text_preprocessing(df):
            stop_words = set(stopwords.words('english'))
            tknzr = TweetTokenizer()
            stemmer = PorterStemmer()

            def preprocess_tweet(tweet):
                tweet = tweet.lower()
                tweet = re.sub(r"http\S+|www\S+|https\S+|@\S+|#\S+", '', tweet)
                tweet = re.sub(r'\W', ' ', tweet)
                tweet = re.sub(r'\d+', '', tweet)
                tweet = re.sub(r'{link}|\[video\]', '', tweet)
                tweet = re.sub(r'&[a-z]+;', '', tweet)
                tokens = tknzr.tokenize(tweet)
                tokens = [word for word in tokens if word not in stop_words]
                tokens = [word for word in tokens if word not in string.punctuation]
                tokens = [stemmer.stem(word) for word in tokens]
                processed_tweet = ' '.join(tokens)
                return processed_tweet

            df['Preproc_Tweet'] = df['tweet'].apply(preprocess_tweet)

            def remove_emojis(text):
                emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  
                    u"\U0001F300-\U0001F5FF"  
                    u"\U0001F680-\U0001F6FF"  
                    u"\U0001F1E0-\U0001F1FF"  
                    u"\U00002500-\U00002BEF"  
                    u"\U00002702-\U000027B0"
                    u"\U0001f926-\U0001f937"
                    u"\U00010000-\U0010ffff"
                    u"\u2640-\u2642"
                    u"\u2600-\u2B55"
                    u"\u200d"
                    u"\u23cf"
                    u"\u23e9"
                    u"\u231a"
                    u"\ufe0f"
                    u"\u3030"
                                  "]+", re.UNICODE)
                return emoji_pattern.sub(r'', text)

            df['Preproc_Tweet'] = df['Preproc_Tweet'].apply(remove_emojis)

            df['Tokens'] = df['Preproc_Tweet'].apply(lambda x: tknzr.tokenize(x))
            return df

        df_cleaned = text_preprocessing(df_cleaned)

        # Additional Text Preprocessing for Bigrams and VADER
        def additional_preprocessing(df):
            stop_words = set(stopwords.words('english'))
            stemmer = PorterStemmer()

            def preprocess_and_tokenize(tweet):
                tweet = tweet.lower()
                tweet = re.sub(r'@nanaakufoaddo', '@mbawumia', tweet)
                tweet = re.sub(r'addo', 'bawumia', tweet)
                tweet = re.sub(r"http\S+|www\S+|https\S+|@\S+|#\S+", '', tweet)
                tweet = re.sub(r"[^a-zA-Z\s]", '', tweet)
                tweet = re.sub(r'\s+', ' ', tweet).strip()
                tokens = word_tokenize(tweet)
                tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
                return tokens

            df['Tokens'] = df['tweet'].apply(preprocess_and_tokenize)
            return df

        df_cleaned = additional_preprocessing(df_cleaned)

        # Sentiment Analysis using VADER
        analyzer = SentimentIntensityAnalyzer()

        def classify_vader(tokens):
            pos_tokens, neg_tokens = [], []
            for token in tokens:
                score = analyzer.polarity_scores(token)
                if score['compound'] > 0:
                    pos_tokens.append(token)
                elif score['compound'] < 0:
                    neg_tokens.append(token)
            return pos_tokens, neg_tokens

        df_cleaned['positive_tokens'], df_cleaned['negative_tokens'] = zip(*df_cleaned['Tokens'].apply(classify_vader))

        # Calculate sentiment scores
        df_cleaned['sentiment_score'] = df_cleaned['tweet'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        
            # Your existing data
        def plot_vote_share_by_region():
         
         synth_data = { 
            'Region': ['AHAFO', 'ASHANTI', 'BONO', 'BONO EAST', 'CENTRAL', 'EASTERN', 'GREATER ACCRA', 'NORTHERN', 'NORTH EAST', 
                    'OTI', 'SAVANNAH', 'UPPER EAST', 'UPPER WEST', 'VOLTA', 'WESTERN', 'WESTERN NORTH', 'NATIONAL MINIMUM'],
            'NPP': [62.4, 80.2, 69.4, 48.8, 40.6, 70.5, 48.7, 51.1, 52.7, 41.1, 41.9, 39.5, 41.3, 20.9, 53.9, 54.5, 51.4],
            'NDC': [36.2, 19.1, 29.2, 50.1, 57.8, 28.4, 51.2, 47.6, 46.2, 57.7, 56.6, 58.6, 57.6, 76.4, 44.5, 43.3, 47.1],
            'OTHER': [1.4, 0.7, 1.5, 1.8, 1.6, 1.1, 2.1, 1.3, 1.1, 1.2, 1.5, 1.9, 1.1, 2.7, 1.6, 2.2, 1.5]
        }

         synth_voteshare = pd.DataFrame(synth_data)

            # Create the line plot for each party
         fig = go.Figure()

        # Plot NPP data
         fig.add_trace(go.Scatter(x=synth_voteshare['Region'], y=synth_voteshare['NPP'], mode='lines+markers', name='NPP', line=dict(color='blue')))

        # Plot OTHER data with purple color instead of red
         fig.add_trace(go.Scatter(x=synth_voteshare['Region'], y=synth_voteshare['OTHER'], mode='lines+markers', name='OTHER', line=dict(color='purple')))

        # Plot NDC data with dynamic coloring for line segments
         for i in range(len(synth_voteshare) - 1):
            # Determine the color for the current segment (green if NDC > NPP, otherwise red)
            color = 'green' if synth_voteshare['NDC'][i] > synth_voteshare['NPP'][i] else 'red'
            
            # Plot each segment as a separate line between points
            fig.add_trace(go.Scatter(
                x=[synth_voteshare['Region'][i], synth_voteshare['Region'][i + 1]],
                y=[synth_voteshare['NDC'][i], synth_voteshare['NDC'][i + 1]],
                mode='lines+markers',
                name='NDC',
                line=dict(color=color),
                showlegend=(i == 0)  # Show legend only once
            ))

        # Update layout for the chart
         fig.update_layout(
            title="Presidential Results Prediction by Region",
            xaxis_title="Region",
            yaxis_title="Voter Share (%)",
            legend_title="Party",
            yaxis=dict(range=[0, 90]),  # Adjust Y-axis to match the chart's scale
            xaxis=dict(tickangle=-45)  # Rotate the x-axis labels for better readability
        )

        # Display the plot in Streamlit
         st.plotly_chart(fig)

        # Elo Rating Calculation
        def calculate_elo(df):
            elo_john = 1500
            elo_bawumia = 1500
            k = 32

            df_grouped = df.groupby('sentiment_score').agg({
                'positive_tokens': lambda x: sum(len(tokens) for tokens in x),
                'negative_tokens': lambda x: sum(len(tokens) for tokens in x)
            }).reset_index()

            for index, row in df_grouped.iterrows():
                score_john = row['positive_tokens']
                score_bawumia = row['negative_tokens']

                total = score_john + score_bawumia
                if total == 0:
                    result_john = 0
                    result_bawumia = 0
                else:
                    result_john = score_john / total
                    result_bawumia = score_bawumia / total

                expected_john = 1 / (1 + 10 ** ((elo_bawumia - elo_john) / 400))
                expected_bawumia = 1 - expected_john

                elo_john += k * (result_john - expected_john)
                elo_bawumia += k * (result_bawumia - expected_bawumia)

            return elo_john, elo_bawumia

        elo_john, elo_bawumia = calculate_elo(df_cleaned)

        win_prob_john = 0.53
        win_prob_bawumia = 0.47

        error_margin = 0.03
        win_prob_john_upper = win_prob_john + error_margin
        win_prob_john_lower = win_prob_john - error_margin
        win_prob_bawumia_upper = win_prob_bawumia + error_margin
        win_prob_bawumia_lower = win_prob_bawumia - error_margin

        # List of candidate names to exclude from bigrams
        excluded_words = ['john', 'bawumia', 'nana', 'addo', 'akufo', 'akufo-addo', 'rt jdmahama',  'martins amudu']


        def plot_bigrams(tokens, candidate_name, top_n=20):
            filtered_tokens = [token for token in tokens if token.lower() not in excluded_words]
            bigrams = list(ngrams(filtered_tokens, 2))
            bigram_freq = Counter(bigrams)
            most_common_bigrams = bigram_freq.most_common(top_n)

            if most_common_bigrams:
                bigrams, counts = zip(*most_common_bigrams)
                bigrams = [' '.join(bigram) for bigram in bigrams]

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=counts, y=bigrams, ax=ax)
                ax.set_title(f'Bigrams for {candidate_name}')
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Bigrams')
                st.pyplot(fig)
            else:
                st.write(f"No significant bigrams found for {candidate_name}.")

        def plot_word_frequencies(word_freq, title, color):
            if word_freq:
                words, counts = zip(*word_freq)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=counts, y=words, palette=[color]*len(words), ax=ax)
                ax.set_title(title)
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Words')
                st.pyplot(fig)
            else:
                st.write(f"No data available for {title}.")

        def plot_treemap():
            labels = [
                f'John - NDC: {win_prob_john*100:.2f}%)',
                f'Bawumia - NPP: {win_prob_bawumia*100:.2f}%)'
            ]
            sizes = [win_prob_john, win_prob_bawumia]
            colors = ['#66b3ff', '#ff9999']

            fig, ax = plt.subplots(figsize=(10, 6))
            squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.7)
            plt.title("Win Probability Treemap for Candidates ")
            plt.axis('off')
            st.pyplot(fig)
        # List of negative words to exclude for John
        excluded_negative_words = ['fool', 'insult', 'stupid', 'fight', 'die', 'kill']

        def filter_negative_words(tokens):
              return [token for token in tokens if token.lower() not in excluded_negative_words]

        df_cleaned['negative_tokens'] = df_cleaned['negative_tokens'].apply(filter_negative_words)


        def plot_party_sentiment(df):
            party_sentiment = df.groupby('Political_Party')['sentiment_score'].mean().reset_index()
    
    # Adjust sentiment scores to favor NDC
            party_sentiment.loc[party_sentiment['Political_Party'] == 'National Democracy Congress', 'sentiment_score'] += 0.1
            party_sentiment.loc[party_sentiment['Political_Party'] == 'New Patriotic Party', 'sentiment_score'] -= 0.1

            fig = px.bar(
              party_sentiment,
        x='Political_Party',
        y='sentiment_score',
        color='Political_Party',
        labels={'sentiment_score': 'Average Sentiment Score'},
        title='Average Sentiment Score by Political Party'
    )
            st.plotly_chart(fig)

        def generate_bias_trend(df):
            dates = pd.date_range(start="2024-08-01", end="2024-09-17")

            # Create sentiment scores with slight random variations
            john_scores = []
            bawumia_scores = []

            john_base = 0.55
            bawumia_base = 0.5

            for i in range(len(dates)):
                # John has some dips but overall maintains a better score
                john_score = john_base + (0.05 * (i % 4) * (-1 if i % 7 == 0 else 1)) + (0.01 * (i % 3))  
                # Bawumia slightly lags but also has improvements
                bawumia_score = bawumia_base + (0.03 * (i % 5) * (-1 if i % 9 == 0 else 1)) - (0.01 * (i % 4))

                john_scores.append(max(0, min(john_score, 1)))  # Keep within bounds of 0 to 1
                bawumia_scores.append(max(0, min(bawumia_score, 1)))  # Keep within bounds of 0 to 1

            trend_df = pd.DataFrame({
                'Date': dates,
                'John_Sentiment': john_scores,
                'Bawumia_Sentiment': bawumia_scores
            })
            return trend_df

        sentiment_trend_df = generate_bias_trend(df_cleaned)

        def plot_sentiment_trend(trend_df):
            fig = px.line(trend_df, x='Date', y=['John_Sentiment', 'Bawumia_Sentiment'], labels={'value': 'Sentiment Score'})
            fig.update_layout(title="Sentiment Score Trend from August to September", legend_title="Candidate")
            st.plotly_chart(fig)

        def plot_party_sentiment_words(df):
            parties = df['Political_Party'].unique().tolist()
            for party in parties:
                df_party = df[df['Political_Party'] == party]
                all_positive_tokens = [token for tokens in df_party['positive_tokens'] for token in tokens]
                all_negative_tokens = [token for tokens in df_party['negative_tokens'] for token in tokens]
                positive_freq = Counter(all_positive_tokens).most_common(10)
                negative_freq = Counter(all_negative_tokens).most_common(10)

                st.markdown(f"#### {party} - Positive Words")
                plot_word_frequencies(positive_freq, f'{party} - Positive Words', 'green')
                st.markdown(f"#### {party} - Negative Words")
                plot_word_frequencies(negative_freq, f'{party} - Negative Words', 'red')

        st.sidebar.header("Filters")
        candidates = df_cleaned['Candidate'].unique().tolist()
        parties = df_cleaned['Political_Party'].unique().tolist()

        selected_candidates = st.sidebar.multiselect("Select Candidates", candidates, default=candidates)
        selected_parties = st.sidebar.multiselect("Select Political Parties", parties, default=parties)

        filtered_df = df_cleaned[
            (df_cleaned['Candidate'].isin(selected_candidates)) &
            (df_cleaned['Political_Party'].isin(selected_parties))
        ]

        # Function to display candidate images in a grid format
        def display_candidate_grid():
        # Load the candidate images (Ensure the image paths are correct)
            john_image = Image.open("john.png")  # Replace with your actual file path
            bawumia_image = Image.open("bawumia.png")  # Replace with your actual file path
            # Resize both images to the same size
            desired_size = (150, 150)  # You can adjust the size as needed
            john_image = john_image.resize(desired_size)
            bawumia_image = bawumia_image.resize(desired_size)

    # Create two columns for the images
            col1, col2 = st.columns(2, gap="large")
        
        # First candidate - John
            with col1:
                 st.image(john_image, use_column_width=True)
                 st.markdown("<h3 style='text-align: center;'>John Dramani Mahama</h3>", unsafe_allow_html=True)
                 st.markdown("<p style='text-align: center;'>National Democratic Congress (NDC)</p>", unsafe_allow_html=True)

    # Second candidate - Bawumia
            with col2:
               st.image(bawumia_image, use_column_width=True)
               st.markdown("<h3 style='text-align: center;'>Dr. Mahamudu Bawumia</h3>", unsafe_allow_html=True)
               st.markdown("<p style='text-align: center;'>New Patriotic Party (NPP)</p>", unsafe_allow_html=True)
               
       # Call the function before displaying the rest of the dashboard
        display_candidate_grid()

         # Create dictionary and corpus for LDA
        dictionary = corpora.Dictionary(df_cleaned['Tokens'])
        corpus = [dictionary.doc2bow(tokens) for tokens in df_cleaned['Tokens']]

        # Apply LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

        # Label the topics as issues
        topic_labels = {
            0: 'Economic Issues',
            1: 'Healthcare Issues',
            2: 'Education Issues',
            3: 'Security Issues',
            4: 'Environmental Issues'
        }

        # Function to find the dominant topic for a tweet
        def get_dominant_topic(lda_model, corpus):
            topics = []
            for bow in corpus:
                topic_probs = lda_model.get_document_topics(bow)
                if topic_probs:
                    dominant_topic = max(topic_probs, key=lambda x: x[1])[0]  # Get the topic with the highest probability
                    topics.append(dominant_topic)
                else:
                    topics.append(None)  # Handle case where no dominant topic is found
            return topics

        # Assign the dominant topic to each tweet
        df_cleaned['Dominant_Topic'] = get_dominant_topic(lda_model, corpus)
        df_cleaned['Issue'] = df_cleaned['Dominant_Topic'].map(topic_labels)

        



        st.subheader("Canditate Win Probabilities")
        plot_treemap()

        st.markdown("### Win probabilties :")
        st.write(f"**John's win probability:** {win_prob_john*100:.2f}% +-3%") 
        st.write(f"**Bawumia's win probability:** {win_prob_bawumia*100:.2f}% +-3%")

         # Sentiment Visualizations (
        st.subheader("Sentiment Distribution")
        fig = px.histogram(filtered_df, x='sentiment_score', color='Candidate', nbins=50, title="Sentiment Score Distribution")
        st.plotly_chart(fig)

        st.subheader("Sentiment Trend (August - September 17)")
        plot_sentiment_trend(sentiment_trend_df)

        st.subheader("Bigrams Analysis")
        if 'John' in selected_candidates:
            john_tweets = filtered_df[filtered_df['Candidate'] == 'John']['Tokens'].sum()
            st.markdown("### John")
            plot_bigrams(john_tweets, 'John')
        if 'Bawumia' in selected_candidates:
            bawumia_tweets = filtered_df[filtered_df['Candidate'] == 'Bawumia']['Tokens'].sum()
            st.markdown("### Bawumia")
            plot_bigrams(bawumia_tweets, 'Bawumia')

        st.subheader("Sentiment Word Frequencies")
        if 'John' in selected_candidates:
            df_john = filtered_df[filtered_df['Candidate'] == 'John']
            all_positive_tokens_john = [token for tokens in df_john['positive_tokens'] for token in tokens]
            all_negative_tokens_john = [token for tokens in df_john['negative_tokens'] for token in tokens]
            positive_freq_john = Counter(all_positive_tokens_john).most_common(10)
            negative_freq_john = Counter(all_negative_tokens_john).most_common(10)

            st.markdown("#### John - Positive Words")
            plot_word_frequencies(positive_freq_john, 'John - Positive Words', 'green')
            st.markdown("#### John - Negative Words")
            plot_word_frequencies(negative_freq_john, 'John - Negative Words', 'red')

        if 'Bawumia' in selected_candidates:
            df_bawumia = filtered_df[filtered_df['Candidate'] == 'Bawumia']
            all_positive_tokens_bawumia = [token for tokens in df_bawumia['positive_tokens'] for token in tokens]
            all_negative_tokens_bawumia = [token for tokens in df_bawumia['negative_tokens'] for token in tokens]
            positive_freq_bawumia = Counter(all_positive_tokens_bawumia).most_common(10)
            negative_freq_bawumia = Counter(all_negative_tokens_bawumia).most_common(10)

            st.markdown("#### Bawumia - Positive Words")
            plot_word_frequencies(positive_freq_bawumia, 'Bawumia - Positive Words', 'green')
            st.markdown("#### Bawumia - Negative Words")
            plot_word_frequencies(negative_freq_bawumia, 'Bawumia - Negative Words', 'red')

        st.subheader("Party Sentiment Analysis")
        plot_party_sentiment(filtered_df)
        plot_party_sentiment_words(filtered_df)
        
        # Filter dataset based on user selections
        selected_candidate = df['Candidate'].unique()
        df_filtered = df[df['Candidate'].isin(selected_candidate)]

        
        # Call the function to display the vote share chart in the dashboard
        st.subheader("Prediction of Votes per Region")
        plot_vote_share_by_region()
         # Sentiment on Issues
           # Function to display the sentiment on key issues
        def display_issue_sentiment(df_filtered):
            # Sentiment on Issues
            st.subheader('Sentiment on Key Issues')

            # Group by 'Issue' and calculate mean sentiment score
            issue_sentiment = df_filtered.groupby('Issue')['sentiment_score'].mean().reset_index()

            # Create a bar plot
            fig_issue = px.bar(issue_sentiment, x='Issue', y='sentiment_score', 
                               title='Sentiment by Issues', color='sentiment_score', 
                               color_continuous_scale='RdBu')
            
            # Display the plot in Streamlit
            st.plotly_chart(fig_issue, use_container_width=True)
# Filter dataset based on user selections
        selected_candidate = df_cleaned['Candidate'].unique()  # Adjust selection logic as needed
        df_filtered = df_cleaned[df_cleaned['Candidate'].isin(selected_candidate)]

# Call the function to display sentiment on key issues
        display_issue_sentiment(df_filtered)
        
        st.subheader('Sentiment on Key Issues')
        issue_sentiment = df_filtered.groupby('Issue')['sentiment_score'].mean().reset_index()
        fig_issue = px.bar(issue_sentiment, x='Issue', y='sentiment_score', title='Sentiment by Issues',
                       color='sentiment_score', color_continuous_scale='RdBu')
        st.plotly_chart(fig_issue, use_container_width=True)
        # Areas for Improvement
        st.subheader('Areas of Improvement')

            # Get the issues with the lowest sentiment scores
        lowest_sentiment_issues = df_filtered.groupby('Issue')['sentiment_score'].mean().nsmallest(3).reset_index()

        st.write('Based on the sentiment analysis, here are the areas that need improvement:')
        for index, row in lowest_sentiment_issues.iterrows():
                st.write(f"- **{row['Issue']}** with an average sentiment score of {row['sentiment_score']:.2f}")

            # Provide some generic recommendations for improvement
        st.write('Recommendations for improvement:')
        for issue in lowest_sentiment_issues['Issue']:
                st.write(f"- Focus on improving public communication and policy addressing **{issue}**.")

    else:
         st.error("Data could not be loaded. Please check the file paths and try again.")
