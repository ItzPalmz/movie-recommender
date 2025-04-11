
# Movie Recommendation System
Deployment: 

This project is a **content-based movie and TV show recommender system** built using Python and Streamlit. It analyzes Netflix's title descriptions and recommends similar content based on **text similarity** using **TF-IDF vectorization** and **cosine similarity**. The system provides an interactive interface where users can select a movie or show from a dropdown and instantly receive top recommendations that match its theme or storyline.

Key Features:

-   Uses the official Netflix Titles dataset
    
-   Processes descriptions for intelligent similarity-based recommendations
    
-   Clean, dark-themed user interface using Streamlit
    
-   Lottie animation integration for a visually appealing experience
    
-   Fully functional local or web-based deployment

The goal is to help users quickly find movies and shows similar to the ones they already enjoy, improving discovery and personalization without needing user-specific ratings or login data.

# Introduction

This project leverages natural language processing (NLP) techniques to help users discover new content based on the descriptions of their favorite Netflix titles. By analyzing textual similarities between movie and show descriptions using TF-IDF vectorization and cosine similarity, the system suggests the top 10 most thematically similar titles. Whether you're in the mood for something just like your last binge or want to explore similar genres, this smart recommender makes finding your next watch effortless and fun.

## Dataset

The dataset used in this project is a cleaned version of the **Netflix Titles Dataset**, which contains metadata about movies and TV shows available on Netflix. Each entry includes information such as the **title**, **year of release**, and a **brief description** of the content. For this content-based recommendation system, the **description** field plays a crucial role, as it allows the model to analyze and compare the thematic content of different titles. By transforming these descriptions into numerical representations using **TF-IDF (Term Frequency-Inverse Document Frequency)**, we can effectively calculate similarities between titles and generate meaningful, context-aware recommendations.

Data Source:  [https://www.kaggle.com/datasets/shivamb/netflix-shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)
## Steps to create recommender systems
-   **Collect and Load the Data**
    
    -   Import your dataset (`netflix_titles.csv`).
        
    -   Ensure the dataset has fields like `Title`, `Description`, and possibly `Year`.
        
-   **Preprocess the Data**
    
    -   Handle missing values (e.g., replace missing descriptions with empty strings).
        
    -   Normalize the data if necessary (e.g., lowercase titles, remove spaces).
        
-   **Feature Extraction using TF-IDF**
    
    -   Use `TfidfVectorizer` to convert text (e.g., movie descriptions) into numerical vectors.
        
    -   This step helps capture the importance of words while ignoring common ones (stop words).
        
-   **Compute Similarity Matrix**
    
    -   Use `cosine similarity` or `linear kernel` to measure the similarity between movie vectors.
        
    -   This creates a similarity score between every pair of movies.
        
-   **Build a Movie Index Mapping**
    
    -   Create a mapping between movie titles (or normalized titles) and their index in the dataset.
        
    -   This helps locate a movie's vector for comparison.
        
-   **Create the Recommendation Function**
    
    -   Given a movie title, retrieve its index and compute similarity scores with all other titles.
        
    -   Sort and select the top N most similar movies (excluding itself)
