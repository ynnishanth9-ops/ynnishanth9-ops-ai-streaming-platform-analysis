import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(">>> Script started")

# 1.Load the data.
df = pd.read_csv('/Users/nishanthyn/Desktop/Movie_Project/moviestreams.csv')
print("Data loaded. First 5 rows:")
print(df.head())

# 2.Drop unnecessary columns
df.drop(['Unnamed: 0', 'ID'], axis=1, inplace=True)
print("Data loaded. First 5 rows:")
print(df.head())

# 3.Age distribution across all movies----Figure
age_counts = df['Age'].value_counts().sort_index()
print("Age counts:")
print(age_counts)

plt.figure(figsize=(8, 6))
age_counts.plot(kind='bar')

plt.title("Figure_1: Number of Movies in each Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")

plt.tight_layout()

# 4.Check missing values
print("Missing values in each column:")
print(df.isna().sum())

df['New_Rotten_Tomatoes'] = (
    df['Rotten Tomatoes']
    .astype(str)
    .str.rstrip('%')
    .replace('nan', np.nan)
    .astype(float)
)
print("New Rotten Tomatoes (first 10):")
print(df['New_Rotten_Tomatoes'].head(10))

# 5.Age-group distribution per platform
print("\n>>> Age-group distribution per platform...")

platform_cols = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']

for platform in platform_cols:
    platform_df = df[df[platform] == 1]
    age_counts_plat = platform_df['Age'].value_counts().sort_index()

    print(f"\n{platform} - age distribution:")
    print(age_counts_plat)

    plt.figure(figsize=(8, 6))
    age_counts_plat.plot(kind='bar')
    plt.title(f"Figure_{platform}: Number of Movies in each Age Group on {platform}")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.tight_layout()

# 6.Runtime distribution (top 10 most common runtimes)----Figure
print("\n>>> Runtime distribution (top 10 most common runtimes)...")

runtime_counts = df['Runtime'].value_counts().sort_values(ascending=False).head(10)
print(runtime_counts)

plt.figure(figsize=(8, 6))
runtime_counts.plot(kind='bar')
plt.title("Figure_3: Top 10 Most Common Movie Runtimes")
plt.xlabel("Runtime (minutes)")
plt.ylabel("Count")
plt.tight_layout()

# 7.Directors and number of movies directed (top 20)----Figure
print("\n>>> Director counts (top 20)...")

df['Directors'] = df['Directors'].astype(str)
new_data = df[df['Directors'].notna()].copy()

directors_count = {}
for xdir in new_data['Directors']:
    curr_dirs = [d.strip() for d in xdir.split(",")]
    for xd in curr_dirs:
        if xd:
            directors_count[xd] = directors_count.get(xd, 0) + 1

dir_count_df = (
    pd.DataFrame(directors_count.items(), columns=['Director', 'Count'])
    .sort_values(by='Count', ascending=False)
    .head(20)
)

print(dir_count_df)

plt.figure(figsize=(10, 6))
plt.barh(dir_count_df['Director'], dir_count_df['Count'])
plt.gca().invert_yaxis()
plt.title("Figure_4: Top 20 Directors by Number of Movies")
plt.xlabel("Number of Movies")
plt.ylabel("Director")
plt.tight_layout()

# 8.Genre analysis----Figure
print("\n>>> Genre frequency analysis...")

genres_series = df['Genres'].dropna().astype(str)
genres_counts = {}

for g in genres_series:
    parts = [x.strip() for x in g.split(',')]
    for genre in parts:
        if genre:
            genres_counts[genre] = genres_counts.get(genre, 0) + 1

genres_df = (
    pd.DataFrame(genres_counts.items(), columns=['Genre', 'Count'])
    .sort_values(by='Count', ascending=False)
)

print(genres_df.head(20))

plt.figure(figsize=(12, 6))
top_n = 20
plt.bar(genres_df['Genre'].head(top_n), genres_df['Count'].head(top_n))
plt.xticks(rotation=45, ha='right')
plt.title(f"Figure_5: Top {top_n} Genres by Count")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.tight_layout()

# 9.Top movies on each platform (IMDb > 8.5)----Figure

print("\n>>> Top movies (IMDb > 8.5) on each platform...")

rating_threshold = 8.5

for platform in platform_cols:
    platform_df = df[(df[platform] == 1) & (df['IMDb'].notna())]
    top_movies = (
        platform_df[platform_df['IMDb'] > rating_threshold][['Title', 'IMDb']]
        .sort_values(by='IMDb', ascending=False)
        .head(20)
    )

    print(f"\nTop {len(top_movies)} movies on {platform} (IMDb > {rating_threshold}):")
    print(top_movies)

    if not top_movies.empty:
        plt.figure(figsize=(10, 6))
        plt.barh(top_movies['Title'], top_movies['IMDb'])
        plt.gca().invert_yaxis()
        plt.title(f"Figure_{platform}: Top IMDb Rated Movies on {platform} (>{rating_threshold})")
        plt.xlabel("IMDb Rating")
        plt.ylabel("Title")
        plt.tight_layout()

print("\n>>> Full EDA completed.")

# AI / ML: Predict IMDb rating using movie metadata
print("\n>>> Starting ML model to predict IMDb rating...")

# We will predict IMDb using Year, Runtime, Age, Genres, Country, Language
# 1) Prepare the ML dataframe: drop rows where IMDb is missing
ml_df = df.copy()
ml_df = ml_df[ml_df['IMDb'].notna()].copy()

features = ['Year', 'Runtime', 'Age', 'Genres', 'Country', 'Language']
target = 'IMDb'

X = ml_df[features]
y = ml_df[target]

numeric_features = ['Year', 'Runtime']
categorical_features = ['Age', 'Genres', 'Country', 'Language']

# 2) Preprocessing:
# Numeric: impute missing values with median
numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
    ]
)

# Categorical: impute missing with most frequent, then one-hot encode
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)

# 3) Model: RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ('preprocess', preprocess),
        ('model', model),
    ]
)

# 4) Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(">>> Fitting model...")
pipeline.fit(X_train, y_train)

# 5) Evaluation
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)  # this returns MSE
rmse = np.sqrt(mse)                       # take square root manually

r2 = r2_score(y_test, y_pred)

print("\n>>> Model performance on test set:")
print(f"MAE  (mean absolute error): {mae:.3f}")
print(f"RMSE (root mean squared error): {rmse:.3f}")
print(f"R²   (coefficient of determination): {r2:.3f}")

# 11. Use the model to fill missing IMDb values
print("\n>>> Using model to fill missing IMDb values...")

df_with_pred = df.copy()
mask_missing_imdb = df_with_pred['IMDb'].isna()

if mask_missing_imdb.any():
    X_missing = df_with_pred.loc[mask_missing_imdb, features]
    imdb_pred = pipeline.predict(X_missing)
    df_with_pred.loc[mask_missing_imdb, 'IMDb_predicted'] = imdb_pred
else:
    print("No missing IMDb values found.")
    df_with_pred['IMDb_predicted'] = pd.NA

# Create a column that uses actual IMDb where available, else predicted
df_with_pred['IMDb_filled'] = df_with_pred['IMDb']
needs_fill = df_with_pred['IMDb_filled'].isna()
df_with_pred.loc[needs_fill, 'IMDb_filled'] = df_with_pred.loc[needs_fill, 'IMDb_predicted']

# 12. Recompute platform quality using IMDb_filled

print("\n>>> Platform quality metrics with IMDb_filled...")

platform_stats = []
for platform in ['Netflix', 'Hulu', 'Prime Video', 'Disney+']:
    subset = df_with_pred[df_with_pred[platform] == 1]

    platform_stats.append({
        'Platform': platform,
        'Num_movies': int(len(subset)),
        'Avg_IMDb_original': float(subset['IMDb'].mean(skipna=True)),
        'Avg_IMDb_filled': float(subset['IMDb_filled'].mean(skipna=True)),
    })

platform_stats_df = pd.DataFrame(platform_stats)
print(platform_stats_df)

# Plot average filled rating per platform
plt.figure(figsize=(8, 6))
plt.bar(platform_stats_df['Platform'], platform_stats_df['Avg_IMDb_filled'])
plt.title("Figure: Average IMDb Rating per Platform (using IMDb_filled)")
plt.xlabel("Platform")
plt.ylabel("Average IMDb (filled)")
plt.tight_layout()

print(">>> Script finished")
plt.show()






