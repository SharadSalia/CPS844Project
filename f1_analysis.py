# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
try:
    circuits = pd.read_csv('circuits.csv')
    constructor_results = pd.read_csv('constructor_results.csv')
    constructor_standings = pd.read_csv('constructor_standings.csv')
    constructors = pd.read_csv('constructors.csv')
    driver_standings = pd.read_csv('driver_standings.csv')
    drivers = pd.read_csv('drivers.csv')
    lap_times = pd.read_csv('lap_times.csv')
    pit_stops = pd.read_csv('pit_stops.csv')
    qualifying = pd.read_csv('qualifying.csv')
    races = pd.read_csv('races.csv')
    results = pd.read_csv('results.csv')
    seasons = pd.read_csv('seasons.csv')
    status = pd.read_csv('status.csv')
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    print("Please ensure all CSV files are in the same directory as the script")
    exit()

# Task 1: Data Preparation
# Merge relevant data
race_results = pd.merge(results, races, on='raceId')
race_results = pd.merge(race_results, drivers, on='driverId')
race_results = pd.merge(race_results, constructors, on='constructorId')

# Create target variable: podium finish (1 if position <= 3, else 0)
race_results['position'] = pd.to_numeric(race_results['position'], errors='coerce')
race_results['podium'] = race_results['position'].apply(lambda x: 1 if not pd.isna(x) and x <= 3 else 0)

# Select and clean relevant features
features = race_results[['grid', 'laps', 'milliseconds', 'fastestLapSpeed', 
                        'driverId', 'constructorId', 'circuitId', 'year']].copy()

# Convert numeric columns properly
features['grid'] = pd.to_numeric(features['grid'], errors='coerce')
features['laps'] = pd.to_numeric(features['laps'], errors='coerce')
features['milliseconds'] = pd.to_numeric(features['milliseconds'], errors='coerce')
features['fastestLapSpeed'] = pd.to_numeric(features['fastestLapSpeed'], errors='coerce')

# Fill missing values
numeric_cols = ['grid', 'laps', 'milliseconds', 'fastestLapSpeed']
features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].median())

# Encode categorical variables
le = LabelEncoder()
features['driverId'] = le.fit_transform(features['driverId'])
features['constructorId'] = le.fit_transform(features['constructorId'])
features['circuitId'] = le.fit_transform(features['circuitId'])

target = race_results['podium']

# Task 2: Exploratory Data Analysis
# Visual exploration
plt.figure(figsize=(12, 6))
sns.countplot(x='podium', data=race_results)
plt.title('Podium vs Non-Podium Finishes')
plt.show()

# Correlation matrix (numeric columns only)
plt.figure(figsize=(12, 8))
sns.heatmap(features[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Task 3: Preprocessing
# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Task 4: Model Comparison
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB()
}

results = {}
for name, clf in classifiers.items():
    try:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cv_score = cross_val_score(clf, X_train_scaled, y_train, cv=5).mean()
        results[name] = {
            'Accuracy': accuracy,
            'CV Score': cv_score,
            'Model': clf
        }
        print(f"{name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"CV Score: {cv_score:.4f}")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Error with {name}: {e}")

# Task 5: Feature Selection
# Method 1: SelectKBest
selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(X_train_scaled, y_train)
selected_features_kbest = features.columns[selector.get_support()]
print("Top 5 features (SelectKBest):", list(selected_features_kbest))

# Method 2: RFE with Random Forest
rfe_selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=5, step=1)
rfe_selector.fit(X_train_scaled, y_train)
selected_features_rfe = features.columns[rfe_selector.get_support()]
print("Top 5 features (RFE):", list(selected_features_rfe))

# Compare performance with selected features
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

selected_results = {}
for name, clf in classifiers.items():
    try:
        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        cv_score = cross_val_score(clf, X_train_selected, y_train, cv=5).mean()
        selected_results[name] = {
            'Accuracy': accuracy,
            'CV Score': cv_score
        }
        print(f"{name} with feature selection:")
        print(f"Accuracy: {accuracy:.4f} (Original: {results[name]['Accuracy']:.4f})")
        print(f"CV Score: {cv_score:.4f} (Original: {results[name]['CV Score']:.4f})")
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Error with {name} and feature selection: {e}")

# Task 6: Additional Analysis - Driver Performance Clustering
# Create a clean version of driver stats without problematic columns
driver_stats = race_results.groupby('driverId').agg({
    'podium': 'mean',
    'grid': 'mean'
}).reset_index()

# Add fastest lap speed separately after proper cleaning
fastest_lap = race_results[['driverId', 'fastestLapSpeed']].copy()
fastest_lap['fastestLapSpeed'] = pd.to_numeric(fastest_lap['fastestLapSpeed'], errors='coerce')
fastest_lap_mean = fastest_lap.groupby('driverId')['fastestLapSpeed'].mean().reset_index()

driver_stats = pd.merge(driver_stats, fastest_lap_mean, on='driverId')

# Perform clustering
kmeans = KMeans(n_clusters=3)
driver_stats['cluster'] = kmeans.fit_predict(driver_stats[['podium', 'grid', 'fastestLapSpeed']].fillna(0))

plt.figure(figsize=(10, 6))
sns.scatterplot(data=driver_stats, x='grid', y='podium', hue='cluster', palette='viridis')
plt.title('Driver Performance Clustering')
plt.xlabel('Average Grid Position')
plt.ylabel('Podium Percentage')
plt.show()

# Feature importance from Random Forest
rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), features.columns[indices], rotation=90)
plt.tight_layout()
plt.show()