import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import MinMaxScaler

file_path = 'D:\Code\ML-Evalutaion\Fuel_cell_performance_data-Full.csv'
data = pd.read_csv(file_path)

X = data[[col for col in data.columns if col.startswith('F')]] 
y = data['Target4']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Support Vector Regressor': SVR()
}


results = []
metrics = ['Mean Squared Error', 'Mean Absolute Error', 'R^2 Score', 'Explained Variance Score']
scaler = MinMaxScaler()

for model_name, model in models.items():
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    
    results.append({
        'Model': model_name,
        'Mean Squared Error': mse,
        'Mean Absolute Error': mae,
        'R^2 Score': r2,
        'Explained Variance Score': evs
    })

# Normalize results for composite scoring
results_df = pd.DataFrame(results)
normalized_results = scaler.fit_transform(results_df[metrics])
norm_df = pd.DataFrame(normalized_results, columns=metrics)

# Add a composite score with weights (R^2 Score gets more weight)
weights = {
    'Mean Squared Error': -0.2,  # Negative weight since lower is better
    'Mean Absolute Error': -0.2,  # Negative weight since lower is better
    'R^2 Score': 0.5,  # Higher weight since higher is better
    'Explained Variance Score': 0.3  # Moderate weight
}

norm_df['Composite Score'] = sum(norm_df[metric] * weight for metric, weight in weights.items())
results_df['Composite Score'] = norm_df['Composite Score']

# Identify the best model based on composite score
best_model_idx = norm_df['Composite Score'].idxmax()
best_model = results_df.iloc[best_model_idx]['Model']
best_composite_score = results_df.iloc[best_model_idx]['Composite Score']


print(results_df)


results_df.to_csv('model_performance_results.csv', index=False)


print(f"The best model is {best_model} with a Composite Score of {best_composite_score:.4f}")
