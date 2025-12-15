"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- 
- 
- 
- 

Dataset: [Name of your dataset]
Predicting: [What you're predicting]
Features: [List your features]
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns
# TODO: Update this with your actual filename
DATA_FILE = '1000_companies.csv'

def load_and_explore_data(filename):
    """
    Load your dataset and print basic information

    TODO:
    - Load the CSV file
    - Print the shape (rows, columns)
    - Print the first few rows
    - Print summary statistics
    - Check for missing values
    """
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)

    df = pd.read_csv('1000_companies.csv')
    df.replace({'State':{'New York':0,'California':1,'Florida':2}},inplace=True)
    df.head
    df.shape
    df.count
    df.describe
    df.info
    return df


def visualize_data(df, button, predictions, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    """
    Create visualizations to understand your data

    TODO:
    - Create scatter plots for each feature vs target
    - Save the figure
    - Identify which features look most important

    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    #df['Profit'].hist(,figsize=(10,5))

    if button :
      axes[0,0].hist(df['Profit'],bins=50)
      axes[0,0].set_xlabel('Profit',)
      axes[0,0].set_ylabel('number of companies')
      axes[0,0].set_title(' number of companies by Profit')

      sns.boxplot(data=df.drop(columns=['State']),palette="winter",ax=axes[0,1])
      axes[0,1].tick_params(axis = 'x',labelrotation = 90)
      axes[0,1].set_title('all data by profit')

      sns.heatmap(df.drop(columns=['State']).corr(),annot=True, ax= axes[1,0])
      axes[1,0].set_title('how all the data correlates')

      axes[1, 1].scatter(y_test, predictions, color='blue', alpha=0.6)
      axes[1, 1].set_xlabel('Profit ($)')
      axes[1, 1].set_ylabel('Predicted Profit ($)')
      axes[1, 1].set_title('Predicted Profit vs Real Profit')
      axes[1, 1].grid(True, alpha=0.6)
      axes[1, 1].tick_params(axis = 'x',labelrotation = 25)
    if not button:
      axes[0, 0].scatter(df['R&D Spend'], df['Profit'], color='blue', alpha=0.6)
      axes[0, 0].set_xlabel('Reasearch and development spend ($)')
      axes[0, 0].set_ylabel('Profit ($)')
      axes[0, 0].set_title('R&D spend vs Profit')
      axes[0, 0].grid(True, alpha=0.6)
      axes[0, 0].tick_params(axis = 'x',labelrotation = 25)

      axes[0, 1].scatter(df['Administration'], df['Profit'], color='blue', alpha=0.6)
      axes[0, 1].set_xlabel('Administration($)')
      axes[0, 1].set_ylabel('Profit ($)')
      axes[0, 1].set_title('Cost of Administration vs Profit')
      axes[0, 1].grid(True, alpha=0.6)
      axes[0, 1].tick_params(axis = 'x',labelrotation = 25)

      axes[1, 0].scatter(df['Marketing Spend'], df['Profit'], color='blue', alpha=0.6)
      axes[1, 0].set_xlabel('Marketing Spend ($)')
      axes[1, 0].set_ylabel('Profit ($)')
      axes[1, 0].set_title('Marketing Spend vs Profit')
      axes[1, 0].grid(True, alpha=0.6)
      axes[1, 0].tick_params(axis = 'x',labelrotation = 25)

      axes[1, 1].scatter(df['State'], df['Profit'], color='blue', alpha=0.6)
      axes[1, 1].set_xlabel('State (New York-0,California-1,Florida-2)')
      axes[1, 1].set_ylabel('Profit ($)')
      axes[1, 1].set_title('State vs Profit')
      axes[1, 1].grid(True, alpha=0.6)

    plt.tight_layout()
    plt.show()

def drop_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    return df[(df[column] >= Q1 - 1.5 * IQR) &
              (df[column] <= Q3 + 1.5 * IQR)]


def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test

    TODO:
    - Separate features (X) and target (y)
    - Split into train/test (80/20)
    - Print the sizes

    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)

    feature_columns = ['R&D Spend', 'Administration','Marketing Spend', 'State']

    X = data[feature_columns]
    y = data['Profit']
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    # TODO: Print the feature column names
    print(f"\nFeature columns: {list(X.columns)}")

    # Use train_test_split for proper data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train the linear regression model

    TODO:
    - Create and train a LinearRegression model
    - Print the equation with all coefficients
    - Print feature importance (rank features by coefficient magnitude)

    Args:
        X_train: training features
        y_train: training target
        feature_names: list of feature names

    Returns:
        trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    feature_names = ['R&D Spend', 'Administration','Marketing Spend', 'State']
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Intercept: ${model.intercept_:.2f}")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.2f}")

    print(f"\nEquation:")
    equation = f"Profit = "
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        if i == 0:
            equation += f"{coef:.2f} × {name}"
        else:
            equation += f" + ({coef:.2f}) × {name}"
    equation += f" + {model.intercept_:.2f}"
    print(equation)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance

    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)

    Args:
        model: trained model
        X_test: test features
        y_test: test target

    Returns:
        predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    feature_names = ['R&D Spend', 'Administration','Marketing Spend', 'State']
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of Profit variation")
    print(f"\nRoot Mean Mean Squared Error: ${rmse:.2f}")
    print(f"  → On average, predictions are off by ${rmse:.2f}")

    feature_importance = list(zip(feature_names, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    # TODO: Return predictions
    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.2f}")

    print(f"{'Actual Price':<15} {'Predicted Price':<18} {'Error':<12} {'% Error'}")
    print("-" * 60)

    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i]
        predicted = predictions[i]
        error = actual - predicted
        pct_error = (abs(error) / actual) * 100

        print(f"${actual:>13.2f}   ${predicted:>13.2f}   ${error:>10.2f}   {pct_error:>6.2f}%")

    return predictions

    pass


def make_prediction(model):
    """
    Make a prediction for a new example

    TODO:
    - Create a sample input (you choose the values!)부터
    - Make a prediction
    - Print the input values and predicted output

    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)

    # Your code here
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)

    car_features = pd.DataFrame([[156749.2, 116897.8, 271784.1,0]],
                                 columns=['R&D Spend','Administration','Marketing Spend','State'])
    # TODO: Make a prediction using model.predict()
    predicted_price = model.predict(car_features)[0]
    # TODO: Print the house specs and predicted price nicely formatted
    print(f"Predicted price: ${predicted_price:,.2f}")
    # TODO: Return the predicted price
    return predicted_price

if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)

    # Step 2: Visualize
    visualize_data(data, False, None,y_test=None)

    # Step 3: Prepare and split
    data = drop_outliers(data,"Profit")
    data = drop_outliers(data,"Administration")

    X_train, X_test, y_train, y_test = prepare_and_split_data(data)

    # Step 4: Train
    model = train_model(X_train, y_train)

    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)

    # Step 6: Make a prediction, add features as an argument
    make_prediction(model)

    visualize_data(data, True, predictions,y_test)
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")
