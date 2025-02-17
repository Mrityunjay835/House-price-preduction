# House Price Prediction using Boston Housing Dataset

## Overview
This project focuses on predicting house prices using the **Boston Housing Dataset**. Although the dataset has been deprecated in Scikit-Learn, it is still widely used for educational purposes. You may need to download it from external sources like Kaggle or the UCI Machine Learning Repository.

## Dataset
The Boston Housing dataset contains information about houses in Boston, including:
- Crime rate
- Zoning classification
- Number of rooms
- Property tax rate
- Pupil-teacher ratio
- Median house value (target variable)

### Loading the Dataset
To load the dataset manually:
```python
import pandas as pd
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
print(df.head())
```

## Dependencies
Ensure you have the following libraries installed:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Project Structure
- `data/` : Stores raw and processed data
- `notebooks/` : Jupyter notebooks for data exploration and model training
- `models/` : Saved models
- `README.md` : Project documentation
- `requirements.txt` : Dependencies

## Model Training
This project includes training a regression model using algorithms such as:
- Linear Regression
- Decision Trees
- Random Forest
- Gradient Boosting

### Example Model Training Code
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['medv']), df['medv'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print("Model Score:", model.score(X_test, y_test))
```

## Results and Analysis
- Evaluate the model using **R² score, RMSE, and MAE**.
- Visualize feature importance and correlations.

## Future Improvements
- Implement deep learning models (e.g., TensorFlow, PyTorch)
- Hyperparameter tuning using GridSearchCV
- Deploy the model using Flask or FastAPI

## License
This project is open-source and available for learning and experimentation.

---

🚀 **Feel free to contribute or fork this repository!**

