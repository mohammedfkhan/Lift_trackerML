import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Hardcoded training data
csv_data = """body_weight,current_bench,reps_done,sleep_score,consistency_days,weeks_to_milestone
150,135,8,85,30,6
152,140,6,70,45,5
148,135,4,90,15,8
155,145,8,80,60,3
150,135,2,60,10,12
160,155,5,75,90,2
150,135,7,88,28,6
153,140,8,82,40,4
149,130,8,65,20,9
151,135,6,80,35,7
"""

# 2. Load the data into a Pandas DataFrame
df = pd.read_csv(io.StringIO(csv_data))

print("--- Step 1: Inspection ---")
print(df.head())
print("\n")

# 3. Separate Features (X) from Target (y)
X = df[['body_weight', 'current_bench', 'reps_done', 'sleep_score', 'consistency_days']]
y = df['weeks_to_milestone']

# 4. Split the data into Training and Validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# 5. Initialize and train the Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate model performance
predictions = model.predict(X_val)
mae = mean_absolute_error(y_val, predictions)

print("--- Step 2: Model Performance ---")
print(f"Model Mean Absolute Error: {mae:.2f} weeks")
print("\n")

# 7. Test it live on your current stats!
my_profile = [[150, 135, 8, 85, 30]] 
predicted_weeks = model.predict(my_profile)

print("--- Step 3: Real-Time Forecast ---")
print(f"Based on your patterns, you are approximately {predicted_weeks[0]:.1f} weeks away from your milestone!")