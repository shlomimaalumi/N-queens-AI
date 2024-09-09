import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Load the data from the Excel file
# write this file path: "C:\Users\t-smaalumi\Desktop\AI\genetic algorithm results.xlsx"
file_path = ".\genetic algorithm results.xlsx"
df = pd.read_excel(file_path)

# Function to convert string lists in the dataframe to actual lists
def eval_lists(row):
    for col in ['Mutation Rate', 'Average Steps']:
        row[col] = eval(row[col])
    return row

# Apply the function to each row
df = df.apply(eval_lists, axis=1)

# Function to predict the best mutation rate using interpolation
def predict_best_mutation_rate(n):
    mutation_rates = np.unique(np.concatenate(df['Mutation Rate'].values))
    best_rates = []

    # Interpolating for each mutation rate
    for rate in mutation_rates:
        valid_rows = df[df['Mutation Rate'].apply(lambda rates: rate in rates)]
        valid_n = valid_rows['N']
        valid_steps = valid_rows.apply(lambda row: row['Average Steps'][row['Mutation Rate'].index(rate)], axis=1)
        if len(valid_n) > 1:  # Ensuring there's enough points to interpolate
            f = interp1d(valid_n, valid_steps, kind='linear', fill_value='extrapolate')
            best_rates.append((rate, f(n)))

    # Finding the mutation rate with the minimum interpolated average steps
    best_rate = min(best_rates, key=lambda x: x[1])
    return best_rate

# Example usage: Predict the best mutation rate for N = 15
new_N = 15
predicted_rate = predict_best_mutation_rate(new_N)
print(f"Predicted best mutation rate for N = {new_N} is {predicted_rate[0]} with average steps {predicted_rate[1]}")