import pandas as pd
import ast
# Load the data from the CSV file
#data = pd.read_csv(r'C:\Users\shiva\Documents\project_essentials\DNB-1\pred_results_K_5.csv')  # Replace 'path_to_your_file.csv' with the actual file path
data = pd.read_csv('C:/Users/shiva/Documents/project_essentials/DNB-2/predictions_lora_12.csv')

# Check the first few entries in the Probability column
data['Probability'] = data['Probability'].apply(ast.literal_eval)

# Extract the first probability from each list (or change the index to 1 for the second probability)
data['Second_Probability'] = data['Probability'].apply(lambda x: x[1])
# Define the threshold
threshold = 0.5

# Convert probabilities to binary using the threshold
data['Prediction'] = (data['Second_Probability'] > threshold).astype(int)

# Save the predictions to a new CSV file
data.to_csv('C:/Users/shiva/Documents/project_essentials/DNB-2/predictions_roundoff.csv', index=False)

print("Predictions have been saved to 'predictions.")
