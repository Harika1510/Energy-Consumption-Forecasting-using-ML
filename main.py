import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create GUI Window
root = tk.Tk()
root.title("Energy Consumption Forecasting using ML Models")
root.geometry("1200x850")

results = {}
model_selection = tk.StringVar()
colors = {
    "Random Forest": "red",
    "Linear Regression": "green",
    "Decision Tree": "blue",
    "Support Vector Regressor": "orange"
}

# UI Elements
frame_top = tk.Frame(root)
frame_top.pack(pady=10)

select_label = tk.Label(frame_top, text="Select Model:", font=("Arial", 12))
select_label.pack(side=tk.LEFT, padx=5)

model_dropdown = ttk.Combobox(frame_top, textvariable=model_selection, font=("Arial", 12))
model_dropdown['values'] = ["All Models", "Random Forest", "Linear Regression", "Decision Tree", "Support Vector Regressor"]
model_dropdown.current(0)
model_dropdown.pack(side=tk.LEFT, padx=5)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

text_area = tk.Text(root, height=20, width=140)
text_area.pack(pady=10)

# Function to load and process dataset
def load_dataset():
    global df_original
    file_path = filedialog.askopenfilename(title="Select Dataset")
    if not file_path:
        return

    try:
        data = pd.read_csv(
            file_path,
            sep=';',
            low_memory=False,
            na_values='?',
            parse_dates=[[0, 1]],
            infer_datetime_format=True
        )
        data.columns = ['Datetime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
                        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        data.dropna(inplace=True)
        data = data.astype(float, errors='ignore')
        data.set_index('Datetime', inplace=True)
        data['hour'] = data.index.hour
        data['day'] = data.index.day
        data['month'] = data.index.month
        data['weekday'] = data.index.weekday
        df_original = data.copy()
        messagebox.showinfo("Success", "Dataset loaded and preprocessed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {e}")

# Function to train and evaluate model
def run_analysis():
    if 'df_original' not in globals():
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return

    data = df_original.copy()
    daily_data = data.resample('D').mean().dropna()
    X = daily_data.drop('Global_active_power', axis=1)
    y = daily_data['Global_active_power']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Support Vector Regressor": SVR()
    }

    text_area.delete(1.0, tk.END)
    selection = model_selection.get()
    selected_models = models.keys() if selection == "All Models" else [selection]

    text_area.insert(tk.END, "\nTraining Models and Evaluating Performance...\n\n")
    for name in selected_models:
        model = models[name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred, squared=False),
            "R2": r2_score(y_test, y_pred),
            "y_pred": y_pred
        }
        text_area.insert(tk.END, f"{name} Results:\n")
        text_area.insert(tk.END, f"MAE: {results[name]['MAE']:.4f}\n")
        text_area.insert(tk.END, f"RMSE: {results[name]['RMSE']:.4f}\n")
        text_area.insert(tk.END, f"R² Score: {results[name]['R2']:.4f}\n\n")

    comparison_data = []
    for name in selected_models:
        result = results[name]
        comparison_data.append({
            'Model': name,
            'MAE': round(result['MAE'], 4),
            'RMSE': round(result['RMSE'], 4),
            'R² Score': round(result['R2'], 4)
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("model_comparison_summary.csv", index=False)
    text_area.insert(tk.END, "Model comparison saved as 'model_comparison_summary.csv'\n\n")

    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_test.index, y_test.values, label='Actual', color='black', linewidth=2)
    for name in selected_models:
        ax.plot(y_test.index, results[name]['y_pred'], label=name, linestyle='--', color=colors.get(name, 'gray'))

    ax.set_title("Energy Consumption Forecasting - Model Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Global Active Power (kW)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()

# Buttons
btn_load = tk.Button(btn_frame, text=" Load Dataset", command=load_dataset, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white")
btn_load.grid(row=0, column=0, padx=10)

btn_run = tk.Button(btn_frame, text=" Run Analysis", command=run_analysis, font=("Arial", 12, "bold"), bg="#2196F3", fg="white")
btn_run.grid(row=0, column=1, padx=10)

btn_exit = tk.Button(btn_frame, text="Exit", command=root.quit, font=("Arial", 12, "bold"), bg="#f44336", fg="white")
btn_exit.grid(row=0, column=2, padx=10)

root.mainloop()
