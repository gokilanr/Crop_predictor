from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import joblib

app = Flask(__name__)

# Load model and dataset
model = joblib.load('crop_model.pkl')
df = pd.read_csv('Crop_Data.csv')

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    # Handle prediction
    if request.method == "POST":
        try:
            features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            values = [float(request.form[f]) for f in features]
            input_df = pd.DataFrame([values], columns=features)
            prediction = model.predict(input_df)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    # Prepare charts for analysis
    plots = []

    # 1. Crop count chart (BIG)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    df['label'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title("Crop Count (Dataset Distribution)")
    ax1.set_ylabel("Number of Samples")
    fig1.tight_layout()
    crop_count = plot_to_base64(fig1)
    plots.append(("Crop Count", crop_count))

    # 2. Boxplots by crop for each feature (BIG)
    for col in ['temperature', 'humidity', 'ph', 'rainfall']:
        fig, ax = plt.subplots(figsize=(8, 5))
        df.boxplot(column=col, by='label', ax=ax, grid=False)
        ax.set_title(f"{col.capitalize()} by Crop Type")
        ax.set_xlabel("Crop")
        ax.set_ylabel(col.capitalize())
        plt.suptitle("")
        fig.tight_layout()
        chart = plot_to_base64(fig)
        plots.append((f"{col.capitalize()} by Crop", chart))

    # 3. Top 5 crops per condition (BIG)
    for feature in ['temperature', 'humidity', 'ph', 'rainfall']:
        top5 = df.groupby('label')[feature].mean().sort_values(ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(8, 5))
        top5.plot(kind='bar', color='orange', ax=ax)
        ax.set_title(f"Top 5 Crops by Avg {feature.capitalize()}")
        ax.set_ylabel(feature.capitalize())
        fig.tight_layout()
        top_chart = plot_to_base64(fig)
        plots.append((f"Top 5 Crops by {feature.capitalize()}", top_chart))

    return render_template("index.html", prediction=prediction, plots=plots)

if __name__ == '__main__':
    app.run(debug=False)
