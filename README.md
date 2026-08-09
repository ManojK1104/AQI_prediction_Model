# 🌫️ AQI Prediction Model

Predicting air-quality parameters from real-world sensor and weather readings using a Random Forest regression pipeline, with an interactive Gradio web app for live predictions.

![Python](https://img.shields.io/badge/python-3.14%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![scikit--learn](https://img.shields.io/badge/model-RandomForestRegressor-yellow)

---

## 📖 Overview

This project builds an end-to-end machine learning pipeline on the **UCI Air Quality dataset** — hourly readings from gas sensors deployed in an Italian city, alongside ground-truth pollutant concentrations and weather data. The pipeline ingests the raw data, cleans and transforms it, trains a `RandomForestRegressor`, and serves the trained model through a themeable, animated Gradio interface.

## ✨ Features

- 🔄 **End-to-end pipeline** — data ingestion → preprocessing → model training, orchestrated from a single entry point
- 🌲 **Random Forest Regressor** (300 estimators) trained with scikit-learn
- 🎛️ **Interactive Gradio UI** — sliders grouped by category (gas concentrations, sensor responses, weather), a live speedometer-style gauge for the prediction, and four switchable color themes (Ocean, Sunset, Forest, Midnight)
- 📦 **Reproducible environment** managed with [`uv`](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`)
- 📄 **Research paper** included documenting the methodology and findings

## 🗂️ Project Structure

```
AQI_prediction_Model/
├── data/                  # Raw / processed dataset files
├── models/                # Serialized trained model(s) — model.pkl
├── research_paper/        # Written report / paper on methodology & results
├── src/
│   ├── data_injection.py      # Loads the raw dataset
│   ├── data_preprocessing.py  # Cleans data, builds train/test splits & transformer
│   └── model_build.py         # Trains the RandomForestRegressor and evaluates it
├── gradio_app.py          # Interactive web UI for making predictions
├── main.py                # Runs the full training pipeline end-to-end
├── pyproject.toml         # Project metadata & dependencies (uv)
├── uv.lock                # Locked dependency versions
└── LICENSE                # Apache-2.0
```

## 📊 Dataset

The model is trained on the **[UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality)**, which contains hourly averaged responses from an array of 5 metal-oxide chemical sensors embedded in an air-quality multisensor device, alongside reference concentrations from a certified analyzer and weather readings.

| Column | Description |
|---|---|
| `Date`, `Time` | Timestamp of the reading |
| `CO(GT)` | True CO concentration (mg/m³) |
| `PT08.S1(CO)` | CO sensor response |
| `NMHC(GT)` | True non-methane hydrocarbons concentration (µg/m³) |
| `C6H6(GT)` | True benzene concentration (µg/m³) |
| `PT08.S2(NMHC)` | NMHC sensor response |
| `NOx(GT)` | True NOx concentration (ppb) |
| `PT08.S3(NOx)` | NOx sensor response |
| `NO2(GT)` | True NO2 concentration (µg/m³) |
| `PT08.S4(NO2)` | NO2 sensor response |
| `PT08.S5(O3)` | O3 sensor response |
| `T` | Temperature (°C) |
| `RH` | Relative Humidity (%) — **prediction target** |
| `AH` | Absolute Humidity (g/m³) |

The model uses the 12 sensor/weather/gas features (excluding `Date`, `Time`, and `RH` itself) to predict **Relative Humidity (RH)**.

## 🧠 Model

- **Algorithm:** `RandomForestRegressor` (scikit-learn)
- **Estimators:** 300 trees
- **Target:** `RH` (Relative Humidity, %)
- **Inputs:** 12 numeric features (gas concentrations, sensor responses, temperature, absolute humidity)

Full methodology, feature engineering decisions, and evaluation metrics are documented in [`research_paper/`](./research_paper).

## 🚀 Getting Started

### Prerequisites

- Python 3.14+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or `pip`

### Installation

Clone the repository:

```bash
git clone https://github.com/ManojK1104/AQI_prediction_Model.git
cd AQI_prediction_Model
```

**Using `uv` (recommended — matches the locked environment):**

```bash
uv sync
```

**Using `pip`:**

```bash
pip install gradio scikit-learn pandas numpy matplotlib seaborn openpyxl ipykernel
```

### Train the Model

Run the full pipeline (load → preprocess → train → evaluate):

```bash
uv run main.py
# or: python main.py
```

This loads the dataset, builds the preprocessing transformer, trains the `RandomForestRegressor`, and prints the resulting shapes and evaluation score.

### Launch the Web App

```bash
uv run gradio_app.py
# or: python gradio_app.py
```

Then open the local URL Gradio prints (typically `http://127.0.0.1:7860`) in your browser. The app auto-detects `model.pkl` whether it's placed next to the script or inside `models/`.

## 🖥️ Using the App

1. Pick a color theme (🌊 Ocean, 🌅 Sunset, 🌲 Forest, 🌌 Midnight) from the top of the page.
2. Adjust the sliders across the three grouped panels — **Gas Concentrations**, **Sensor Responses**, and **Weather Conditions** — or load one of the quick example presets.
3. Click **🔮 Predict RH** to see the predicted Relative Humidity on the animated gauge, tagged as *Dry*, *Comfortable*, or *Humid*.

## 🛠️ Tech Stack

- **Language:** Python 3.14+
- **ML:** scikit-learn (RandomForestRegressor)
- **Data:** pandas, numpy, openpyxl
- **Visualization:** matplotlib, seaborn
- **Web UI:** Gradio
- **Environment/Dependency management:** uv

## 📄 License

This project is licensed under the **Apache License 2.0** — see [`LICENSE`](./LICENSE) for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or file an issue.

## 👤 Author

**Manoj K** — [@ManojK1104](https://github.com/ManojK1104)
