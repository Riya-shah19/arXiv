import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add path so we can import timeseries_utils
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import timeseries_utils as tu

# Historic milestones for each field
FIELD_CONFIGS = {
    "cs": {
        "label": "Computer Science (CS)",
        "color": "#2196F3",
        "bg_color": "#E8EAF6",
        "folder": "plots/timeseries/cs",
        "events": {
            1991: "arXiv launched",
            1998: "Google founded / Web boom",
            2001: "Dot-com crash",
            2008: "Financial crisis / Cloud computing",
            2012: "AlexNet - Deep learning revolution",
            2014: "GANs invented / Deep RL",
            2017: "Transformer paper published",
            2018: "BERT / GPT-1 released",
            2020: "COVID-19 research surge",
            2022: "ChatGPT / LLM explosion",
            2023: "GPT-4 / Generative AI boom",
        }
    },
    "math": {
        "label": "Mathematics (MATH)",
        "color": "#9C27B0",
        "bg_color": "#E8EAF6",
        "folder": "plots/timeseries/math",
        "events": {
            1991: "arXiv launched",
            2000: "arXiv math section created",
            2008: "Financial crisis - mathematical modelling in demand",
            2010: "Compressed sensing and sparse methods boom",
            2012: "Deep learning - maths of optimisation needed",
            2014: "ML needs probability and statistical theory",
            2016: "Deep learning theory research begins",
            2018: "Neural tangent kernel - maths of deep learning",
            2020: "COVID-19 - mathematical modelling surge",
            2022: "ChatGPT - theory of LLMs needs mathematics",
            2023: "Geometric deep learning and topology in ML",
        }
    },
    "physics": {
        "label": "Physics",
        "color": "#4CAF50",
        "bg_color": "#E8F5E9",
        "folder": "plots/timeseries/physics",
        "events": {
            1991: "arXiv launched - physics was first field",
            2008: "Large Hadron Collider operations begin",
            2012: "Higgs Boson discovered at CERN",
            2015: "LIGO detects gravitational waves",
            2016: "Gravitational waves confirmed - Nobel Prize",
            2019: "First image of a black hole released",
            2020: "COVID-19 research surge",
            2022: "James Webb Space Telescope launched",
            2023: "AI applied to physics simulations surge",
        }
    },
    "eess": {
        "label": "Electrical Engineering (EESS)",
        "color": "#00BCD4",
        "bg_color": "#E0F7FA",
        "folder": "plots/timeseries/eess",
        "events": {
            1991: "arXiv launched",
            2012: "Deep learning revolution - image processing",
            2014: "Deep learning applied to speech recognition",
            2017: "Transformer - NLP and signal processing",
            2018: "WaveNet and neural audio synthesis",
            2020: "COVID - remote communication research surge",
            2022: "Diffusion models - image and audio generation",
            2023: "Large multimodal models boom",
        }
    },
    "stat": {
        "label": "Statistics (STAT)",
        "color": "#FFA726",
        "bg_color": "#FFF3E0",
        "folder": "plots/timeseries/stat",
        "events": {
            1991: "arXiv launched",
            2006: "R language became widely adopted",
            2008: "Financial crisis - statistics in demand",
            2010: "Big Data era begins",
            2012: "Deep learning revolution / Data science boom",
            2014: "Data science becomes mainstream career",
            2016: "Reproducibility crisis in statistics",
            2017: "Transformer paper - ML meets statistics",
            2019: "Federated learning / Privacy statistics",
            2020: "COVID-19 - statistics in global spotlight",
            2022: "ChatGPT - LLM and Bayesian methods surge",
            2023: "Causal AI and uncertainty quantification boom",
        }
    }
}

def main():
    sns.set_theme(style="whitegrid")
    
    data_path = tu.resolve_data_path(__file__)
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    penalties = [3, 5, 8, 10, 12, 15, 20]
    model_types = ["rbf", "l1", "l2"]

    for field, config in FIELD_CONFIGS.items():
        print(f"\nProcessing {config['label']}...")
        os.makedirs(config["folder"], exist_ok=True)
        
        # Prepare the monthly time series
        ts, monthly = tu.prepare_monthly_series(df, field=field)
        
        # Sweep all model types and penalty parameters
        for model_type in model_types:
            for pen in penalties:
                print(f"  -> Generating plot for Pen={pen} (Model={model_type.upper()})")
                
                # Detect the changepoints
                cp_dates, _ = tu.detect_changepoints(ts, pen=pen, model_type=model_type)
                
                title = f"{config['label']} (Model: {model_type.upper()}) - Changepoint Detection (1991-2025) (Pen={pen})\nRed dashed lines show detected structural breaks"
                save_path = f"{config['folder']}/{field}_overall_changepoints_pen{pen}_{model_type}.png"
                
                # Plot results (only show text event labels on the cleaner RBF plots)
                tu.plot_changepoints(
                    monthly,
                    cp_dates,
                    title=title,
                    color=config["color"],
                    save_path=save_path,
                    events=config["events"],
                    label_bg_color=config["bg_color"],
                    show_event_labels=(model_type == "rbf")
                )

    print("\nAll penalty plots generated and saved successfully!")

if __name__ == "__main__":
    main()
