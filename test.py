import pickle

with open("combined_forensics_model_fixed.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
print(type(data.get("classifier")))
print(type(data.get("vectorizer")))
#python mini_mesa_EG-SIEM_Enron.py --preset forensics_primary --forensics-mode full --model combined_forensics_model.pkl
#python mini_mesa_EG-SIEM_Enron.py --preset forensics_primary --forensics-mode model_only --model combined_forensics_model.pkl
#python mini_mesa_EG-SIEM_Enron.py --preset forensics_primary --forensics-mode disabled
#python mini_mesa_EG-SIEM_Enron.py --preset forensics_primary --forensics-mode keyword_only