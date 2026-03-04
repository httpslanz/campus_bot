import pickle

with open("ml_models/hybrid_model_20260220_215355.pkl", "rb") as f:
    data = pickle.load(f)

# Show first 10 training samples
for i in range(10):
    print("Q:", data["training_questions"][i])
    print("Intent:", data["training_intents"][i])
    print("-" * 40)