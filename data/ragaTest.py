from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# 1. Prepare your dataset
# This should be a Hugging Face Dataset object
# In a real scenario, you would load this from your RAG system's logs or a file
data = {
    "question": [
        "Who is the founder of Apple?",
        "What is the capital of France?"
    ],
    "answer": [
        "Steve Jobs is the founder of Apple.",
        "The capital of France is Paris, a city known for its art and culture."
    ],
    "contexts": [
        ["Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976."],
        ["Paris is the capital and most populous city of France.", "France is a country in Western Europe."]
    ],
    "ground_truth": [
        "Steve Jobs, Steve Wozniak, and Ronald Wayne are the founders of Apple.",
        "The capital of France is Paris."
    ]
}

dataset = Dataset.from_dict(data)

# 2. Define the metrics for evaluation
# Ragas provides these metrics as pre-defined objects
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

# 3. Run the evaluation
# The evaluate function takes the dataset and the list of metrics
# It automatically maps the columns (question, answer, contexts, ground_truth) to the metrics
result = evaluate(dataset, metrics)

# 4. View the results
print(result)

# The result is a dictionary containing the scores for each metric
# You can also convert it to a pandas DataFrame for better visualization
result_df = result.to_pandas()
print(result_df)