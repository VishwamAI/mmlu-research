import matplotlib.pyplot as plt

# Data for performance metrics
models = ['Gemini Ultra', 'Gemini Pro', 'GPT-4', 'GPT-3.5', 'PaLM 2-L', 'Claude 2', 'Inflection-2', 'Grok 1', 'LLAMA-2', 'Palligemma']
mmlu_scores = [90.04, 79.13, 86.4, 70.0, 78.3, 78.5, 0, 0, 68.0, 0]
gsm8k_scores = [94.4, 86.5, 92.0, 57.1, 80.0, 88.0, 81.4, 62.9, 56.8, 0]
math_scores = [53.2, 91.1, 32.6, 52.9, 34.1, 34.4, 0, 0, 34.8, 0]

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot MMLU scores
axs[0].bar(models, mmlu_scores, color='blue')
axs[0].set_title('MMLU Benchmark Scores')
axs[0].set_ylabel('Accuracy (%)')
axs[0].set_ylim(0, 100)

# Plot GSM8K scores
axs[1].bar(models, gsm8k_scores, color='green')
axs[1].set_title('GSM8K Benchmark Scores')
axs[1].set_ylabel('Accuracy (%)')
axs[1].set_ylim(0, 100)

# Plot MATH scores
axs[2].bar(models, math_scores, color='red')
axs[2].set_title('MATH Benchmark Scores')
axs[2].set_ylabel('Accuracy (%)')
axs[2].set_ylim(0, 100)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('/home/ubuntu/performance_metrics_graphs.png')

# Show the figure
plt.show()
