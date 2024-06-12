# Detailed Report on Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference

## Paper Title
Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference

## Publication Date
7 Mar 2024

## Authors
- Wei-Lin Chiang
- Lianmin Zheng
- Ying Sheng
- Anastasios Nikolas Angelopoulos
- Tianle Li
- Dacheng Li
- Hao Zhang
- Banghua Zhu
- Michael Jordan
- Joseph E. Gonzalez
- Ion Stoica

## Paper Summary
Large Language Models (LLMs) have unlocked new capabilities and applications; however, evaluating the alignment with human preferences still poses significant challenges. To address this issue, we introduce Chatbot Arena, an open platform for evaluating LLMs based on human preferences. Our methodology employs a pairwise comparison approach and leverages input from a diverse user base through crowdsourcing. The platform has been operational for several months, amassing over 240K votes. This paper describes the platform, analyzes the data we have collected so far, and explains the tried-and-true statistical methods we are using for efficient and accurate evaluation and ranking of models. We confirm that the crowdsourced questions are sufficiently diverse and discriminating and that the crowdsourced human votes are in good agreement with those of expert raters. These analyses collectively establish a robust foundation for the credibility of Chatbot Arena. Because of its unique value and openness, Chatbot Arena has emerged as one of the most referenced LLM leaderboards, widely cited by leading LLM developers and companies. Our demo is publicly available at [Chatbot Arena](https://chat.lmsys.org).

## Links
- [PDF](https://arxiv.org/pdf/2403.04132v1.pdf)
- [Abstract](https://arxiv.org/abs/2403.04132v1)

## Code Repositories
- [lm-sys/fastchat](https://github.com/lm-sys/fastchat)
- [Peter-Devine/multilingual_mt_bench](https://github.com/Peter-Devine/multilingual_mt_bench)
- [BirgerMoell/SwedishLLMBenchmark](https://github.com/BirgerMoell/SwedishLLMBenchmark)

## Tasks
- Chatbot

## Datasets
- MMLU
- HellaSwag
- MT-Bench

## Results
- The platform has been operational since April 2023, receiving over 240K votes from about 90K users in over 100 different languages as of January 2024.
- The data involves more than 50 models, including proprietary models like GPT-4, Claude, and Gemini, as well as open models such as LLaMA and Mistral.
- The average number of votes collected for each model is 8,000.
- The agreement rate between crowd-users, experts, and GPT-4 as a judge is presented in Table 3 of the paper.
- The platform ran a replay of 213,576 historical votes to calculate the Bradley-Terry (BT) coefficients for model ranking, with confidence intervals provided in Figure 5 of the paper.
- The agreement rates between crowd-users and experts range from 72% to 83%, with expert-expert agreement rates around 79.4% to 89.8%.

## Methods
- The platform uses a pairwise comparison approach and leverages input from a diverse user base through crowdsourcing.
- Statistical models and efficient sampling algorithms are employed to rank models based on user preferences.
- The system does not preset prompts, allowing users to input any prompt, which encourages a diverse set of inputs and real-world usage representation.
- The human's response is recorded as Ht, indicating their preference between the two models in each comparison.
- The win matrix estimation involves defining Xt(a) as a function of the probability of sampling pair a at time t, Pt(a), and the human's response, Ht. The estimator θ̂T is then calculated as the average of Xt over time T.

## Conclusion
Chatbot Arena is an open platform for evaluating LLMs based on human preferences using a pairwise comparison approach and crowdsourcing. The platform has been operational for several months, amassing over 240K votes. The paper describes the platform, analyzes the data collected, and explains the statistical methods used for evaluation and ranking. The demo is publicly available at [Chatbot Arena](https://chat.lmsys.org). Future work includes further fine-tuning and releasing larger models trained on larger pretraining corpora.
