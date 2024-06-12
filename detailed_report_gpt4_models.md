# Detailed Report on GPT-4 Models

## Abstract
GPT-4 is a large-scale, multimodal model capable of processing both text and image inputs to produce text outputs. It exhibits human-level performance on various professional and academic benchmarks, including a top 10% score on a simulated bar exam. The model uses a Transformer-based architecture with a post-training alignment process that enhances factuality and behavior adherence. A significant aspect of the project was the creation of infrastructure and optimization methods that scale predictably, enabling accurate predictions of GPT-4's performance from smaller models using significantly less compute.

## Introduction
GPT-4 is a large multimodal model that processes both image and text inputs to produce text outputs, with potential applications in dialogue systems, text summarization, and machine translation. It has been the subject of substantial interest and progress in recent years. GPT-4 was evaluated on a variety of exams originally designed for humans, often outperforming the vast majority of human test takers. For example, on a simulated bar exam, GPT-4 achieves a score that falls in the top 10% of test takers, contrasting with GPT-3.5, which scores in the bottom 10%. On the MMLU benchmark, GPT-4 outperforms existing models by a considerable margin in English and demonstrates strong performance in other languages, surpassing the English-language state-of-the-art in 24 of 26 languages considered.

## Scope and Limitations
This report focuses on the capabilities, limitations, and safety properties of GPT-4. GPT-4 is a Transformer-style model pre-trained to predict the next token in a document, using both publicly available data and data licensed from third-party providers. The model was then fine-tuned using Reinforcement Learning from Human Feedback (RLHF). Due to competitive and safety considerations, the report omits specific technical details such as architecture size and training methods. However, there is a commitment to independent auditing and potential sharing of more detailed information with trusted third parties.

## Predictable Scaling
A large focus of the GPT-4 project was building a deep learning stack that scales predictably, avoiding the need for extensive model-specific tuning for large training runs. Infrastructure and optimization methods were developed to have predictable behavior across multiple scales, allowing for accurate predictions of GPT-4's performance from smaller models using significantly less compute. This included successful predictions of GPT-4's final loss and the pass rate on the HumanEval dataset, which assesses the model's ability to generate Python functions.

## Evaluation on Benchmarks
GPT-4 was evaluated on various benchmarks, including exams designed for humans, without specific training for these exams. Some exam problems were seen during training, and a variant with these questions removed was also run to ensure representativeness. The evaluation used publicly available materials, with both multiple-choice and free-response formats, and included images where necessary. Scores were combined according to publicly available methodologies, and percentiles were estimated and reported for overall scores.

## Performance on Academic and Professional Exams
GPT-4 exhibits human-level performance on the majority of professional and academic exams, including a top 10% score on a simulated bar exam. The model's capabilities on exams are primarily attributed to the pre-training process, with RLHF not having a significant impact on its performance, particularly in multiple-choice questions.

## Limitations
Despite its capabilities, GPT-4 has similar limitations as earlier GPT models, such as hallucinating facts and making reasoning errors. Care should be taken when using language model outputs, especially in high-stakes contexts, with protocols such as human review, grounding with additional context, or avoiding high-stakes uses altogether. GPT-4 has made progress in reducing hallucinations compared to GPT-3.5, scoring 19 percentage points higher in internal evaluations for factuality.

## Risks and Mitigations
Significant effort was invested in improving the safety and alignment of GPT-4, including the use of domain experts for adversarial testing and red-teaming, and the development of a model-assisted safety pipeline. While GPT-4 shares risks common to smaller language models, its enhanced capabilities introduce new risks, necessitating deeper understanding and mitigation strategies.

## Conclusion
GPT-4 is characterized as a large multimodal model with human-level performance on certain difficult professional and academic benchmarks. It outperforms existing large language models on a collection of NLP tasks and exceeds the vast majority of reported state-of-the-art systems. Improved capabilities are demonstrated in many different languages. Predictable scaling allowed for accurate predictions on the loss and capabilities of GPT-4. While GPT-4 presents new risks due to increased capability, significant steps have been taken to understand and improve its safety and alignment. GPT-4 represents a significant advancement towards broadly useful and safely deployed AI systems.
