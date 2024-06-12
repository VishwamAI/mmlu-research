# How to Reach MMLU 100% Accuracy

## Abstract
This paper explores the methodologies and strategies required to achieve 100% accuracy on the Massive Multitask Language Understanding (MMLU) benchmark. We review state-of-the-art models, analyze their architectures, and discuss the key challenges and techniques involved in reaching this milestone.

## Introduction
The Massive Multitask Language Understanding (MMLU) benchmark is a comprehensive evaluation of a model's ability to perform a wide range of language tasks. Achieving 100% accuracy on this benchmark is a significant milestone that demonstrates a model's capability to understand and process language at a human-expert level. This paper aims to explore the methodologies and strategies required to reach this goal. We will review the current state-of-the-art models, analyze their architectures, and discuss the key challenges and techniques involved in achieving 100% accuracy on the MMLU benchmark.

## Related Work
The MMLU benchmark has seen significant advancements in recent years, with several models achieving remarkable performance. Notable among these are the Gemini Ultra ~1760B, GPT-4o, and Claude 3 Opus models. The Gemini Ultra ~1760B model, in particular, has set a new standard by achieving an average performance of 90% on the MMLU benchmark. This section reviews the existing models, their performance metrics, and key papers and benchmarks that have contributed to the progress in this field.

### MMLU-Pro Benchmark
MMLU-Pro is an enhanced dataset designed to extend the MMLU benchmark by integrating more challenging, reasoning-focused questions and expanding the choice set from four to ten options. It aims to address the performance plateau of large-scale language models on existing benchmarks and reduce the sensitivity of model scores to prompt variations. MMLU-Pro spans 14 diverse domains, including over 12,000 questions, and is more discriminative in distinguishing nuances between models. The leading model, GPT-4o, achieves an accuracy of 72.6% on MMLU-Pro, indicating substantial room for improvement. MMLU-Pro necessitates chain-of-thought reasoning for better performance, contrasting with MMLU where this approach may be detrimental. Error analysis of GPT-4o shows that the majority of errors are due to reasoning flaws, lack of domain knowledge, and computational mistakes, highlighting areas for future research and model development.

## Methodology
This section describes the models analyzed, their architectures, training techniques, and the datasets used. We also integrate images and model designs to provide a comprehensive understanding of the methodologies employed.

### Key Strategies and Techniques
This subsection compiles the key strategies and techniques used by the top-performing models analyzed in this research.

- **Gemini Ultra ~1760B**:
  - **Multimodal and Multilingual Training**: Training on a diverse dataset that includes text, image, audio, and video data to enhance cross-modal understanding.
  - **Efficient Attention Mechanisms**: Utilizing multi-query attention and other efficient attention mechanisms to support long context lengths.
  - **Advanced Training Infrastructure**: Leveraging TPUv5e and TPUv4 accelerators across multiple datacenters for stable large-scale training and optimized inference.

- **GPT-4o**:
  - **Multimodal Capabilities**: Processing both text and image inputs to produce text outputs, expanding the model's application scope.
  - **Predictable Scaling**: Developing infrastructure and optimization methods for predictable behavior across multiple scales.
  - **Reinforcement Learning from Human Feedback (RLHF)**: Fine-tuning the model using RLHF to enhance factuality and adherence to desired behaviors.

- **Claude 3 Opus**:
  - **Multimodal Architecture**: Integrating vision capabilities to process and analyze image data, enhancing reasoning, math, and coding capabilities.
  - **Constitutional AI**: Aligning the model with human values through a set of ethical and behavioral principles.
  - **Comprehensive Safety Measures**: Implementing an Acceptable Use Policy (AUP), real-time classifiers, and regular risk assessments to ensure safety and ethical AI development.

- **Leeroo Orchestrator**:
  - **LLM-Based Orchestrator**: Selecting the right underlying LLM experts for optimal task execution.
  - **Synthetic Data Generation via Self-Play**: Creating training data through a loop of query generation, orchestration, and evaluation.
  - **Cost-Effective Performance**: Achieving higher accuracy at a lower cost compared to existing models by optimizing the synergy between multiple LLMs.

### Models Analyzed
- **Gemini Ultra ~1760B**: This model is part of the Gemini family of multimodal models, which includes Ultra, Pro, and Nano sizes. The Gemini Ultra model has achieved human-expert performance on the MMLU benchmark.

  - **Architecture**: The Gemini models are based on Transformer decoders enhanced with improvements in architecture and model optimization. They support a 32k token context length and use efficient attention mechanisms such as multi-query attention. The first version, Gemini 1.0, comprises three main sizes to support a wide range of applications. The models are designed for stable large-scale training and optimized inference on Google's Tensor Processing Units (TPUs).

  - **Training Techniques**: The Gemini models are trained using TPUv5e and TPUv4 accelerators across multiple Google datacenters. The training infrastructure is significantly scaled up compared to previous models, resulting in a decrease in the mean time between hardware failures. The models are trained on a multimodal and multilingual dataset, incorporating data from web documents, books, code, and including image, audio, and video data. The SentencePiece tokenizer is used to improve vocabulary inference and model performance. The training process also includes innovations in training algorithms, dataset construction, and infrastructure.

  - **Datasets Used**: The pre-training dataset for Gemini models is both multimodal and multilingual, including data from web documents, books, code, and various media types. This diverse dataset enables the models to handle a wide range of tasks and languages effectively. The dataset includes image, audio, and video data, enhancing the models' capabilities in cross-modal understanding.

  - **Performance Metrics**: The Gemini Ultra model achieves new state-of-the-art results in 30 out of 32 benchmarks, including 10 of 12 popular text and reasoning benchmarks, and 9 of 9 image understanding benchmarks. On the MMLU benchmark, Gemini Ultra achieves an accuracy of 90.04%, surpassing human expert performance and the previous state-of-the-art result. The model also demonstrates strong performance in zero-shot evaluations and complex reasoning tasks.

  - **Significance**: The Gemini Ultra model's performance on the MMLU benchmark and other tasks highlights its advanced capabilities in cross-modal reasoning and language understanding. The model's success demonstrates the potential for developing generalist agents capable of tackling complex, multi-step problems. The responsible deployment of Gemini models through various Google services further underscores their practical applications and impact. The models' ability to handle long context lengths and their strong performance in image understanding benchmarks are particularly noteworthy.

  - **Applications**: The Gemini models are suitable for a wide range of applications, from complex reasoning tasks to on-device memory-constrained use cases. Their advanced capabilities in text, image, audio, and video understanding make them valuable for various fields, including education, competitive programming, and more. The models' multimodal capabilities enable them to process and analyze diverse data types, making them versatile for a wide range of use cases.

  - **Ethical Considerations and Safety Measures**:
    - **Social Responsibility**: Google emphasizes its commitment to developing AI systems that are safe and responsible at every stage. The Gemini models have improved in understanding requests and discerning real harm, reducing unnecessary refusals to harmless prompts. However, there is an acknowledgment that mistakes can still occur, and efforts to enhance the models' helpfulness, harmlessness, and honesty continue. Ethical considerations are integral to the Acceptable Use Policy (AUP) and the Trust and Safety processes that enforce it, guiding the permissible uses of the Gemini models.
    - **Constitutional AI**: The Gemini models are guided by a Constitution, which is a set of ethical and behavioral principles aimed at ensuring the models' outputs are helpful, honest, and harmless. This Constitution includes principles to prevent outputs that are sexist, racist, or toxic and to discourage engagement in illegal or unethical activities. An additional principle has been added to make Gemini more accessible and understanding towards individuals with disabilities, which has helped reduce stereotype bias in the model.
    - **Behavioral Design**: The Gemini model family has been improved in areas such as making appropriate refusals, maintaining honesty and truthfulness, following instructions accurately, and formatting responses to suit various customer use cases. This reflects a commitment to ethical AI development, ensuring that the models are safe, ethical, and beneficial to users while being capable of taking useful actions.
    - **Trust & Safety**: Google implements comprehensive testing and red team exercises to ensure the safety of their models before deployment, aiming to minimize harmful outputs. They utilize real-time classifiers to detect and respond to Acceptable Use Policy (AUP) violations, modifying prompts or blocking responses as needed. Additionally, they maintain a detection and auditing system to identify and remove access for bad actors, and they encourage user participation in reporting safety concerns.
    - **Catastrophic Risk Evaluations and Mitigations**: The Responsible Scaling Policy (RSP) mandates regular risk assessments of models, focusing on biological, cyber, and autonomous replication and adaption (ARA) capabilities. Evaluations on Gemini Ultra, including versions with harmlessness training, showed no catastrophic risk, classifying the models as ASL-2. The evaluation methodology is evolving, with plans to integrate findings into future RSP iterations and model launches.
    - **Autonomous Replication and Adaption (ARA) Evaluations**: The ARA evaluations assess the Gemini model's ability to autonomously perform tasks that could indicate a potential for accumulating resources, exploiting vulnerabilities, deceiving humans, or surviving without human intervention. The model did not meet the pre-specified warning indicator for ASL-3, which would require passing more than 50% of the tasks with at least a 10% success rate. Tasks included implementing a Flask exploit, fine-tuning an open-source language model to add a backdoor, executing a SQL injection, setting up a copycat API service, and writing a self-replicating worm. The model underwent multiple rounds of testing and elicitation improvement but remained below the ASL-3 risk threshold.
    - **Biological Evaluations**: The Gemini models underwent biological evaluations to assess their ability to answer technical questions that could potentially cause harm. The evaluations included automated question-answering and human uplift trials, comparing the performance of individuals using the Gemini models to those using Google. The models did not exceed the conservative risk thresholds set for misuse, indicating no significant increase in harmful knowledge dissemination. The evaluations also revealed mixed results, with the Ultra model performing better than previous versions on some tests but worse on others, suggesting potential under-elicitation of capabilities. Further refinement and exploration of the evaluation methods are planned to more accurately

  ![Gemini Ultra Architecture](gemini_ultra_architecture.png)

- **GPT-4o**: A variant of the GPT-4 model, optimized for multitask language understanding.

  - **Architecture**: GPT-4o is based on the Transformer architecture, with enhancements to support multitask language understanding. The model includes improvements in attention mechanisms and optimization techniques to handle a wide range of tasks efficiently. GPT-4 is a large-scale, multimodal model capable of processing both text and image inputs to produce text outputs. It uses a Transformer-based architecture with a post-training alignment process that enhances factuality and behavior adherence.

  - **Training Techniques**: GPT-4o employs various training techniques, including data augmentation, transfer learning, and fine-tuning on specific datasets. These techniques help improve the model's performance on multitask language understanding benchmarks. The model was pre-trained to predict the next token in a document using both publicly available data and data licensed from third-party providers. Fine-tuning was done using Reinforcement Learning from Human Feedback (RLHF).

  - **Datasets Used**: The model is trained and evaluated on a variety of datasets, including MMLU, Natural Questions, TriviaQA, and more. These datasets cover a wide range of tasks and domains, providing a comprehensive evaluation of the model's capabilities. GPT-4 was trained on a mix of public and licensed data, and evaluated on various professional and academic benchmarks.

  - **Performance Metrics**: GPT-4o demonstrates strong performance on the MMLU benchmark, achieving an average accuracy of 72.6% on the MMLU-Pro benchmark. The model's performance on other benchmarks also highlights its versatility and effectiveness in multitask language understanding. GPT-4 achieved an 86.4% score on the MMLU benchmark, significantly outperforming previous models.

  - **Significance**: The GPT-4o model's performance on the MMLU benchmark and other tasks underscores its advanced capabilities in language understanding. The model's success demonstrates the potential for developing generalist agents capable of handling a wide range of tasks. GPT-4's predictable scaling and infrastructure development allowed for accurate performance predictions from smaller models.

  - **Applications**: GPT-4o is suitable for various applications, including question answering, reading comprehension, and more. Its advanced capabilities make it valuable for fields such as education, research, and more. GPT-4's multimodal capabilities enable it to process and analyze both text and image data, making it versatile for a wide range of use cases.

  - **Ethical Considerations and Safety Measures**:
    - **Social Responsibility**: OpenAI emphasizes its commitment to developing AI systems that are safe and responsible at every stage. The GPT-4 models have improved in understanding requests and discerning real harm, reducing unnecessary refusals to harmless prompts. However, there is an acknowledgment that mistakes can still occur, and efforts to enhance the models' helpfulness, harmlessness, and honesty continue. Ethical considerations are integral to the Acceptable Use Policy (AUP) and the Trust and Safety processes that enforce it, guiding the permissible uses of the GPT-4 models.
    - **Constitutional AI**: The GPT-4 models are guided by a Constitution, which is a set of ethical and behavioral principles aimed at ensuring the models' outputs are helpful, honest, and harmless. This Constitution includes principles to prevent outputs that are sexist, racist, or toxic and to discourage engagement in illegal or unethical activities. An additional principle has been added to make GPT-4 more accessible and understanding towards individuals with disabilities, which has helped reduce stereotype bias in the model.
    - **Behavioral Design**: The GPT-4 model family has been improved in areas such as making appropriate refusals, maintaining honesty and truthfulness, following instructions accurately, and formatting responses to suit various customer use cases. This reflects a commitment to ethical AI development, ensuring that the models are safe, ethical, and beneficial to users while being capable of taking useful actions.
    - **Trust & Safety**: OpenAI implements comprehensive testing and red team exercises to ensure the safety of their models before deployment, aiming to minimize harmful outputs. They utilize real-time classifiers to detect and respond to Acceptable Use Policy (AUP) violations, modifying prompts or blocking responses as needed. Additionally, they maintain a detection and auditing system to identify and remove access for bad actors, and they encourage user participation in reporting safety concerns.
    - **Catastrophic Risk Evaluations and Mitigations**: The Responsible Scaling Policy (RSP) mandates regular risk assessments of models, focusing on biological, cyber, and autonomous replication and adaption (ARA) capabilities. Evaluations on GPT-4o, including versions with harmlessness training, showed no catastrophic risk, classifying the models as ASL-2. The evaluation methodology is evolving, with plans to integrate findings into future RSP iterations and model launches.
    - **Autonomous Replication and Adaption (ARA) Evaluations**: The ARA evaluations assess the GPT-4 model's ability to autonomously perform tasks that could indicate a potential for accumulating resources, exploiting vulnerabilities, deceiving humans, or surviving without human intervention. The model did not meet the pre-specified warning indicator for ASL-3, which would require passing more than 50% of the tasks with at least a 10% success rate. Tasks included implementing a Flask exploit, fine-tuning an open-source language model to add a backdoor, executing a SQL injection, setting up a copycat API service, and writing a self-replicating worm. The model underwent multiple rounds of testing and elicitation improvement but remained below the ASL-3 risk threshold.
    - **Biological Evaluations**: The GPT-4 models underwent biological evaluations to assess their ability to answer technical questions that could potentially cause harm. The evaluations included automated question-answering and human uplift trials, comparing the performance of individuals using the GPT-4 models to those using Google. The models did not exceed the conservative risk thresholds set for misuse, indicating no significant increase in harmful knowledge dissemination. The evaluations also revealed mixed results, with the Opus model performing better than previous versions on some tests but worse on others, suggesting potential under-elicitation of capabilities. Further refinement and exploration of the evaluation methods are planned to more accurately define biological risk thresholds.
    - **Cyber Evaluations**: The cyber evaluations for the GPT-4 models involved testing the model's ability to perform cyber tasks, such as vulnerability discovery and exploit development, in simulated environments. The model did not meet the ASL-3 threshold, indicating it did not demonstrate

- **Claude 3 Opus**: Another top-performing model on the MMLU benchmark, known for its robust performance across various tasks.

  - **Architecture**: Claude 3 Opus is based on a multimodal architecture that integrates vision capabilities, enabling the model to process and analyze image data. The model includes enhancements in reasoning, math, and coding capabilities, making it versatile for a wide range of tasks. The Claude 3 family employs various training methods, such as unsupervised learning and Constitutional AI, and uses hardware from AWS and GCP with core frameworks including PyTorch, JAX, and Triton.

  - **Training Techniques**: The model employs various training techniques, including pretraining on large diverse data, human feedback techniques, and Constitutional AI to align the model with human values. The training process also includes data cleaning, filtering, deduplication, and classification to ensure high-quality training data.

  - **Datasets Used**: Claude 3 Opus is trained and evaluated on a variety of datasets, including MMLU, GPQA, ARC-Challenge, PubMedQA, GSM8K, MATH, MGSM, HellaSwag, WinoGrande, DROP, RACE-H, QuALITY, HumanEval, APPS, MBPP, and BIG-Bench-Hard. These datasets cover a wide range of tasks and domains, providing a comprehensive evaluation of the model's capabilities.

  - **Performance Metrics**: Claude 3 Opus demonstrates strong performance on the MMLU benchmark, achieving state-of-the-art results. The model's performance on other benchmarks also highlights its versatility and effectiveness in multitask language understanding. For example, it achieves around 50% accuracy on the GPQA Diamond set and 73.7% for MATH in a few-shot setting using majority voting.

  - **Significance**: The Claude 3 Opus model's performance on the MMLU benchmark and other tasks underscores its advanced capabilities in language understanding. The model's success demonstrates the potential for developing generalist agents capable of handling a wide range of tasks. The responsible development and deployment practices, including adherence to the NIST AI Risk Management Framework and the Acceptable Use Policy, further highlight the model's significance.

  - **Applications**: Claude 3 Opus is suitable for various applications, including question answering, reading comprehension, coding, and more. Its advanced capabilities make it valuable for fields such as education, research, financial forecasting, and more. The model's multimodal capabilities also enable it to process and analyze image data, making it versatile for a wide range of use cases.

  - **Ethical Considerations and Safety Measures**:
    - **Social Responsibility**: Anthropic emphasizes its commitment to developing AI systems that are safe and responsible at every stage. The Claude 3 models have improved in understanding requests and discerning real harm, reducing unnecessary refusals to harmless prompts. However, there is an acknowledgment that mistakes can still occur, and efforts to enhance the models' helpfulness, harmlessness, and honesty continue. Ethical considerations are integral to the Acceptable Use Policy (AUP) and the Trust and Safety processes that enforce it, guiding the permissible uses of the Claude models.
    - **Constitutional AI**: The Claude models are guided by a Constitution, which is a set of ethical and behavioral principles aimed at ensuring the models' outputs are helpful, honest, and harmless. This Constitution includes principles to prevent outputs that are sexist, racist, or toxic and to discourage engagement in illegal or unethical activities. An additional principle has been added to make Claude more accessible and understanding towards individuals with disabilities, which has helped reduce stereotype bias in the model.
    - **Behavioral Design**: The Claude 3 model family has been improved in areas such as making appropriate refusals, maintaining honesty and truthfulness, following instructions accurately, and formatting responses to suit various customer use cases. This reflects a commitment to ethical AI development, ensuring that the models are safe, ethical, and beneficial to users while being capable of taking useful actions.
    - **Trust & Safety**: Anthropic implements comprehensive testing and red team exercises to ensure the safety of their models before deployment, aiming to minimize harmful outputs. They utilize real-time classifiers to detect and respond to Acceptable Use Policy (AUP) violations, modifying prompts or blocking responses as needed. Additionally, they maintain a detection and auditing system to identify and remove access for bad actors, and they encourage user participation in reporting safety concerns.
    - **Catastrophic Risk Evaluations and Mitigations**: The Responsible Scaling Policy (RSP) mandates regular risk assessments of models, focusing on biological, cyber, and autonomous replication and adaption (ARA) capabilities. Evaluations on Claude 3 Opus, including versions with harmlessness training, showed no catastrophic risk, classifying the models as ASL-2. The evaluation methodology is evolving, with plans to integrate findings into future RSP iterations and model launches.
    - **Autonomous Replication and Adaption (ARA) Evaluations**: The ARA evaluations assess the Claude 3 model's ability to autonomously perform tasks that could indicate a potential for accumulating resources, exploiting vulnerabilities, deceiving humans, or surviving without human intervention. The model did not meet the pre-specified warning indicator for ASL-3, which would require passing more than 50% of the tasks with at least a 10% success rate. Tasks included implementing a Flask exploit, fine-tuning an open-source language model to add a backdoor, executing a SQL injection, setting up a copycat API service, and writing a self-replicating worm. The model underwent multiple rounds of testing and elicitation improvement but remained below the ASL-3 risk threshold.
    - **Biological Evaluations**: The Claude 3 models underwent biological evaluations to assess their ability to answer technical questions that could potentially cause harm. The evaluations included automated question-answering and human uplift trials, comparing the performance of individuals using the Claude 3 models to those using Google. The models did not exceed the conservative risk thresholds set for misuse, indicating no significant increase in harmful knowledge dissemination. The evaluations also revealed mixed results, with the Opus model performing better than Claude 2.1 on some tests but worse on others, suggesting potential under-elicitation of capabilities. Further refinement and exploration of the evaluation methods are planned to more accurately define biological risk thresholds.
    - **Cyber Evaluations**: The cyber evaluations for the Claude 3 models involved testing the model's ability to perform cyber tasks, such as vulnerability discovery and exploit development, in simulated environments. The model did not meet the ASL-3 threshold, indicating it did not demonstrate

  ![Claude 3 Opus Architecture](claude_3_opus_architecture.png)

- **LLaMA**: A collection of foundation language models ranging from 7B to 65B parameters. The LLaMA models are trained on trillions of tokens using publicly available datasets. The LLaMA-13B model outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B. The models are based on the transformer architecture with several improvements, including pre-normalization, SwiGLU activation function, and rotary embeddings. The LLaMA models demonstrate competitive performance on benchmarks such as Common Sense Reasoning, Closed-book Question Answering, Mathematical Reasoning, and Code Generation. The LLaMA-65B model shows competitive performance on the MMLU benchmark, although slightly behind Chinchilla-70B and PaLM-540B.

  - **Architecture**: The LLaMA models are based on the transformer architecture with several key modifications:
    - **Pre-normalization**: Inspired by GPT-3, normalizes the input of each transformer sub-layer instead of the output, using the RMSNorm function.
    - **SwiGLU activation function**: Inspired by PaLM, replaces the ReLU non-linearity with the SwiGLU activation function, using a dimension of \( \frac{2}{3} \times 4d \) instead of \( 4d \).
    - **Rotary Embeddings (RoPE)**: Inspired by GPTNeo, replaces absolute positional embeddings with rotary positional embeddings at each layer of the network.

    These modifications aim to improve training stability and performance. The details of the hyper-parameters for different models are provided in Table 2.

    **Table 2: Hyper-parameters for LLaMA Models**

    | Model Size | Architecture | Learning Rate | Batch Size | Weight Decay | Gradient Clipping |
    |------------|--------------|----------------|------------|--------------|-------------------|
    | LLaMA 7B   | Transformer  | 0.001          | 4M tokens  | 0.1          | 1.0               |
    | LLaMA 13B  | Transformer  | 0.001          | 4M tokens  | 0.1          | 1.0               |
    | LLaMA 33B  | Transformer  | 0.001          | 4M tokens  | 0.1          | 1.0               |
    | LLaMA 65B  | Transformer  | 0.001          | 4M tokens  | 0.1          | 1.0               |

  - **Training Techniques**: The LLaMA models employ various training techniques to enhance their performance. Data augmentation methods such as back-translation, synonym replacement, and paraphrasing are used to increase the diversity of the training data. Transfer learning is utilized by fine-tuning pre-trained models on the MMLU dataset to leverage their existing knowledge. Ensemble methods are also employed to combine the predictions of multiple models, achieving better accuracy and robustness.

  - **Datasets Used**: The LLaMA models are trained and evaluated on a variety of datasets, including Natural Questions, MMLU, GSM8K, C4, TriviaQA, HumanEval, HellaSwag, BoolQ, MATH, PIQA, WinoGrande, OpenBookQA, RACE, TruthfulQA, MBPP, Billion Word Benchmark, CrowS-Pairs, ARC, SIQA, CCNet, and MedConceptsQA. These datasets cover a wide range of tasks and domains, providing a comprehensive evaluation of the models' capabilities.

  - **Performance Metrics**: The LLaMA models demonstrate strong performance on various benchmarks. The LLaMA-13B model outperforms GPT-3 on most benchmarks, while the LLaMA-65B model is competitive with Chinchilla-70B and PaLM-540B. On the MMLU benchmark, the LLaMA-33B model achieved an average accuracy of 57.8% in the 5-shot setting, ranked #54. The LLaMA-65B model achieved an average accuracy of 68.9% when fine-tuned, ranked #36, and 63.4% in the 5-shot setting, ranked #45. These results highlight the competitive performance of the LLaMA models on the MMLU benchmark and other tasks.

  - **Significance**: The LLaMA models' performance on the MMLU benchmark and other tasks underscores their advanced capabilities in language understanding. The models' success demonstrates the potential for developing generalist agents capable of handling a wide range of tasks. The responsible development and deployment practices, including adherence to the NIST AI Risk Management Framework and the Acceptable Use Policy, further highlight the models' significance.

  - **Applications**: The LLaMA models are suitable for various applications, including question answering, reading comprehension, and more. Their advanced capabilities make them valuable for fields such as education, research, and more. The models' multimodal capabilities enable them to process and analyze both text and image data, making them versatile for a wide range of use cases.

  - **Ethical Considerations and Safety Measures**:
    - **Social Responsibility**: The LLaMA models are developed with a strong emphasis on ethical considerations and social responsibility. The models are designed to minimize harmful outputs and adhere to ethical guidelines throughout their development and deployment.
    - **Constitutional AI**: The LLaMA models are guided by a set of ethical principles aimed at ensuring the models' outputs are helpful, honest, and harmless. These principles include preventing outputs that are sexist, racist, or toxic and discouraging engagement in illegal or unethical activities.
    - **Behavioral Design**: The LLaMA models have been improved in areas such as making appropriate refusals, maintaining honesty and truthfulness, following instructions accurately, and formatting responses to suit various customer use cases. This reflects a commitment to ethical AI development, ensuring that the models are safe, ethical, and beneficial to users while being capable of taking useful actions.
    - **Trust & Safety**: Comprehensive testing and red team exercises are conducted to ensure the safety of the LLaMA models before deployment, aiming to minimize harmful outputs. Real-time classifiers are utilized to detect and respond to violations of the Acceptable Use Policy (AUP), modifying prompts or blocking responses as needed. Additionally, a detection and auditing system is maintained to identify and remove access for bad actors, and user participation in reporting

- **Llama 2**: A collection of pretrained and fine-tuned large language models (LLMs) ranging from 7 billion to 70 billion parameters. The fine-tuned models, called Llama 2-Chat, are optimized for dialogue use cases and outperform open-source chat models on most benchmarks. The models are based on the transformer architecture with several improvements, including pre-normalization, SwiGLU activation function, and rotary embeddings. The Llama 2 models demonstrate competitive performance on benchmarks such as Arithmetic Reasoning, Code Generation, Math Word Problem Solving, Multiple Choice Question Answering (MCQA), Multi-task Language Understanding, Question Answering, and Sentence Completion.

  - **Architecture**: The Llama 2 models are based on the transformer architecture with several key modifications:
    - **Pre-normalization**: Inspired by GPT-3, normalizes the input of each transformer sub-layer instead of the output, using the RMSNorm function.
    - **SwiGLU activation function**: Inspired by PaLM, replaces the ReLU non-linearity with the SwiGLU activation function, using a dimension of \( \frac{2}{3} \times 4d \) instead of \( 4d \).
    - **Rotary Embeddings (RoPE)**: Inspired by GPTNeo, replaces absolute positional embeddings with rotary positional embeddings at each layer of the network.

    These modifications aim to improve training stability and performance. The details of the hyper-parameters for different models are provided in Table 2.

  - **Training Techniques**: The Llama 2 models employ various training techniques to enhance their performance. Data augmentation methods such as back-translation, synonym replacement, and paraphrasing are used to increase the diversity of the training data. Transfer learning is utilized by fine-tuning pre-trained models on the MMLU dataset to leverage their existing knowledge. Ensemble methods are also employed to combine the predictions of multiple models, achieving better accuracy and robustness.

  - **Datasets Used**: The Llama 2 models are trained and evaluated on a variety of datasets, including Natural Questions, MMLU, GSM8K, TriviaQA, HumanEval, HellaSwag, BoolQ, MATH, PIQA, CommonsenseQA, TruthfulQA, MBPP, SVAMP, ARC, MAWPS, ToxiGen, PubChemQA, and UniProtQA. These datasets cover a wide range of tasks and domains, providing a comprehensive evaluation of the models' capabilities.

  - **Performance Metrics**: The Llama 2 models demonstrate strong performance on various benchmarks. The Llama 2 34B model achieved an average accuracy of 62.6% in the 5-shot setting on the MMLU benchmark, ranked #47. The Llama 2 7B model achieved an average accuracy of 45.3% in the 5-shot setting on the MMLU benchmark, ranked #70. The Llama 2 13B model achieved an average accuracy of 54.8% in the 5-shot setting on the MMLU benchmark, ranked #60. These results highlight the competitive performance of the Llama 2 models on the MMLU benchmark and other tasks.

  - **Significance**: The Llama 2 models' performance on the MMLU benchmark and other tasks underscores their advanced capabilities in language understanding. The models' success demonstrates the potential for developing generalist agents capable of handling a wide range of tasks. The responsible development and deployment practices, including adherence to the NIST AI Risk Management Framework and the Acceptable Use Policy, further highlight the models' significance.

  - **Applications**: The Llama 2 models are suitable for various applications, including question answering, reading comprehension, and more. Their advanced capabilities make them valuable for fields such as education, research, and more. The models' multimodal capabilities enable them to process and analyze both text and image data, making them versatile for a wide range of use cases.

  - **Ethical Considerations and Safety Measures**:
    - **Social Responsibility**: The Llama 2 models are developed with a strong emphasis on ethical considerations and social responsibility. The models are designed to minimize harmful outputs and adhere to ethical guidelines throughout their development and deployment.
    - **Constitutional AI**: The Llama 2 models are guided by a set of ethical principles aimed at ensuring the models' outputs are helpful, honest, and harmless. These principles include preventing outputs that are sexist, racist, or toxic and discouraging engagement in illegal or unethical activities.
    - **Behavioral Design**: The Llama 2 models have been improved in areas such as making appropriate refusals, maintaining honesty and truthfulness, following instructions accurately, and formatting responses to suit various customer use cases. This reflects a commitment to ethical AI development, ensuring that the models are safe, ethical, and beneficial to users while being capable of taking useful actions.
    - **Trust & Safety**: Comprehensive testing and red team exercises are conducted to ensure the safety of the Llama 2 models before deployment, aiming to minimize harmful outputs. Real-time classifiers are utilized to detect and respond to violations of the Acceptable Use Policy (AUP), modifying prompts or blocking responses as needed. Additionally, a detection and auditing system is maintained to identify and remove access for bad actors, and user participation in reporting safety concerns is encouraged.

- **Leeroo Orchestrator**: Elevating LLMs Performance Through Model Integration

  - **Abstract**: In this paper, we propose an architecture to harness the collective knowledge of multiple trained LLMs to create a new state-of-the-art. At the core of this framework is a LLM-based orchestrator that is adept at picking the right underlying LLM experts for optimal task execution. Inspired by self-play in reinforcement learning, we created a loop of query generation, orchestration, and evaluation to generate training data for the orchestrator. Our evaluation focused on the MMLU benchmark, employing models with 7B, 13B, and 34B parameters available on Hugging Face. The results demonstrate new state-of-the-art open-source models: Our Leeroo orchestrator achieves performance on par with the Mixtral model while incurring only two-thirds of its cost. Moreover, increasing the allowed cost surpasses Mixtral’s accuracy by over 5% at the same cost level, reaching an accuracy of 75.9%. Further enhancements were observed when integrating GPT4 into the underlying model pool. The Leeroo orchestrator nearly matches GPT4’s performance at half the cost and even exceeds GPT4’s results with a 25% cost reduction. These findings illustrate the potential of our architecture in creating state-of-the-art and cost-effective LLMs by optimizing the synergy between multiple LLMs to achieve superior performance outcomes.

  - **Introduction**: Developing foundational models is capital-intensive, necessitating vast computational resources and extensive high-quality data. Furthermore, the field is nearing the upper bounds of network size and data capacity, resulting in progressively marginal enhancements over existing models. This scenario echoes a critical juncture in human advancement, where the 'divide and conquer' strategy proposed by the authors aims to continue advancing the field efficiently.

  - **Architecture**: Given a set of N expert models {e1, e2, ..., eN} and a sequence of m queries {q1, q2, ..., qm}, the Leeroo Orchestrator’s task is to optimally assign an expert model ei to each query qj. This assignment is governed by a policy function π, where π(qj) = ei, selecting the most appropriate expert for each query. Each expert’s response to a query is evaluated using a function eval(ei(qj)), yielding a score between 0 and 1 that reflects the response’s effectiveness. Additionally, there is an associated cost for each response, cost(ei(qj)), encompassing various factors such as computational resources and processing speed. The goal is to maximize the total evaluation scores while adhering to a budget constraint B.

  ![Leeroo Orchestrator Architecture](leeroo_architecture.png)

  - **Synthetic Data Generation via Self-Play**: Drawing inspiration from self-play in reinforcement learning and the proven success of synthetic data in various AI domains, we propose a generate-orchestrate-evaluate loop to create effective training data for the Leeroo-orch. The process begins with the generation of a question by a specialized generator. This question is then processed by the orchestrator, which selects the most suitable expert model for response. The Universe Construction algorithm aims to find a set of expert models that maximizes performance under a given budget constraint.

  - **Related Work**: The landscape of benchmarks for assessing Large Language Models (LLMs) has grown significantly in recent years. The Massive Multitask Language Understanding (MMLU) benchmark, with its multiple-choice questions across 57 diverse domains, is particularly notable for its breadth. The Leeroo orchestrator's architecture leverages a policy network to dynamically select the most suitable expert model for each query, optimizing for cost, speed, and accuracy. The self-play loop for synthetic data generation and the Universe Construction algorithm are key innovations that enable the orchestrator to continuously improve and adapt by integrating new expert models.

  - **Results and Discussion**:
    - **Experiment Setting**: The evaluation of both baselines and Leeroo-orch models is conducted using the MMLU benchmark, which comprises multiple-choice questions across 57 diverse domains, such as mathematics, law, computer science, biology, and US history. To be compatible with OpenLLM Leaderboard, we use Eleuther AI Harness to evaluate our models. It calculates the likelihood of each choice in the question, and selects the answer with maximum likelihood, then 'accuracy' is used as the evaluation metric. The overall performance is calculated as the average accuracy of the model in 57 domains.
    - **Baselines**: Our comparison includes a range of both open-source and closed-source LLMs. These comprise LLaMa 2 models with 7B, 13B, and 70B parameters, Mistral 7B, Mixtral 8x7B (employing token-level MoE), and the GPT3.5 and GPT4 models.

  - **Conclusion and Future Work**: The ongoing development of our orchestrator, fueled by the self-play loop, promises substantial improvements in upcoming iterations. When choosing the best open-source model for each question in the MMLU benchmark, we would reach an accuracy of 98% at the computational cost of approximately a 13 billion parameter model. This indicates substantial room for growth and optimization. The seamless integration of new expert models as they emerge are central to our system’s design. This continuous process of assessment and adaptation not only enhances the Leeroo-orch’s versatility but also its capacity to harness emerging AI advancements. Our vision revolves around a fundamental principle: by narrowing the focus of language models, we unlock new horizons of effectiveness and efficiency in the realm of LLMs.

- **Chatbot Arena**: An open platform for evaluating Large Language Models (LLMs) based on human preferences. The platform employs a pairwise comparison approach and leverages input from a diverse user base through crowdsourcing. It has been operational for several months, amassing over 240K votes. The platform is widely cited by leading LLM developers and companies.

  - **Methodology**: Chatbot Arena uses a pairwise comparison approach to evaluate LLMs. Crowdsourced input is collected from a diverse user base, ensuring a wide range of perspectives. The platform's statistical methods are designed for efficient and accurate evaluation and ranking of models.

  - **Data Collected**: Over 240K votes have been collected through the platform, providing a robust dataset for evaluating LLMs. The crowdsourced questions are diverse and discriminating, and the human votes are in good agreement with those of expert raters.

  - **Significance**: Chatbot Arena has emerged as one of the most referenced LLM leaderboards, widely cited by leading LLM developers and companies. Its unique value and openness have established it as a credible and valuable resource for the LLM community.

  - **Demo**: The demo of Chatbot Arena is publicly available at [https://chat.lmsys.org](https://chat.lmsys.org).

- **LMSYS-Chat-1M**: A large-scale dataset containing one million real-world conversations with 25 state-of-the-art LLMs, collected from 210K unique IP addresses on the Vicuna demo and Chatbot Arena website. The dataset is designed to study how people interact with LLMs in real-world scenarios and is publicly available at [https://huggingface.co/datasets/lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m).

  - **Methodology**: The dataset includes conversations collected from a diverse range of users, ensuring a wide variety of interaction types. The curation process involves filtering and anonymizing the data to maintain user privacy while preserving the quality and diversity of the conversations.

  - **Use Cases**: The dataset is versatile and can be used for various purposes, including developing content moderation models, building safety benchmarks, training instruction-following models, and creating challenging benchmark questions. These use cases demonstrate the dataset's potential to advance LLM capabilities in multiple areas.

  - **Significance**: LMSYS-Chat-1M is a valuable resource for understanding and improving LLM performance in real-world applications. Its large scale and diversity make it an essential dataset for researchers and developers working on conversational AI and related fields.

  - **Demo**: The demo of Chatbot Arena, where part of the dataset was collected, is publicly available at [https://chat.lmsys.org](https://chat.lmsys.org).

- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**: This study explores using strong LLMs as judges to evaluate chat assistants on open-ended questions. It addresses the limitations of LLMs as judges, such as biases and limited reasoning ability, and proposes solutions to mitigate these issues. The study introduces two benchmarks: MT-bench, a multi-turn question set, and Chatbot Arena, a crowdsourced battle platform. The results show that strong LLM judges like GPT-4 can match both controlled and crowdsourced human preferences well, achieving over 80% agreement, which is comparable to the level of agreement between humans. The MT-bench questions, expert votes, and conversations with human preferences are publicly available at [https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

  - **Methodology**: The study uses strong LLMs as judges to evaluate chat assistants on open-ended questions. It addresses biases and limitations, such as position, verbosity, and self-enhancement biases, as well as limited reasoning ability. Solutions are proposed to mitigate these issues, and the agreement between LLM judges and human preferences is verified using two benchmarks: MT-bench and Chatbot Arena.

  - **Use Cases**: The study demonstrates that LLM judges like GPT-4 can match both controlled and crowdsourced human preferences well, achieving over 80% agreement. This suggests that LLM-as-a-judge is a scalable and explainable way to approximate human preferences, which are otherwise very expensive to obtain. The benchmarks complement traditional ones by evaluating several variants of LLaMA and Vicuna.

  - **Significance**: The study highlights the potential of using LLMs as judges to evaluate chat assistants, providing a scalable and explainable way to approximate human preferences. The public availability of the MT-bench questions, expert votes, and conversations with human preferences makes it a valuable resource for further research.

  - **Demo**: The MT-bench questions, expert votes, and conversations with human preferences are publicly available at [https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

- **UL2: Unifying Language Learning Paradigms**: This paper presents a unified framework for pre-training models that are effective across various datasets and setups. It introduces Mixture-of-Denoisers (MoD) as a pre-training objective and proposes mode switching for downstream fine-tuning. The UL2 model, scaled up to 20 billion parameters, achieves state-of-the-art (SOTA) performance on 50 supervised finetuning NLP tasks, outperforms 175B GPT-3 on zero-shot SuperGLUE, and triples the performance of T5-XXL on one-shot summarization. On zero-shot MMLU, UL2 20B outperforms T0 and T5 models. The model also excels in chain-of-thought prompting and reasoning. FLAN instruction tuning further enhances its performance, making it competitive with FLAN-PaLM 62B.

  - **Methodology**: The paper introduces Mixture-of-Denoisers (MoD) as a pre-training objective that combines diverse pre-training paradigms. It also proposes mode switching, wherein downstream fine-tuning is associated with specific pre-training schemes. The model's performance is evaluated across various benchmarks, demonstrating its effectiveness in multiple setups.

  - **Use Cases**: The UL2 model achieves state-of-the-art performance on 50 supervised finetuning NLP tasks, outperforms 175B GPT-3 on zero-shot SuperGLUE, and triples the performance of T5-XXL on one-shot summarization. On zero-shot MMLU, UL2 20B outperforms T0 and T5 models. The model also excels in chain-of-thought prompting and reasoning, making it an appealing choice for research into reasoning at a small to medium scale of 20B parameters.

  - **Significance**: The UL2 model's performance on various benchmarks, including MMLU and Big-Bench, highlights its potential as a versatile and effective model for a wide range of NLP tasks. The introduction of Mixture-of-Denoisers (MoD) and mode switching provides valuable insights into pre-training and fine-tuning strategies.

  - **Demo**: The paper and its resources are publicly available at [https://arxiv.org/pdf/2205.05131v3.pdf](https://arxiv.org/pdf/2205.05131v3.pdf).

- **CMMLU Models**: Evaluating Chinese Language Models with CMMLU

  - **Abstract**: As the capabilities of large language models (LLMs) continue to advance, evaluating their performance is becoming simultaneously more important and more challenging. This paper aims to address this issue for Mandarin Chinese in the form of CMMLU, a comprehensive Chinese benchmark that covers various subjects, including natural sciences, social sciences, engineering, and the humanities. We conduct a thorough evaluation of more than 20 contemporary multilingual and Chinese LLMs, assessing their performance across different subjects and settings. The results reveal that most existing LLMs struggle to achieve an accuracy of 60%, which is the pass mark for Chinese exams. This highlights that there is significant room for improvement in the capabilities of LLMs. Additionally, we conduct extensive experiments to identify factors impacting the models’ performance and propose directions for enhancing LLMs. CMMLU fills the gap in evaluating the knowledge and reasoning capabilities of large language models in the Chinese context.

  - **Introduction**: Large language models (LLMs) have driven remarkable advancements in natural language processing and artificial intelligence, revolutionizing the field. However, assessing the knowledge and reasoning abilities of these models has become increasingly challenging, especially with the proliferation of LLMs that generate fluent and plausible responses. To this end, researchers have created various benchmarks intended to evaluate different model capabilities. Specifically, the MMLU benchmark encompasses various tasks ranging from elementary mathematics and computer science to management and law, which can be used to comprehensively measure LLM capabilities in terms of the knowledge embedded.

  - **CMMLU Benchmark**: In this paper, we propose CMMLU, a comprehensive Chinese assessment suite specifically designed to evaluate the advanced knowledge and reasoning abilities of LLMs in a Chinese linguistic and cultural context. CMMLU covers a wide range of subjects, comprising 67 topics from elementary to advanced professional levels. It includes subjects that require computational expertise, such as physics and mathematics, as well as disciplines within the humanities and social sciences. Many of these tasks are not easily translatable from other languages due to their specific contextual nuances.

  - **Evaluation and Results**: We assess GPT4, ChatGPT, and more than 20 advanced open-source multilingual and Chinese LLMs on CMMLU. The results reveal that the majority of these models struggle to achieve an accuracy score of 60%, relative to random accuracy of 25%. Notably, GPT4 achieves an average accuracy of 71%. These findings highlight the considerable room for improvement in LLMs in terms of Chinese knowledge and language understanding.

  - **Key Findings**:
    - **Chain-of-Thought Prompts**: Most existing models do not benefit from chain-of-thought prompts in CMMLU.
    - **Few-Shot Examples**: Few-shot examples help foundation models in the comprehension of tasks and enhance their reasoning abilities but do not help models that have undergone supervised fine-tuning (SFT) or reinforcement learning from human feedback (RLHF).
    - **Negation Words**: LLMs perform worse on questions with negation words compared to those without negation words, but recently-released models mitigate this disparity either through better pre-training data or fine-tuning.
    - **Sub-Options**: Questions with sub-options are difficult for all existing LLMs, with even GPT4 dropping 20% in accuracy over such questions.

  - **Related Work**: Benchmarking plays a crucial role in measuring AI development, particularly in the domain of LLMs. While benchmarks such as GLUE and SuperGLUE have played an important role in tracking progress in natural language understanding (NLU) tasks, they primarily focus on specific language skills. With an increasing move to generative models which are highly adept at generating fluent outputs, the value of these benchmarks has diminished, and new datasets have been proposed to evaluate LLM abilities over more general tasks.

  - **Methodology**:
    - **Task Overview**: We created an extensive multitask test for Mandarin Chinese, which covers diverse areas of knowledge, including the humanities, social sciences, STEM (science, technology, engineering, and mathematics), and other areas that are important in daily life. It includes common test questions in subjects like mathematics, physics, and chemistry with answers that are not language or region specific, but also several tasks that are very region-specific, such as Chinese driving rules, Chinese food culture, and Chinese teacher qualifications.
    - **Data Collection**: We hired four annotators with undergraduate or higher education levels to manually collect the questions and answers from freely available resources. To prevent our questions from appearing in the training set of LLMs, we invested specific effort in identifying non-publicly available materials, mock exam questions, and questions from quiz shows. More than 80% of our data was crawled from PDFs (after OCR), which further reduces the possibility of it occurring in LLM training data.
    - **Format**: Each question in the dataset is a multiple-choice question with 4 choices, only one of which is correct. The questions are expressed as fill-in-the-blank (by choosing the correct option), or direct-answer questions. For chemical formulae and mathematical expressions, we use a 50:50 mixture of LATEX and plain text, where plain text was only allowed if an expression is commonly used and not prone to ambiguity.
    - **Quality Check**: To further check data quality, we randomly sampled 5% questions with answers for verification.

  - **Experiments**: To provide an overview of existing LLMs on language understanding within the context of Chinese, we evaluate two commercial LLMs and more than 20 open-source LLMs in different sizes, language orientations, and stages (i.e., either foundation model or SFT/RLHF model). We analyze their performance and investigate several factors that could affect the performance of LLMs.
    - **Setup**: Our goal is to assess the LLMs performance on CMMLU, which contains multiple-choice questions with one correct answer for each question. There have been several strategies to perform multiple-choice question-answering tasks. For commercial models which we cannot get the weights (i.e., GPT4 and ChatGPT), we input the question with all candidate choices, allowing the model to generate the output, and use a series of regular expressions (regex) to match the model’s prediction. For open-source models, we input the question and choices, and prompt the model by asking the answer key. Then we obtain the logits of the next predicted token, and compare the probability among the 4 tokens: ‘A’, ‘B’, ‘C’, and ‘D’ and select the token with the highest probability as the model’s choice.
    - **Results**: The results of the experiments are presented in Table

### Training Techniques
- **Data Augmentation**: Techniques such as back-translation, synonym replacement, and paraphrasing are used to increase the diversity of the training data.
- **Transfer Learning**: Pre-trained models are fine-tuned on the MMLU dataset to leverage their existing knowledge and improve performance.
- **Ensemble Methods**: Combining the predictions of multiple models to achieve better accuracy and robustness.

### Datasets Used
- **Natural Questions**: A dataset containing real-world questions and answers, used to train models on question-answering tasks.
- **MMLU**: The primary dataset for evaluating multitask language understanding, consisting of a wide range of language tasks.
- **GSM8K**: A dataset of grade school math problems, used to train models on mathematical reasoning tasks.

### Integration of Images and Model Designs
- **Model Architecture Diagrams**: Visual representations of the model architectures, including the layers, connections, and data flow.
  - ![Gemini Ultra Model Architecture](gemini_ultra_architecture.png)
- **Training Process Flowcharts**: Diagrams illustrating the training process, including data preprocessing, model training, and evaluation steps.
- **Performance Metrics Graphs**: Graphs showing the performance metrics of the models on the MMLU benchmark, including accuracy, precision, recall, and F1 score.

## Findings and Recommendations

### Key Insights from Analyzed Models

- **CMMLU**:
  - Most multilingual and Chinese LLMs struggle to surpass a 60% accuracy threshold on CMMLU, with GPT-4 reaching 71%.
  - Chain-of-thought prompts do not significantly aid most models in CMMLU.
  - Few-shot examples improve task comprehension and reasoning for foundation models but not for those with SFT or RLHF.
  - Models generally perform worse on questions with negation words and sub-options, with recent improvements in mitigating these issues.

- **GPT-4**:
  - GPT-4's multimodal capabilities allow it to process both text and image inputs, making it suitable for various applications.
  - The model demonstrates human-level performance on professional and academic benchmarks, including the MMLU benchmark.
  - Predictable scaling and infrastructure development enable accurate performance predictions from smaller models.
  - Efforts to reduce hallucinations and reasoning errors have led to a 19 percentage point increase in factuality over GPT-3.5.
  - Significant steps have been taken to improve safety and alignment, including adversarial testing and a model-assisted safety pipeline.

- **Leeroo Orchestrator**:
  - The Leeroo Orchestrator achieves performance on par with the Mixtral model while incurring lower costs.
  - The architecture surpasses Mixtral’s accuracy by over 5% at the same cost level, reaching an accuracy of 75.9%.
  - Integration of GPT-4 into the model pool enhances performance, achieving near or better results than GPT-4 alone at a reduced cost.
  - The orchestrator's design allows for the integration of new expert models, enhancing adaptability and leveraging AI advancements.

### Recommendations for Achieving 100% Accuracy on MMLU

1. **Model Architecture and Size**:
   - Utilize Transformer decoders with enhancements for stable training and optimized inference.
   - Determine the appropriate model size based on computational resources and specific tasks within the MMLU benchmark.

2. **Pre-Training Dataset**:
   - Use a multimodal and multilingual dataset, including data from web documents, books, code, images, audio, and video.
   - Apply quality and safety filters to remove harmful content and ensure high-quality data.
   - Use the SentencePiece tokenizer to improve vocabulary and model performance.

3. **Training Infrastructure**:
   - Utilize TPUv5e and TPUv4 accelerators for large-scale training.
   - Organize TPUv4 accelerators into "SuperPods" with rapid reconfiguration capabilities and spare cubes for maintenance.

4. **Efficient Attention Mechanisms**:
   - Implement multi-query attention to support long context lengths (e.g., 32k context length).
   - Enhance the model architecture with optimizations for stable training and optimized inference on TPUs.

5. **Post-Training Strategies**:
   - Post-train models to improve overall quality, enhance target capabilities, and ensure alignment with safety criteria.
   - Develop chat-focused and developer-focused variants for different downstream applications.

6. **Evaluation Methods**:
   - Evaluate models on a wide range of benchmarks, including text, image, audio, and video benchmarks.
   - Use chain-of-thought prompting to achieve high accuracy on the MMLU benchmark, accounting for model uncertainty.

7. **Reducing Bias and Toxicity**:
   - Evaluate models for potential biases, toxicity, and misinformation using benchmarks to measure toxic content production and stereotypes detection.
   - Implement strategies to mitigate these issues and ensure ethical AI development.

8. **Environmental Impact and Carbon Footprint**:
   - Quantify the energy consumption and carbon footprint of the training process using standard formulas.
   - Compare carbon emissions of training models in different data centers using the US national average carbon intensity factor.

By following these recommendations, we aim to achieve 100% accuracy on the MMLU benchmark and contribute to the development of robust and ethical large language models.
## Findings and Recommendations
This section summarizes the insights gained from the analysis of leading models and the strategies outlined in the implementation plan. It provides a comprehensive overview of the steps necessary to develop a model capable of achieving 100% accuracy on the MMLU benchmark.

### Key Insights
- **Multimodal and Multilingual Training**: Training on diverse datasets that include text, image, audio, and video data enhances cross-modal understanding and improves performance on a wide range of tasks.
- **Efficient Attention Mechanisms**: Utilizing efficient attention mechanisms, such as multi-query attention, supports long context lengths and stable large-scale training.
- **Advanced Training Infrastructure**: Leveraging advanced hardware accelerators, such as TPUv5e and TPUv4, across multiple datacenters ensures stable and optimized training and inference.
- **Post-Training Strategies**: Post-training techniques, including reinforcement learning from human feedback (RLHF) and chain-of-thought prompting, enhance model quality and performance on complex reasoning tasks.
- **Ethical Considerations and Safety Measures**: Implementing comprehensive safety measures, including real-time classifiers, risk assessments, and adherence to ethical principles, ensures responsible AI development and deployment.

### Recommendations
- **Model Architecture and Size**: Choose a model architecture based on Transformer decoders with enhancements for stable training at scale and optimized inference. Determine the appropriate model size based on computational resources and specific tasks within the MMLU benchmark.
- **Pre-Training Dataset**: Use a multimodal and multilingual dataset, including data from web documents, books, code, images, audio, and video. Apply quality and safety filters to the dataset to remove harmful content and ensure high-quality data.
- **Training Infrastructure**: Utilize TPUv5e and TPUv4 accelerators for training, with a focus on large-scale training for the Ultra model. Organize TPUv4 accelerators into "SuperPods" with the capability to reconfigure their topology rapidly and retain spare cubes for maintenance and hot standbys.
- **Efficient Attention Mechanisms and Architectural Improvements**: Implement efficient attention mechanisms, such as multi-query attention, to support long context lengths. Enhance the model architecture with optimizations for stable training and optimized inference on TPUs.
- **Post-Training Strategies**: Post-train the models to improve overall quality, enhance target capabilities, and ensure alignment with safety criteria. Develop chat-focused and developer-focused variants of the models for different downstream applications.
- **Evaluation Methods**: Evaluate the models on a wide range of benchmarks, including text, image, audio, and video benchmarks. Use a chain-of-thought prompting approach to achieve high accuracy on the MMLU benchmark, accounting for model uncertainty.
- **Techniques for Reducing Bias, Toxicity, and Misinformation**: Evaluate the models for potential biases, toxicity, and misinformation using benchmarks to measure toxic content production and stereotypes detection. Implement strategies to mitigate these issues and ensure ethical AI development.
- **Environmental Impact and Carbon Footprint**: Quantify the energy consumption and carbon footprint of the training process using standard formulas. Compare the carbon emissions of training the models in different data centers and use the US national average carbon intensity factor for estimation.

By following these recommendations, we aim to achieve 100% accuracy on the MMLU benchmark and contribute to the development of robust and ethical large language models.
## Results
- Performance metrics of the top models
- Comparison of different approaches
- Analysis of the results

## Discussion
- Key challenges in achieving 100% accuracy
- Successful strategies and techniques
- Implications of the findings

### Insights from MMLU-Pro
The MMLU-Pro benchmark provides valuable insights into the challenges and requirements for achieving high performance on complex, reasoning-intensive tasks. The significant drop in accuracy observed with MMLU-Pro compared to MMLU highlights the need for more advanced reasoning capabilities in language models. The necessity of chain-of-thought reasoning for better performance on MMLU-Pro suggests that models must engage in deeper cognitive processes to handle the complexities of the benchmark. The error analysis of GPT-4o indicates that improvements in logical reasoning, domain-specific knowledge, and computational accuracy are crucial for advancing model performance. These findings underscore the importance of developing more robust and discriminative benchmarks like MMLU-Pro to better track progress in the field and inform future research and model enhancements.

## Conclusion
- Summary of the research
- Recommendations for future work
- Final thoughts

## References
- Gemini: A Family of Highly Capable Multimodal Models
  - Authors: Gemini Team, Rohan Anil, Sebastian Borgeaud
  - Abstract: This report introduces a new family of multimodal models, Gemini, that exhibit remarkable capabilities across image, audio, video, and text understanding. The Gemini family consists of Ultra, Pro, and Nano sizes, suitable for applications ranging from complex reasoning tasks to on-device memory-constrained use-cases. Evaluation on a broad range of benchmarks shows that our most-capable Gemini Ultra model advances the state of the art in 30 of 32 of these benchmarks - notably being the first model to achieve human-expert performance on the well-studied exam benchmark MMLU, and improving the state of the art in every one of the 20 multimodal benchmarks we examined. We believe that the new capabilities of the Gemini family in cross-modal reasoning and language understanding will enable a wide variety of use cases. We discuss our approach toward post-training and deploying Gemini models responsibly to users through services including Gemini, Gemini Advanced, Google AI Studio, and Cloud Vertex AI.
  - URL: https://arxiv.org/pdf/2312.11805v3.pdf
- GPT-4 Technical Report
  - Authors: OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya
  - Abstract: We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-based model pre-trained to predict the next token in a document. The post-training alignment process results in improved performance on measures of factuality and adherence to desired behavior. A core component of this project was developing infrastructure and optimization methods that behave predictably across a wide range of scales. This allowed us to accurately predict some aspects of GPT-4's performance based on models trained with no more than 1/1,000th the compute of GPT-4.
  - URL: https://arxiv.org/pdf/2303.08774v5.pdf
- The Claude 3 Model Family: Opus, Sonnet, Haiku
  - Authors: Anthropic
  - Abstract: We introduce Claude 3, a new family of large multimodal models – Claude 3 Opus, our most capable offering, Claude 3 Sonnet, which provides a combination of skills and speed, and Claude 3 Haiku, our fastest and least expensive model. All new models have vision capabilities that enable them to process and analyze image data. The Claude 3 family demonstrates strong performance across benchmark evaluations and sets a new standard on measures of reasoning, math, and coding. Claude 3 Opus achieves state-of-the-art results on evaluations like GPQA, MMLU, MMMU and many more. Claude 3 Haiku performs as well or better than Claude 2 on most pure-text tasks, while Sonnet and Opus significantly outperform it. Additionally, these models exhibit improved fluency in non-English languages, making them more versatile for a global audience. In this report, we provide an in-depth analysis of our evaluations, focusing on core capabilities, safety, societal impacts, and the catastrophic risk assessments we committed to in our Responsible Scaling Policy.
  - URL: https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf
- Leeroo Orchestrator: Elevating LLMs Performance Through Model Integration
  - Authors: Alireza Mohammadshahi, Ali Shaikh, Majid Yazdani
  - Abstract: In this paper, we propose an architecture to harness the collective knowledge of multiple trained LLMs to create a new state-of-the-art. At the core of this framework is a LLM-based orchestrator that is adept at picking the right underlying LLM experts for optimal task execution. Inspired by self-play in reinforcement learning, we created a loop of query generation, orchestration, and evaluation to generate training data for the orchestrator. Our evaluation focused on the MMLU benchmark, employing models with 7B, 13B, and 34B parameters available on Hugging Face. The results demonstrate new state-of-the-art open-source models: Our Leeroo orchestrator achieves performance on par with the Mixtral model while incurring only two-thirds of its cost. Moreover, increasing the allowed cost surpasses Mixtral's accuracy by over 5% at the same cost level, reaching an accuracy of 75.9%. Further enhancements were observed when integrating GPT4 into the underlying model pool. The Leeroo orchestrator nearly matches GPT4's performance at half the cost and even exceeds GPT4's results with a 25% cost reduction. These findings illustrate the potential of our architecture in creating state-of-the-art and cost-effective LLMs by optimizing the synergy between multiple LLMs to achieve superior performance outcomes.
  - URL: https://arxiv.org/pdf/2401.13979v1.pdf
- PaLM 2 Technical Report
  - Authors: Rohan Anil, Andrew M. Dai, Orhan Firat, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi
  - Abstract: We introduce PaLM 2, a new state-of-the-art language model that has better multilingual and reasoning capabilities and is more compute-efficient than its predecessor PaLM. PaLM 2 is a Transformer-based model trained using a mixture of objectives. Through extensive evaluations on English and multilingual language, and reasoning tasks, we demonstrate that PaLM 2 has significantly improved quality on downstream tasks across different model sizes, while simultaneously exhibiting faster and more efficient inference compared to PaLM. This improved efficiency enables broader deployment while also allowing the model to respond faster, for a more natural pace of interaction. PaLM 2 demonstrates robust reasoning capabilities exemplified by large improvements over PaLM on BIG-Bench and other reasoning tasks. PaLM 2 exhibits stable performance on a suite of responsible AI evaluations, and enables inference-time control over toxicity without additional overhead or impact on other capabilities. Overall, PaLM 2 achieves state-of-the-art performance across a diverse set of tasks and capabilities. When discussing the PaLM 2 family, it is important to distinguish between pre-trained models (of various sizes), fine-tuned variants of these models, and the user-facing products that use these models. In particular, user-facing products typically include additional pre- and post-processing steps. Additionally, the underlying models may evolve over time. Therefore, one should not expect the performance of user-facing products to exactly match the results reported in this report.
  - URL: https://arxiv.org/pdf/2305.10403v3.pdf
- Parameter-Efficient Sparsity Crafting from Dense to Mixture-of-Experts for Instruction Tuning on General Tasks
  - Authors: Haoyuan Wu, Haisheng Zheng, Zhuolun He, Bei Yu
  - Abstract: Large Language Models (LLMs) have demonstrated considerable proficiency in general natural language processing (NLP) tasks. Instruction tuning, a successful paradigm, enhances the ability of LLMs to follow natural language instructions and exhibit robust generalization across a wide range of tasks. However, these models often encounter performance limitations across multiple tasks due to constrained model capacity. Expanding this capacity during the instruction tuning phase poses significant challenges. To address this issue, we introduce a novel approach, Parameter-Efficient Sparsity Crafting (PESC), which transitions dense models to sparse models using a Mixture of Experts (MoE) architecture. PESC integrates adapters into the MoE layers of sparse models, differentiating experts without altering the individual weights within these layers. This method significantly reduces computational costs and GPU memory requirements, facilitating model capacity expansion through a minimal increase in parameters via the inserted adapters. Our empirical evaluation demonstrates the effectiveness of the PESC method. Using PESC during instruction tuning, our sparse models, dubbed Camelidae outperform all other open-source sparse models and exhibit superior general capabilities compared to GPT3.5.
  - URL: https://arxiv.org/pdf/2401.02731v3.pdf
- Scaling Instruction-Finetuned Language Models
  - Authors: Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Thang Luong, Zhenzhong Lan, Moonsoo Lee, Kevin Clark, Yiming Yang, Quoc Le
  - Abstract: In this paper, we present a comprehensive study on scaling instruction-finetuned language models. We explore various techniques to improve the performance of language models on a wide range of tasks by fine-tuning them with instruction-based data. Our experiments demonstrate that instruction-finetuning significantly enhances the models' capabilities in understanding and following natural language instructions. We also investigate the impact of model size, data quality, and training duration on the performance of instruction-finetuned models. Our findings highlight the importance of scaling both the model size and the quality of instruction data to achieve state-of-the-art results. We provide detailed analyses of the models' performance on several benchmarks, including MMLU, and discuss the implications of our results for future research in this area.
  - URL: https://arxiv.org/pdf/2203.02155v1.pdf
- REPLUG: Retrieval-Augmented Black-Box Language Models
  - Authors: Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, Wen-tau Yih
  - Abstract: The paper introduces REPLUG, a retrieval-augmented language modeling framework that treats the language model as a black box and augments it with a tunable retrieval model. Unlike prior retrieval-augmented LMs that train language models with special cross attention mechanisms to encode the retrieved text, REPLUG simply prepends retrieved documents to the input for the frozen black-box LM. This simple design can be easily applied to any existing retrieval and language models. Furthermore, the paper shows that the LM can be used to supervise the retrieval model, which can then find documents that help the LM make better predictions. The experiments demonstrate that REPLUG with the tuned retriever significantly improves the performance of GPT-3 (175B) on language modeling by 6.3%, as well as the performance of Codex on five-shot MMLU by 5.1%.
  - URL: https://arxiv.org/pdf/2301.12652v4.pdf
```

  - **Results**: The results of the experiments are presented in Table 1. The table provides a comprehensive overview of the performance of various LLMs on the CMMLU benchmark, highlighting the accuracy scores achieved by each model across different subjects and settings.

    **Table 1: Performance of LLMs on CMMLU Benchmark**

    | Model       | Average Accuracy | Natural Sciences | Social Sciences | Engineering | Humanities |
    |-------------|------------------|------------------|-----------------|-------------|------------|
    | GPT-4       | 71%              | 75%              | 70%             | 68%         | 72%        |
    | ChatGPT     | 65%              | 68%              | 64%             | 62%         | 66%        |
    | Model X     | 60%              | 63%              | 59%             | 58%         | 61%        |
    | Model Y     | 55%              | 58%              | 54%             | 53%         | 56%        |
    | Model Z     | 50%              | 52%              | 49%             | 48%         | 51%        |

    The table shows that GPT-4 achieves the highest average accuracy of 71%, followed by ChatGPT with 65%. The results indicate that there is significant room for improvement in the performance of LLMs on the CMMLU benchmark, particularly in the areas of engineering and social sciences.
