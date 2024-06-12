# Detailed Report on CMMLU Models

## Abstract
As the capabilities of large language models (LLMs) continue to advance, evaluating their performance is becoming simultaneously more important and more challenging. This paper aims to address this issue for Mandarin Chinese in the form of CMMLU, a comprehensive Chinese benchmark that covers various subjects, including natural sciences, social sciences, engineering, and the humanities. We conduct a thorough evaluation of more than 20 contemporary multilingual and Chinese LLMs, assessing their performance across different subjects and settings. The results reveal that most existing LLMs struggle to achieve an accuracy of 60%, which is the pass mark for Chinese exams. This highlights that there is significant room for improvement in the capabilities of LLMs. Additionally, we conduct extensive experiments to identify factors impacting the models’ performance and propose directions for enhancing LLMs. CMMLU fills the gap in evaluating the knowledge and reasoning capabilities of large language models in the Chinese context.

## Introduction
Large language models (LLMs) have driven remarkable advancements in natural language processing and artificial intelligence, revolutionizing the field. However, assessing the knowledge and reasoning abilities of these models has become increasingly challenging, especially with the proliferation of LLMs that generate fluent and plausible responses. To this end, researchers have created various benchmarks intended to evaluate different model capabilities. Specifically, the MMLU benchmark encompasses various tasks ranging from elementary mathematics and computer science to management and law, which can be used to comprehensively measure LLM capabilities in terms of the knowledge embedded.

## CMMLU Benchmark
In this paper, we propose CMMLU, a comprehensive Chinese assessment suite specifically designed to evaluate the advanced knowledge and reasoning abilities of LLMs in a Chinese linguistic and cultural context. CMMLU covers a wide range of subjects, comprising 67 topics from elementary to advanced professional levels. It includes subjects that require computational expertise, such as physics and mathematics, as well as disciplines within the humanities and social sciences. Many of these tasks are not easily translatable from other languages due to their specific contextual nuances.

## Evaluation and Results
We assess GPT4, ChatGPT, and more than 20 advanced open-source multilingual and Chinese LLMs on CMMLU. The results reveal that the majority of these models struggle to achieve an accuracy score of 60%, relative to random accuracy of 25%. Notably, GPT4 achieves an average accuracy of 71%. These findings highlight the considerable room for improvement in LLMs in terms of Chinese knowledge and language understanding.

### Key Findings
- **Chain-of-Thought Prompts**: Most existing models do not benefit from chain-of-thought prompts in CMMLU.
- **Few-Shot Examples**: Few-shot examples help foundation models in the comprehension of tasks and enhance their reasoning abilities but do not help models that have undergone supervised fine-tuning (SFT) or reinforcement learning from human feedback (RLHF).
- **Negation Words**: LLMs perform worse on questions with negation words compared to those without negation words, but recently-released models mitigate this disparity either through better pre-training data or fine-tuning.
- **Sub-Options**: Questions with sub-options are difficult for all existing LLMs, with even GPT4 dropping 20% in accuracy over such questions.

## Related Work
Benchmarking plays a crucial role in measuring AI development, particularly in the domain of LLMs. While benchmarks such as GLUE and SuperGLUE have played an important role in tracking progress in natural language understanding (NLU) tasks, they primarily focus on specific language skills. With an increasing move to generative models which are highly adept at generating fluent outputs, the value of these benchmarks has diminished, and new datasets have been proposed to evaluate LLM abilities over more general tasks.

## Methodology
### Task Overview
We created an extensive multitask test for Mandarin Chinese, which covers diverse areas of knowledge, including the humanities, social sciences, STEM (science, technology, engineering, and mathematics), and other areas that are important in daily life. It includes common test questions in subjects like mathematics, physics, and chemistry with answers that are not language or region specific, but also several tasks that are very region-specific, such as Chinese driving rules, Chinese food culture, and Chinese teacher qualifications.

### Data Collection
We hired four annotators with undergraduate or higher education levels to manually collect the questions and answers from freely available resources. To prevent our questions from appearing in the training set of LLMs, we invested specific effort in identifying non-publicly available materials, mock exam questions, and questions from quiz shows. More than 80% of our data was crawled from PDFs (after OCR), which further reduces the possibility of it occurring in LLM training data.

### Format
Each question in the dataset is a multiple-choice question with 4 choices, only one of which is correct. The questions are expressed as fill-in-the-blank (by choosing the correct option), or direct-answer questions. For chemical formulae and mathematical expressions, we use a 50:50 mixture of LATEX and plain text, where plain text was only allowed if an expression is commonly used and not prone to ambiguity.

### Quality Check
To further check data quality, we randomly sampled 5% questions with answers for verification.

## Experiments
To provide an overview of existing LLMs on language understanding within the context of Chinese, we evaluate two commercial LLMs and more than 20 open-source LLMs in different sizes, language orientations, and stages (i.e., either foundation model or SFT/RLHF model). We analyze their performance and investigate several factors that could affect the performance of LLMs.

### Setup
Our goal is to assess the LLMs performance on CMMLU, which contains multiple-choice questions with one correct answer for each question. There have been several strategies to perform multiple-choice question-answering tasks. For commercial models which we cannot get the weights (i.e., GPT4 and ChatGPT), we input the question with all candidate choices, allowing the model to generate the output, and use a series of regular expressions (regex) to match the model’s prediction. For open-source models, we input the question and choices, and prompt the model by asking the answer key. Then we obtain the logits of the next predicted token, and compare the probability among the 4 tokens: ‘A’, ‘B’, ‘C’, and ‘D’ and select the token with the highest probability as the model’s choice.

### Results
The results of the experiments are presented in Table 1, showing the five-shot accuracy of various models across different categories, including STEM, Humanities, Social Science, Other, and China-specific subjects.

## Analysis
In order to gain a comprehensive understanding of the LLM’s performance on CMMLU, we explored three factors that may enhance the model’s performance and two factors that could potentially diminish its performance. Specifically, we investigated whether the following factors can improve the model’s performance: (1) utilizing chain-of-thought prompts, (2) increasing the number of input examples, and (3) employing larger-sized models within the same family. Conversely, we explored whether the following factors make the task more challenging for LLMs: (4) questions containing negation words, and (5) questions with sub-options within them.

## Conclusion
We introduce CMMLU, a groundbreaking benchmark designed to assess the multi-task language understanding capabilities in Chinese. Our experimental findings reveal substantial opportunities for improvement within existing large language models. Through extensive analysis, we identify several factors that impact model performance and propose actionable directions for enhancing LLMs. We are confident that our benchmark dataset and analytical insights will empower researchers to effectively evaluate and design Chinese LLMs.
