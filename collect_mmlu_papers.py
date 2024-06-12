import json

# JSON data from the browser console logs
json_data = '''
[
    {"title":"MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark","authors":"Y Wang, X Ma, G Zhang, Y Ni, A Chandra… - arXiv preprint arXiv…, 2024 - arxiv.org","link":"https://arxiv.org/abs/2406.01574"},
    {"title":"Evaluating large language models: Chatgpt-4, mistral 8x7b, and google gemini benchmarked against mmlu","authors":"K Ono, A Morita - Authorea Preprints, 2024 - techrxiv.org","link":"https://www.techrxiv.org/doi/full/10.36227/techrxiv.170956672.21573677"},
    {"title":"Reducing llm hallucination using knowledge distillation: A case study with mistral large and mmlu benchmark","authors":"D McDonald, R Papadopoulos, L Benningfield - Authorea Preprints, 2024 - techrxiv.org","link":"https://www.techrxiv.org/doi/full/10.36227/techrxiv.171665607.76504195"},
    {"title":"Fine-tuning Llama For Better Performance With the MMLU Benchmark","authors":"ML Yim, CH Yip - osf.io","link":"https://osf.io/e3v5x/download"},
    {"title":"Higher Performance of Mistral Large on MMLU Benchmark through Two-Stage Knowledge Distillation","authors":"J Wilkins, M Rodriguez - 2024 - researchsquare.com","link":"https://www.researchsquare.com/article/rs-4410506/latest"},
    {"title":"KMMLU: Measuring Massive Multitask Language Understanding in Korean","authors":"G Son, H Lee, S Kim, S Kim, N Muennighoff… - arXiv preprint arXiv…, 2024 - arxiv.org","link":"https://arxiv.org/abs/2402.11548"},
    {"title":"Scaling instruction-finetuned language models","authors":"HW Chung, L Hou, S Longpre, B Zoph, Y Tay… - Journal of Machine…, 2024 - jmlr.org","link":"https://www.jmlr.org/papers/v25/23-0870.html"},
    {"title":"Rethinking LLM Language Adaptation: A Case Study on Chinese Mixtral","authors":"Y Cui, X Yao - arXiv preprint arXiv:2403.01851, 2024 - arxiv.org","link":"https://arxiv.org/abs/2403.01851"},
    {"title":"Instruction Tuning With Loss Over Instructions","authors":"Z Shi, AX Yang, B Wu, L Aitchison, E Yilmaz… - arXiv preprint arXiv…, 2024 - arxiv.org","link":"https://arxiv.org/abs/2405.14394"},
    {"title":"Self-Regulated Sample Diversity in Large Language Models","authors":"M Liu, J Frawley, S Wyer, HPH Shum, SL Uckelman… - 2024 - hubertshum.com","link":"http://hubertshum.com/publications/naacl2024llm/files/naacl2024llm.pdf"}
]
'''

# Load the JSON data
papers = json.loads(json_data)

# Write the details to a file
output_file = '/home/ubuntu/mmlu_paper_details.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    for paper in papers:
        file.write(f"Title: {paper['title']}\n")
        file.write(f"Authors: {paper['authors']}\n")
        file.write(f"Link: {paper['link']}\n")
        file.write("\n")

print(f"Details of MMLU-related papers have been written to {output_file}")
