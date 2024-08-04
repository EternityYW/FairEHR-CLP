import pandas as pd
from transformers import AutoTokenizer, pipeline
import torch

df = pd.read_csv('deidentified_notes.csv')

model_name = "meta-llama/Llama-2-70b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llama_pipeline = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

# function to generate paraphrased text
def paraphrase(note):
    prompt = f"Please paraphrase the provided clinical notes, ensuring no critical medical components such as medical history, diagnoses, and treatments are omitted while maintaining the integrity of authentic documentation: {note}"
    paraphrased_text = llama_pipeline(prompt, max_length=1024, num_return_sequences=1)[0]['generated_text']
    return paraphrased_text

df['paraphrased_notes'] = df['Masked'].apply(paraphrase)

df.to_csv('paraphrased_notes.csv', index=False)
