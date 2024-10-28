import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import AutoPeftModelForCausalLM
from langchain.prompts import PromptTemplate
from docx import Document
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab')

def parse_doc_to_sentences(filename = 'document.docx'):
    """
    Прочитать документ (по умолчанию 'document.docx') и разбить его на предложения
    """
    doc = Document(filename)
    
    text = " ".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
    
    return sent_tokenize(text)

############################################################################################################
# Определение моделей и функций, используемых для нахождения схожих предложений
sentence_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
sentence_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Использовать gpu по возможности
device="cuda" if torch.cuda.is_available() else "cpu"

def get_embedding(sentence):
    """
    Получение эмбеддинга предложения
    """
    #Mean Pooling - учет attention mask для корректного усреднения
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] # Первый элемент model_output содержит эмбеддинги токенов
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Токенизация предложения
    encoded_input = sentence_tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')

    # Вычисление эмбеддингов токенов
    with torch.no_grad():
        model_output = sentence_model(**encoded_input)

    # Выполнение пулинга
    sentence_embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])

    # Нормализация эмбеддингов
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings

def cosine_similarity(query_embedding, embeddings):
    """
    Косинусное сходство
    """
    # Нормализация эмбеддингов
    query_norm = query_embedding / query_embedding.norm(dim=1, keepdim=True)
    embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    # Расчет косинусного сходства
    similarity = torch.mm(query_norm, embeddings_norm.T)
    return similarity

def retrieve_similar_sentences(query, embeddings, sentences, k=5):
    """
    Отбор более подходящих предложений
    """
    # Получение эмбеддингов для запроса
    query_embedding = get_embedding(query)
    # Вычисление косинусного сходства
    similarity = cosine_similarity(query_embedding, embeddings)
    # Получение индексов наиболее похожих предложений
    top_k_indices = similarity[0].topk(k).indices
    # Возвращение соответствующих предложений
    retrieved_sentences = [sentences[i] for i in top_k_indices]
    return retrieved_sentences
############################################################################################################


############################################################################################################
# Определение моделей и функций, используемых для генерации текста на основе контекста
adapt_model_name = "IlyaGusev/saiga_mistral_7b_lora"
base_model_name = "Open-Orca/Mistral-7B-OpenOrca"

tokenizer = AutoTokenizer.from_pretrained(
              base_model_name,
              trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
device_map = {"": 0}

model = AutoPeftModelForCausalLM.from_pretrained(
              adapt_model_name,
              device_map=device_map,
              torch_dtype=torch.bfloat16)

info_prompt = PromptTemplate.from_template("""
    <s>user
    Текст: {context}
    Вопрос: {question}</s>
    <s>bot
    Ответ:</s>""")

def get_answer(context, question):
    """
    Получение ответа на основе контекста
    """
    prompt = info_prompt.format(context=context, question=question)   
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(input_ids=inputs["input_ids"].to(device), 
                            top_p=0.5,
                            temperature=0.3,
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=50,
                            pad_token_id=tokenizer.eos_token_id,
                            do_sample=True)

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    parsed_answer = output.split("Ответ:")[1].strip()

    if "bot\n" in parsed_answer:
        parsed_answer = parsed_answer.replace("bot\n", "").strip()

    return parsed_answer
############################################################################################################


# Основная программа
if __name__ == "__main__":
    # Парсинг документа в массив предложений и создание эмбеддингов
    filename = 'document.docx'
    sentences = parse_doc_to_sentences(filename)
    embedding = torch.cat([get_embedding(sentence) for sentence in sentences])
    
    while True:
        query = input("Задайте вопрос: ") #Например, "В каких задачах безопасности может применяется конструктор?"
        retrieved_sentences = retrieve_similar_sentences(query, embedding, sentences)
        answer = get_answer(retrieved_sentences, query)
        print("Ответ: " + answer)