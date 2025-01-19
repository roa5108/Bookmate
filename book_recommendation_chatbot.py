import streamlit as st
import openai
import os
import huggingface_hub
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import requests
import re
import torch
import time
from bs4 import BeautifulSoup

# ================ 환경 변수 설정 ================
# API 키 설정
os.environ["OPENAI_API_KEY"] = "KEY_OF_OPENAI"
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
os.environ["NAVER_API_CLIENT_ID"] = "NAVER_API_CLIENT_ID"
os.environ["NAVER_API_CLIENT_SECRET"] = "NAVER_API_CLIENT_SECRET"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if not HUGGINGFACE_API_TOKEN:
    st.error("Hugging Face API 토큰이 설정되지 않았습니다. 환경 변수를 확인해 주세요.")
    st.stop()

try:
    huggingface_hub.login(HUGGINGFACE_API_TOKEN)
except requests.exceptions.HTTPError as e:
    st.error("Hugging Face 로그인에 실패했습니다. API 토큰을 확인해주세요.")
    st.stop()

# OpenAI Client 생성
client = openai.OpenAI()

# 전역 변수 선언
conversation_text = ""  # 전역 변수 초기화

# ================ 전역 설정 및 모델 로드 ================
# 모델 변경 및 예외 처리
model_name = "haramjang/gemma-2b-it-chatKeyword"
try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    model_name = "google/gemma-2b-it"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as fallback_e:
        st.stop()

def wrapper_generate(tokenizer, model, input_prompt, do_stream=False):
    def get_text_after_prompt(text):
        pattern = r"<start_of_turn>model\s*(.*?)(<end_of_turn>|$)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted_text = match.group(1).strip()
            return extracted_text
        else:
            return "매칭되는 텍스트가 없습니다."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data = tokenizer(input_prompt, return_tensors="pt")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    input_ids = data.input_ids[..., :-1]
    with torch.no_grad():
        pred = model.generate(
            input_ids=input_ids.cuda() if torch.cuda.is_available() else input_ids,
            streamer=streamer if do_stream else None,
            use_cache=True,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded_text = tokenizer.batch_decode(pred, skip_special_tokens=False)

    # gemma 결과에 대해 특별 처리
    return get_text_after_prompt(decoded_text[0])

def function_prepare_sample_text(tokenizer, for_train=True):
    """클로저"""

    def _prepare_sample_text(example):
        """Prepare the text from a sample of the dataset."""
        user_prompt= (
                  "다음 대화를 읽고 사용자의 관심사, 성향, 고민을 각각 키워드 형식으로 추출하시오. "
                  "1. 관심사는 사용자가 중요하게 여기는 주제나 활동을 나타냅니다. "
                  "2. 성향은 사용자의 성격적 특성, 행동 패턴, 또는 문제를 바라보는 태도를 나타냅니다. "
                  "3. 고민은 사용자가 대화 중에 표현한 문제나 걱정을 나타냅니다. "
                  "결과는 다음 형식으로 작성하십시오:\n"
                  "관심사: [키워드1, 키워드2, ...]\n"
                  "성향: [키워드1, 키워드2, ...]\n"
                  "고민: [키워드1, 키워드2, ...]\n"
                  "### 대화 내용: "
              )

        messages = [
            {"role": "user", "content": f"{user_prompt}{example['input']}"},
        ]
        if for_train:
            messages.append({"role": "assistant", "content": f"{example['output']}"})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False if for_train else True)
        return text
    return _prepare_sample_text

# 네이버 API를 이용한 도서 검색
def search_books_naver(client_id, client_secret, keywords):
    url = "https://openapi.naver.com/v1/search/book.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    all_books = []
    for keyword in keywords:
        params = {"query": keyword, "display": 10}
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            for item in data.get("items", []):
                title = item.get("title", "").replace("<b>", "").replace("</b>", "")
                author = item.get("author", "")
                description = item.get("description", "").replace("<b>", "").replace("</b>", "")
                all_books.append({"title": title, "author": author, "description": description})
        except Exception as e:
            st.error(f"네이버 API 오류: {e}")
    return all_books

def filter_books(books, keywords):
    scored_books = []
    for book in books:
        relevance_score = sum(kw.lower() in book['title'].lower() or kw.lower() in book['description'].lower() for kw in keywords)
        if relevance_score > 0:
            scored_books.append((book, relevance_score))
    scored_books.sort(key=lambda x: x[1], reverse=True)
    return [book for book, score in scored_books]

# 대화 기록 저장
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 요즘 어떻게 지내고 계세요?"}]

# 이전 대화 표시
st.title("📚 북메이트 - 책 추천 챗봇")
st.write("**당신의 취향에 맞는 책을 추천해드려요!** 저와의 대화를 통해서 맞춤형 책을 추천해 드릴게요!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if user_input := st.chat_input("메시지를 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 상담사 응답 처리
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages
        )
        assistant_message = response.choices[0].message.content
    except Exception as e:
        assistant_message = f"에러 발생: {e}"

    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    with st.chat_message("assistant"):
        st.markdown(assistant_message)

# 키워드 추출 및 책 추천
if st.button("대화를 종료하고 책 추천 받기"):
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    keywords_output = wrapper_generate(tokenizer, model, conversation_text)
    st.write("**키워드 분석 결과:**", keywords_output)

    keywords = re.findall(r"\[([^\]]+)\]", keywords_output)
    extracted_keywords = sum((kw.split(',') for kw in keywords), [])
    extracted_keywords = [k.strip() for k in extracted_keywords if k.strip()]

    if extracted_keywords:
        client_id = os.getenv("NAVER_API_CLIENT_ID")
        client_secret = os.getenv("NAVER_API_CLIENT_SECRET")
        books = search_books_naver(client_id, client_secret, extracted_keywords)
        filtered_books = filter_books(books, extracted_keywords)
        st.subheader("📖 추천 도서")
        for book in filtered_books[:4]:
            st.write(f"**제목:** {book['title']}")
            st.write(f"**저자:** {book['author']}")
            st.write(f"**설명:** {book['description']}")
            st.markdown("---")
    else:
        st.write("추천할 키워드가 없습니다.")