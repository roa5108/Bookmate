# -*- coding: utf-8 -*-
"""keyword_extraction_chatbot.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1m6GIhBVk7oJzS4dTr7-aXoeHMssQ9NCI
"""

import huggingface_hub
from huggingface_hub import login

# 환경 변수에서 토큰 가져오기
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
huggingface_hub.login(HUGGINGFACE_API_TOKEN)

import re  # 정규식을 위해 필요
import torch
from transformers import TextStreamer
import os
import openai

os.environ["OPENAI_API_KEY"] = "KEY_OF_OPENAI"

# OpenAI Client 생성
client = openai.OpenAI()

# 전역 변수 선언
conversation_text = ""  # 전역 변수 초기화

# 챗봇 대화 함수
def chatbot_conversation():
    global conversation_text  # 함수 내에서 전역 변수 사용
    messages = [
        {"role": "system", "content": "당신은 친절하고 공감 능력이 뛰어난 상담사입니다. 존댓말을 유지해주세요. 유저의 일상 대화를 통해 심리적 안정감을 주고, 생각과 감정을 이끌어내는 질문을 던지세요. 최대한 사람처럼 딱딱한 말투는 피해주세요. 너무 공격적인 질문은 자제해주세요. 채팅으로 대화가 가능하도록 다음의 입력된 내용을 바탕으로 반드시 한 문장으로 대답해주세요. "}
    ]
    conversation_log = []  # 대화 내용을 저장할 리스트

    print("안녕하세요! 요즘 어떻게 지내고 계세요?")
    conversation_log.append("상담사: 안녕하세요! 요즘 어떻게 지내고 계세요?")

    while True:
        # 유저 입력 받기
        user_input = input("")
        if not user_input.strip() or user_input in ["대화를 종료하고 싶어", "대화를 종료할게"]:  # 빈 입력 시 종료
            closing_remark = "대화를 종료합니다. 당신의 관심사에 도움이 되는 책을 찾고 있어요."
            user_closing = "대화를 종료하고 싶어"
            print(closing_remark)
            conversation_log.append(f"유저: {user_closing}")
            conversation_log.append(f"상담사: {closing_remark}")
            break

        # 유저 메시지 추가
        messages.append({"role": "user", "content": user_input})
        conversation_log.append(f"유저: {user_input}")

        # GPT 모델 호출
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            assistant_message = response.choices[0].message.content
            print(f"{assistant_message}")

            # 챗봇 메시지 추가
            messages.append({"role": "assistant", "content": assistant_message})
            conversation_log.append(f"상담사: {assistant_message}")

        except Exception as e:
            print(f"에러 발생: {e}")
            break

    # 대화 내용을 포맷하여 출력
    conversation_text = " ".join(conversation_log)
    print("\n=== 대화 로그 ===")
    print(conversation_text)

# 실행
if __name__ == "__main__":
    chatbot_conversation()

from transformers import AutoModelForCausalLM, AutoTokenizer

# Hugging Face Hub에 업로드된 모델 이름
model_name = "google/gemma-2-2b-it"

# 모델과 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
            input_ids=input_ids.cuda(),
            streamer=streamer if do_stream else None,
            use_cache=True,
            max_new_tokens=float("inf"),
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
            # {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": f"{user_prompt}{example['input']}"},
        ]
        if for_train:
            messages.append({"role": "assistant", "content": f"{example['output']}"})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False if for_train else True)
        return text
    return _prepare_sample_text

input_text = conversation_text

preprocessor = function_prepare_sample_text(tokenizer, for_train=False)

preprocessor({'input' : conversation_text})

# 테스트 입력 문장
model_output = wrapper_generate(tokenizer, model, input_prompt=preprocessor({'input': conversation_text}), do_stream=False)
print("당신의 키워드:", model_output)

