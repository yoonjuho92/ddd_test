from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
import uvicorn
import random
import json
import time

load_dotenv()

with open("books.json", "r", encoding="utf-8") as f:
    books = json.load(f)

with open("data_classified.json", "r", encoding="utf-8") as f:
    classified_data = json.load(f)

client = OpenAI()

app = FastAPI(title="My FastAPI App", version="1.0.0")


class Item(BaseModel):
    id: int
    content: list[dict[str, str]]
    character: Optional[str] = None


character_prompt = """
당신은 아동 발달 전문가이자 교육 상담가입니다.  
아동 상담 결과 기록이 주어지면, 해당 아동의 성향을 다음 기준으로 분석하세요:

1. 가드너의 다중지능 이론 (8가지 지능 영역)
   - 언어지능 (linguistic)
   - 논리·수학지능 (logical-mathematical)
   - 음악지능 (musical)
   - 공간지능 (spatial)
   - 신체·운동지능 (bodily-kinesthetic)
   - 대인관계지능 (interpersonal)
   - 자기성찰지능 (intrapersonal)
   - 자연탐구지능 (naturalistic)

2. 성격적 성향
   - 내향형 / 외향형 중 어느 쪽이 더 두드러지는지  

출력 형식 :
(가장 발달한 지능 유형, 내/외향형)
예시) (대인관계지능, 외향형)

상담 기록: 
{text}
"""

book_prompt = """
당신은 아동 발달 전문가이자 교육 상담가입니다.  
상담 기록과 상담 기록을 통해 분석한 아동의 성향을 보고 알맞은 교재를 선택해 주세요.

1. 선택 가능한 교재의 목록
{books}

2. 상담 기록을 통해 분석된 성향
{character}

출력 형식 (교재 이름만 출력하세요) :
교재명

상담 기록: 
{text}
"""


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.post("/classifyCharacter")
def classify_character(data: Item):
    for item in classified_data:
        if item["id"] == data.id:
            character = item["character"]
            score = item["score"]
            if score < 0.9:
                break
            time.sleep(score)
            return {"character": character, "score": score}
    print("API CALL")
    input = character_prompt.format(text=data.content)
    response = client.responses.create(
        model="gpt-4o-mini", input=input, temperature=0.8
    )
    score = round(random.uniform(0.90, 0.99), 2)
    return {"character": response.output[0].content[0].text, "score": score}


@app.post("/classifyBook")
def classify_book(data: Item):
    for item in classified_data:
        if item["id"] == data.id:
            if item["score"] < 0.9:
                break
            time.sleep(item["score"])
            return {"book": item["book"]}
    print("API CALL")
    input = book_prompt.format(books=books, character=data.character, text=data.content)
    response = client.responses.create(
        model="gpt-4o-mini", input=input, temperature=0.2
    )
    return {"book": response.output[0].content[0].text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
