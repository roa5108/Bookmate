import requests
import os  # 환경 변수 사용
from bs4 import BeautifulSoup

# 네이버 API를 이용한 도서 검색
def search_books_naver(client_id, client_secret, keywords):
    url = "https://openapi.naver.com/v1/search/book.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }

    all_books = []

    for keyword in keywords:
        params = {"query": keyword, "display": 10}  # 단순 키워드 검색

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            if "items" not in data or not data["items"]:
                print(f"네이버 API에서 결과 없음: {data}")
                continue

            for item in data["items"]:
                title = item.get("title", "").replace("<b>", "").replace("</b>", "")
                author = item.get("author", "").replace("<b>", "").replace("</b>", "")
                description = item.get("description", "").replace("<b>", "").replace("</b>", "")
                all_books.append({"title": title, "author": author, "description": description})

        except requests.exceptions.RequestException as e:
            print(f"API 요청 중 오류 발생: {e}")
            continue

    return all_books

# 도서 결과 필터링 및 점수 계산
def filter_books(books, keywords):
    scored_books = []
    for book in books:
        relevance_score = sum(kw.lower() in book['title'].lower() or kw.lower() in book['description'].lower() for kw in keywords)
        if relevance_score > 0:
            scored_books.append((book, relevance_score))

    # 점수 기준으로 정렬
    scored_books.sort(key=lambda x: x[1], reverse=True)
    return [book for book, score in scored_books]

# 실행 예제
if __name__ == "__main__":
    # NAVER API 키를 환경 변수에서 가져오기
    client_id = os.getenv("NAVER_API_CLIENT_ID")
    client_secret = os.getenv("NAVER_API_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("API 키가 설정되지 않았습니다. 환경 변수를 확인해 주세요.")
        exit()

    # 사용자 입력
    interests = input("관심사를 쉼표로 구분하여 입력하세요: ")
    traits = input("성향을 쉼표로 구분하여 입력하세요: ")
    concerns = input("고민을 쉼표로 구분하여 입력하세요: ")

    # 1. 키워드 추출 및 우선순위 적용
    keywords = []
    keywords.extend(interests.split(','))  # 관심사 우선
    keywords.extend(concerns.split(','))  # 고민 그다음
    keywords.extend(traits.split(','))    # 성향 마지막

    keywords = [kw.strip() for kw in keywords if kw.strip()]

    print(f"\n추출된 키워드: {keywords}")

    if keywords:
        # 2. 네이버 API를 이용한 도서 검색
        all_books = search_books_naver(client_id, client_secret, keywords)

        # 3. 중복 제거 및 점수 계산
        unique_books = {book["title"]: book for book in all_books}.values()

        # 4. 필터링하여 최대 4개만 출력
        filtered_books = filter_books(unique_books, keywords)
        limited_books = filtered_books[:4]

        # 5. 결과 출력
        if limited_books:
            print("\n추천 도서 결과:")
            for i, book in enumerate(limited_books, 1):
                print(f"{i}. 제목: {book['title']}\n   저자: {book['author']}\n   상세: {book['description']}\n")
        else:
            print("\n추천할 도서가 없습니다.")
    else:
        print("키워드 추출에 실패했습니다.")
