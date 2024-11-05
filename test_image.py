import time

start_time = time.time()  # 코드 시작 시간 기록

import numpy as np  # 행렬 연산 및 수학적 계산을 위한 라이브러리
import torch  # PyTorch 딥러닝 라이브러리
import cv2  # OpenCV 라이브러리
from retinaface import RetinaFace  # RetinaFace 라이브러리
from PIL import Image  # 이미지 처리 및 변환을 위한 라이브러리
from torchvision import models, transforms  # PyTorch의 컴퓨터 비전 모델 및 변환 기능
from konlpy.tag import Okt  # 한국어 형태소 분석을 위한 KoNLPy의 Okt 클래스
import io  # 바이트 스트림을 다루기 위한 라이브러리
from azure.cosmos import CosmosClient  # Azure Cosmos DB와 상호작용하기 위한 클라이언트
from azure.storage.blob import BlobServiceClient  # Azure Blob Storage와 상호작용하기 위한 클라이언트
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리
from pinecone import Pinecone, ServerlessSpec

# ResNet50 모델과 디바이스 설정 (GPU 사용 가능 여부에 따라 설정)
device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 가능하면 'cuda', 아니면 'cpu' 사용
resnet50_model = models.resnet50(weights="DEFAULT")  # ResNet50 모델을 미리 학습된 가중치로 불러오기
resnet50_model = resnet50_model.to(device)  # 모델을 설정된 디바이스로 이동
resnet50_model.eval()  # 모델을 평가 모드로 설정 (드롭아웃 등 비활성화)

# 이미지 전처리 설정 (ResNet50 입력에 맞게 크기 조정 및 정규화)
preprocess = transforms.Compose([
    transforms.Resize(256),  # 이미지 크기를 256x256으로 조정
    transforms.CenterCrop(224),  # 이미지를 중앙에서 224x224 크기로 자름
    transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 이미지 정규화
])

# Pinecone 설정
pinecone_key = "b3735e0a-9145-4822-84e0-ace768f646f2"
pinecone_environment = "us-east-1"
pinecone_index_name = "travelnuri"

pc = Pinecone(api_key=pinecone_key, environment=pinecone_environment)   # Pinecone 인스턴스 생성
index = pc.Index(pinecone_index_name)   # Pinecone 인덱스 초기화

# Cosmos DB 설정
cosmos_endpoint = "https://suheonchoi1.documents.azure.com:443/"  # Cosmos DB 엔드포인트 URL
cosmos_key = "aMj7nXDrcGizt92gR0xeTeFkwI8yIkb7G5Fac2FaCl8dLgiU4HhVymZnLJUkFaxJXcJvnIbrE7BfACDb1BBTOQ=="  # Cosmos DB 액세스 키
cosmos_database_name = "ImageDatabase"  # 사용할 Cosmos DB 데이터베이스 이름
image_container_name = "ImageTable"  # 사용할 Cosmos DB 컨테이너 이름

# Blob Storage 설정
connection_string = "DefaultEndpointsProtocol=https;AccountName=suheonchoi1;AccountKey=uZ10rRIn+q3cbHd04iimIZY38pcda1ePeyyDbe1YEG9WkCQsu+6WXj2EsOw3mwMzWP4MVfRwkejj+AStg0UJYg==;EndpointSuffix=core.windows.net"  # Azure Blob Storage 연결 문자열
blob_service_client = BlobServiceClient.from_connection_string(connection_string)  # Blob 서비스 클라이언트 생성
blob_container_client = blob_service_client.get_container_client("image-container-travelnuri")  # Blob 컨테이너 클라이언트 생성
blob_container_client2 = blob_service_client.get_container_client("embedding-container-travelnuri")  # Blob 컨테이너 클라이언트 생성

# Okt 객체 생성
okt = Okt()  # 한국어 명사 추출을 위한 Okt 객체 생성

# 이미지 파일에서 ResNet50 모델을 사용하여 임베딩을 구하는 함수
def get_image_features(image_path):
    print(f"Extracting features from image: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')  # 이미지를 열어 RGB 형식으로 변환
    except Exception as e:
        print(f"Error opening image: {e}")  # 이미지 열기 실패 시 오류 메시지 출력
        return None  # 오류 발생 시 None 반환

    image_input = preprocess(image).unsqueeze(0).to(device)  # 이미지 전처리 후 배치 차원 추가 및 디바이스로 이동

    with torch.no_grad():  # 그래디언트 계산 비활성화 (추론 모드)
        image_features = resnet50_model(image_input)  # 모델을 사용하여 이미지 임베딩 계산

    print(f"Features extracted successfully for image: {image_path}")
    return image_features.cpu().numpy()  # 계산된 임베딩을 CPU로 이동 후 numpy 배열로 변환

# 입력된 텍스트에서 명사만 추출하는 함수
def extract_nouns_and_adjectives(text):
    print(f"Loading stopword...")
    with open('stopword_ko.txt', encoding='utf-8') as f:  # 불용어 파일 읽기
        stopwords = set(line.strip() for line in f)  # 불용어를 집합(set)으로 저장
    print(f"Loaded {len(stopwords)} stopwords.")

    print(f"Extracting nouns and adjectives from text: '{text}'")
    pos_tagged = okt.pos(text)  # 텍스트에서 품사 태깅
    nouns_and_adjectives = [word for word, pos in pos_tagged if pos in ['Noun', 'Adjective']]  # 명사와 형용사 추출

    filtered_nouns_and_adjectives = [word for word in nouns_and_adjectives if word not in stopwords]  # 불용어 제외

    print(f"Extracted nouns and adjectives (after stopword removal): {filtered_nouns_and_adjectives}")

    return set(filtered_nouns_and_adjectives)  # 중복 제거를 위해 set으로 변환하여 반환

# 얼굴을 원 또는 타원 모양으로 블러 처리하는 함수
def blur_faces(image):
    img_array = np.array(image)

    # RetinaFace를 사용하여 얼굴 감지
    faces = RetinaFace.detect_faces(img_array)

    if isinstance(faces, dict):  # 감지된 얼굴이 있을 경우
        for face_key in faces.keys():
            face = faces[face_key]
            facial_area = face["facial_area"]
            x1, y1, x2, y2 = facial_area

            # 얼굴 중심점과 크기 계산
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            width, height = x2 - x1, y2 - y1

            # 원형 또는 타원형 마스크 생성 (이 예제에서는 타원형)
            mask = np.zeros_like(img_array, dtype=np.uint8)
            mask = cv2.ellipse(mask, (center_x, center_y), (width // 2, height // 2), 0, 0, 360, (255, 255, 255), -1)

            # 블러 처리된 얼굴 영역
            blurred_face = cv2.GaussianBlur(img_array[y1:y2, x1:x2], (99, 99), 30)

            # 원형 또는 타원형 마스크를 적용하여 블러 처리된 얼굴과 원본 이미지 합성
            img_array = cv2.bitwise_and(img_array, 255 - mask)
            mask_face = cv2.bitwise_and(blurred_face, mask[y1:y2, x1:x2])
            img_array[y1:y2, x1:x2] += mask_face

    return Image.fromarray(img_array)


# Blob Storage에서 이미지를 다운로드하여 유사도 점수에 따라 정렬된 순서로 시각화하는 함수
def display_images(top_image_ids, place_ids, region_ids, category_ids, top_N_scores):
    print("Displaying images...")
    plt.figure(figsize=(15, 10))

    cosmos_client = CosmosClient(cosmos_endpoint, cosmos_key)
    database = cosmos_client.get_database_client(cosmos_database_name)
    image_container = database.get_container_client(image_container_name)

    for i, image_id in enumerate(top_image_ids):
        image_id = int(image_id)
        query = f"SELECT c.ImageName FROM c WHERE c.ImageID = {image_id}"
        items = list(image_container.query_items(query, enable_cross_partition_query=True))

        if items:
            image_name = items[0]['ImageName']
            print(f"Downloading and displaying image: {image_name}")

            blob_client = blob_container_client.get_blob_client(image_name)
            blob_data = blob_client.download_blob().readall()
            image = Image.open(io.BytesIO(blob_data))

            # 얼굴 블러 처리 적용
            image = blur_faces(image)

            # 유사도 점수 및 추가 정보 표시
            plt.subplot(1, len(top_image_ids), i + 1)
            plt.imshow(image)
            plt.title(f"ImageID: {image_id}\nPlaceID: {place_ids[i]}\nRegionID: {region_ids[i]}\nCategoryID: {category_ids[i]}\nScore: {top_N_scores[i]:.4f}")
            plt.axis('off')
        else:
            print(f"No image found for ImageID: {image_id}")

    plt.show()


def find_similar_image(user_image_path, user_text=None, region_ids=None, category_ids=None, top_N=5):
    print(f"Starting similarity search for image: {user_image_path}")
    user_image_features = get_image_features(user_image_path)
    if user_image_features is None:
        print("Failed to extract image features. Exiting.")
        return

    # Pinecone에서 필터 조건에 맞는 전체 결과 검색
    print("Querying Pinecone for similar images...")
    query_results = index.query(
        vector=user_image_features.flatten().tolist(),
        top_k=10000,  # 최대 10,000개의 결과를 가져옴
        include_metadata=True
    )

    filtered_count = len(query_results['matches'])
    print(f"Initial results count: {filtered_count}")

    if filtered_count == 0:
        print("No relevant images found based on the provided filters.")
        return

    # RegionID 필터링
    if region_ids:
        region_ids_set = set(map(str, region_ids.split(',')))
        query_results['matches'] = [match for match in query_results['matches']
                                    if match['metadata'].get('RegionID') in region_ids_set]
        print(f"Filtered results count after RegionID filter: {len(query_results['matches'])}")

    # CategoryID 필터링
    if category_ids:
        category_ids_set = set(map(str, category_ids.split(',')))
        query_results['matches'] = [match for match in query_results['matches']
                                    if match['metadata'].get('CategoryID') in category_ids_set]
        print(f"Filtered results count after CategoryID filter: {len(query_results['matches'])}")

    # 텍스트 필터링을 위해 overview를 가져옴
    filtered_matches = query_results['matches']  # 기본적으로 필터링된 이미지 목록

    user_keywords = extract_nouns_and_adjectives(user_text)
    if len(user_keywords) == 0:
        print("명사 또는 형용사를 인식할 수 없습니다.")
        print("user_text 무시")

    if user_text and len(user_keywords) != 0:  # user_text가 있을 때만 필터링 수행
        # overview 기반 필터링
        filtered_matches = []
        for match in query_results['matches']:
            overview = match['metadata'].get('overview', '')  # 'overview' 한글 처리

            # 텍스트 기반 필터링 (overview에 user_text 키워드가 포함되어 있는지 확인)
            if any(keyword in overview for keyword in user_keywords):
                filtered_matches.append(match)

    final_filtered_count = len(filtered_matches)
    print(f"Filtered results count after applying text filter: {final_filtered_count}")

    if final_filtered_count == 0:
        print("No relevant images found after text-based filtering.")
        return

    # 최종 상위 N개의 결과로 제한
    top_image_ids = []
    top_N_scores = []
    place_ids = []
    region_ids = []
    category_ids = []

    for match in filtered_matches[:top_N]:
        top_image_ids.append(match['id'])
        top_N_scores.append(match['score'])
        place_ids.append(match['metadata'].get('PlaceID'))
        region_ids.append(match['metadata'].get('RegionID'))
        category_ids.append(match['metadata'].get('CategoryID'))

    print(f"Top {top_N} most similar images found after all filters:")
    for image_id, score in zip(top_image_ids, top_N_scores):
        print(f"ImageID: {image_id}, Similarity Score: {score}")

    # 실행 시간 계산 및 출력
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    # 이미지 시각화
    display_images(top_image_ids, place_ids, region_ids, category_ids, top_N_scores)
    print("Image similarity search completed.")


# 사용자로부터 입력받을 로컬 이미지 경로와 텍스트 설정
user_image_path = "testset/test1.jfif"  # 사용자 이미지 파일 경로를 지정
user_text = "공원"  # 사용자 입력 텍스트 (텍스트가 없으면 None으로 설정)
region_ids = ""  # 사용자 입력 RegionID (필터링 기준)
category_ids = ""  # 사용자 입력 CategoryID (필터링 기준)

# 가장 유사한 이미지 5장 찾기
find_similar_image(user_image_path, user_text, region_ids, category_ids, top_N=5)  # 함수 호출하여 유사한 이미지 찾기
