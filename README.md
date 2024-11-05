1. test_image.py, 1_13023.csv, Embeddings.csv, Embeddings.npy, testset, stopword_ko.txt 같은 디렉터리에 놓기

2. pip install -r requirements.txt

3. pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

4. pip install tf-keras

4. test_image.py 실행

5. user_image_path: 입력 이미지 경로 | user_text: 입력 텍스트 경로 | top_N: 찾아낼 이미지의 수
   region_ids: 지역 ID(0:기타, 1:서울, 2:강원특별자치도, 3:인천, 4:충청북도, 5:대전, 6:충청남도, 7:대구, 8:경상북도, 9:광주
                      10:경상남도, 11:부산, 12:전북특별자치도, 13:울산, 14:전라남도, 15:세종특별자치시, 16:제주도, 17:경기도)
   category_ids: 카테고리 ID(0:기타, 1:자연, 2:역사, 3:휴양, 4:체험, 5:산업, 6:건축)
