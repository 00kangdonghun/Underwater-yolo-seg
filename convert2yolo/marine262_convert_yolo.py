import os
import json
from tqdm import tqdm

def convert_coco_to_yolov8_segmentation(coco_dir, yolo_dir):
    """
    COCO 형식의 JSON 파일이 담긴 폴더를 YOLOv8 segmentation 형식으로 변환합니다.
    각 JSON 파일은 단일 이미지에 대한 정보를 포함해야 합니다.

    Args:
        coco_dir (str): COCO JSON 파일들이 있는 입력 폴더 경로.
        yolo_dir (str): 변환된 YOLOv8 라벨 파일을 저장할 출력 폴더 경로.
    """
    # 출력 폴더가 없으면 생성
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)

    # COCO 라벨 폴더에 있는 모든 JSON 파일 목록 가져오기
    json_files = [f for f in os.listdir(coco_dir) if f.endswith('.json')]
    
    print(f"총 {len(json_files)}개의 JSON 파일을 변환합니다...")

    # tqdm을 사용하여 진행 상황 표시
    for filename in tqdm(json_files, desc="Converting COCO to YOLOv8"):
        json_path = os.path.join(coco_dir, filename)

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 이미지 정보 추출
            image_info = data.get('images')
            if not image_info:
                print(f"경고: {filename} 파일에 'images' 정보가 없습니다. 건너뜁니다.")
                continue
            
            image_width = image_info.get('width')
            image_height = image_info.get('height')

            # 카테고리 ID를 0부터 시작하는 인덱스로 매핑
            # category_id가 1부터 시작한다고 가정
            categories = data.get('categories', [])
            category_map = {cat['id']: i for i, cat in enumerate(categories)}

            # YOLO 라벨 파일 경로 설정 (JSON 파일명과 동일하게)
            yolo_filename = os.path.splitext(filename)[0] + '.txt'
            yolo_filepath = os.path.join(yolo_dir, yolo_filename)

            with open(yolo_filepath, 'w') as yolo_file:
                annotations = data.get('annotations', [])
                for ann in annotations:
                    category_id = ann.get('category_id')
                    class_index = category_map.get(category_id)

                    if class_index is None:
                        print(f"경고: {filename}에서 알 수 없는 category_id {category_id}를 발견했습니다.")
                        continue

                    # Segmentation 정보 추출
                    segmentation = ann.get('segmentation')
                    if not segmentation or not isinstance(segmentation, list) or not segmentation[0]:
                        continue

                    # 각 폴리곤 좌표를 정규화
                    # segmentation은 [[x1, y1, x2, y2, ...]] 형태일 수 있음
                    for poly in segmentation:
                        normalized_coords = []
                        for i in range(0, len(poly), 2):
                            x = poly[i] / image_width
                            y = poly[i+1] / image_height
                            normalized_coords.append(f"{x:.6f}") # 소수점 6자리까지
                            normalized_coords.append(f"{y:.6f}")
                        
                        # YOLO 형식으로 라인 작성
                        line = f"{class_index} " + " ".join(normalized_coords)
                        yolo_file.write(line + '\n')
                        
        except json.JSONDecodeError:
            print(f"오류: {filename} 파일이 올바른 JSON 형식이 아닙니다.")
        except Exception as e:
            print(f"오류: {filename} 파일 처리 중 에러 발생: {e}")

    print(f"\n변환 완료! {len(json_files)}개의 파일이 {yolo_dir} 폴더에 저장되었습니다.")


# --- 스크립트 사용 예시 ---
if __name__ == '__main__':
    # 📁 1. 입력 폴더 설정 (COCO JSON 파일이 있는 곳)
    # 예시: 'marine_262/Training/label'
    input_coco_dir = '/home1/danny472/Underwater/dataset/marine_262/Validation/label'

    # 📁 2. 출력 폴더 설정 (YOLOv8 라벨 파일이 저장될 곳)
    output_yolo_dir = '/home1/danny472/Underwater/dataset/marine_262/labels/val'

    # 🚀 3. 변환 함수 실행
    convert_coco_to_yolov8_segmentation(input_coco_dir, output_yolo_dir)
