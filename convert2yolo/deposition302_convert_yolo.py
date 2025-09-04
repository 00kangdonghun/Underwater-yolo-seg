import os
import json
from tqdm import tqdm

def convert_labelme_to_yolov8(json_dir, yolo_dir):
    """
    LabelMe 형식의 JSON 파일 폴더를 YOLOv8 segmentation 형식으로 변환합니다.

    Args:
        json_dir (str): 원본 LabelMe JSON 파일들이 있는 입력 폴더 경로.
        yolo_dir (str): 변환된 YOLOv8 라벨 파일을 저장할 출력 폴더 경로.
    """
    # 출력 폴더가 없으면 생성
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    # --- 1단계: 모든 클래스 이름 수집 ---
    print("1단계: 전체 클래스 목록을 생성합니다...")
    unique_classes = set()
    for filename in tqdm(json_files, desc="Finding classes"):
        json_path = os.path.join(json_dir, filename)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 'shapes' 키에서 각 객체의 'label'을 가져옴
        for shape in data.get('shapes', []):
            unique_classes.add(shape['label'])
            
    # 클래스 이름을 가나다순으로 정렬하여 고유 인덱스 부여
    class_to_index = {name: i for i, name in enumerate(sorted(list(unique_classes)))}
    
    # 사용자가 dataset.yaml을 만들 수 있도록 결과 출력
    print("\n✅ 클래스 목록 생성이 완료되었습니다. 아래 내용을 dataset.yaml 파일에 사용하세요:")
    print("names:")
    for name, index in class_to_index.items():
        print(f"  {index}: {name}")
    print("-" * 30)

    # --- 2단계: YOLOv8 형식으로 변환 ---
    print("\n2단계: 라벨 파일을 YOLOv8 형식으로 변환합니다...")
    for filename in tqdm(json_files, desc="Converting labels"):
        json_path = os.path.join(json_dir, filename)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 이미지 크기 정보 추출
        img_width = data.get('imageWidth')
        img_height = data.get('imageHeight')

        if not img_width or not img_height:
            print(f"경고: {filename}에서 이미지 크기 정보를 찾을 수 없습니다. 건너뜁니다.")
            continue

        # 출력 파일 경로 설정
        output_filename = os.path.splitext(filename)[0] + '.txt'
        output_path = os.path.join(yolo_dir, output_filename)

        with open(output_path, 'w') as yolo_file:
            for shape in data.get('shapes', []):
                label = shape['label']
                class_index = class_to_index[label]
                
                points = shape.get('points', [])
                if not points:
                    continue

                # 폴리곤 좌표 정규화
                normalized_coords = []
                for point in points:
                    x_norm = point[0] / img_width
                    y_norm = point[1] / img_height
                    normalized_coords.append(f"{x_norm:.6f}") # 소수점 6자리까지
                    normalized_coords.append(f"{y_norm:.6f}")
                
                # YOLO 형식으로 라인 작성
                if normalized_coords:
                    line = f"{class_index} " + " ".join(normalized_coords)
                    yolo_file.write(line + "\n")

    print(f"\n변환 완료! {len(json_files)}개의 파일이 {yolo_dir} 폴더에 저장되었습니다.")


# --- 스크립트 사용 예시 ---
if __name__ == '__main__':
    # 📁 1. 입력 폴더 설정 (JSON 파일이 있는 곳)
    # 예시: 'deposition_302/Training/labels'
    input_json_dir = '/home1/danny472/Underwater/dataset/deposition_302/Validation/labels'

    # 📁 2. 출력 폴더 설정 (YOLOv8 라벨 파일이 저장될 곳)
    output_yolo_dir = '/home1/danny472/Underwater/dataset/deposition_302/labels/val'

    # 🚀 3. 변환 함수 실행
    convert_labelme_to_yolov8(input_json_dir, output_yolo_dir)
