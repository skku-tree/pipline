import os
import cv2

def rotate_image(image):
    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
    rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated_90,rotated_180,rotated_270
    
    

def process_image_in_canny(file_path: str, output_dir):
    # 파일 이름 및 확장자 추출
    file_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    # 이미지 파일 읽기
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_90, img_180, img_270 = rotate_image(image)
    rotate_path = file_path+"_rotate"
    output_path_90 = os.path.join(rotate_path, f"{file_name_without_ext}_rotated_90.jpg")
    output_path_180 = os.path.join(rotate_path, f"{file_name_without_ext}_rotated_180.jpg")
    output_path_270 = os.path.join(rotate_path, f"{file_name_without_ext}_rotated_270.jpg")
    print(f"이미지 변환 완료: {rotate_path}")
    cv2.imwrite(output_path_90, img_90)
    cv2.imwrite(output_path_180, img_180)
    cv2.imwrite(output_path_270, img_270)
    # Canny Edge 변환
    edges = cv2.Canny(image, 100, 200)
    canny_90, canny_180, canny_270 = rotate_image(edges)

    
    # canny 저장할 경로 생성
    output_path = os.path.join(output_dir, f"{file_name_without_ext}_edge.jpg")
      # 이미지 저장
    cv2.imwrite(output_path, edges)
    print(f"이미지 변환 완료: {output_path}")
    
    # canny 회전된 이미지 저장
    output_path_90 = os.path.join(output_dir + "_rotate", f"{file_name_without_ext}_edge_rotated_90.jpg")
    output_path_180 = os.path.join(output_dir + "_rotate", f"{file_name_without_ext}_edge_rotated_180.jpg")
    output_path_270 = os.path.join(output_dir + "_rotate", f"{file_name_without_ext}_edge_rotated_270.jpg")
    cv2.imwrite(output_path_90, canny_90)
    cv2.imwrite(output_path_180, canny_180)
    cv2.imwrite(output_path_270, canny_270)

def process_images_in_directory(input_dir, output_dir):
    # 입력 디렉토리의 모든 파일 검색
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        
        # 파일인지 확인
        if os.path.isfile(file_path):
            # 이미지 파일인지 확인
            if any(file_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                # 이미지 처리
                process_image_in_canny(file_path, output_dir)

# 입력 및 출력 디렉토리 경로 설정
input_directory = '/Users/minseo/Desktop/2023-01/skt-ai-fellowship/k-ium/images/train_set'
output_directory = '/Users/minseo/Desktop/2023-01/skt-ai-fellowship/k-ium/images/canny_train_set'

# 이미지 변환 처리 실행
process_images_in_directory(input_directory, output_directory)
