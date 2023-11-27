import cv2
from facenet_pytorch import MTCNN, fixed_image_standardization
import torchvision.models as models
import torch
from torchvision import transforms
import numpy as np

# 모델 초기화
mtcnn = MTCNN()
googlenet_age = models.googlenet(pretrained=True)
model_path = 'C:\\github_repo\\CapstoneDesign\\age_estimation\\googlenet\\tsinghua\\googlenet_tsinghua_cross_entropy_v2.pth'
googlenet_age.load_state_dict(torch.load(model_path))
googlenet_age.eval()

googlenet_gender = models.googlenet(pretrained=True)
model_path = 'C:\\github_repo\\CapstoneDesign\\gender_estimation\\googlenet\\tsinghua\\googlenet_tsinghua_cross_entropy_v2.pth'
googlenet_gender.load_state_dict(torch.load(model_path))
googlenet_gender.eval()

googlenet_emotion = models.googlenet(pretrained=True)
model_path = 'C:\\github_repo\\CapstoneDesign\\emotion_estimation\\googlenet\\korean\\emotion_googlenet_korean_cross_entropy.pth'
googlenet_emotion.load_state_dict(torch.load(model_path))
googlenet_emotion.eval()

emotion_dict = {0: 'angry', 1: 'disgusting', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


# 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

# 이미지 변환
transform = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    fixed_image_standardization
])

while cap.isOpened():
    ret, frame = cap.read()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # MTCNN을 통해 얼굴 감지
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:  # 얼굴을 찾은 경우만 예측
        for box in boxes:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        
            # 얼굴 영역 추출 및 크기 조정
            face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = transform(face)
            face = torch.unsqueeze(face, 0)

            # GoogleNet에 이미지 입력 및 예측
            with torch.no_grad():
                output_age = googlenet_age(face)
                output_gender = googlenet_gender(face)
                output_emotion = googlenet_emotion(face)
            
            _, predicted_age = torch.max(output_age, 1)
            _, predicted_gender = torch.max(output_gender, 1)
            _, predicted_emotion = torch.max(output_emotion, 1)

            # 예측 결과 출력
            
            if predicted_age.item() == 0:
                predicted_age_result = '0~19'
            elif predicted_age.item() == 1:
                predicted_age_result = '20~39'
            else:
                predicted_age_result = '40~59'

            cv2.putText(frame, f"Age: {predicted_age_result}s", (int(box[0]), int(box[1]-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
            cv2.putText(frame, f"Gender: {'male' if predicted_gender.item() == 1 else 'female'}", (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
            cv2.putText(frame, f"Emotion: {emotion_dict[predicted_emotion.item()]}", (int(box[0]), int(box[1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
        
        cv2.imshow("Webcam", frame)
    else:
        print("No face detected in the image.")

cap.release()
cv2.destroyAllWindows()
