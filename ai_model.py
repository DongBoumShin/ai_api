import cv2
from facenet_pytorch import MTCNN, fixed_image_standardization
import torchvision.models as models
import torch
from torchvision import transforms
import numpy as np
import cv2


class AIModel:
    def __init__(self):
        self.dict_age = {-1: 'unknown', 0: 'tens', 1: 'ybs', 2: 'obs', 3: 'old'}
        self.dict_gender = {-1: 'unknown', 0: 'women', 1: 'men'}
        self.dict_emotion = {-1: 'unknown', 0: 'angry', 1: 'disgusting', 2: 'fear',
                             3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'}

        self.googlenet_age = models.googlenet(pretrained=True)
        model_path = 'age_googlenet_korean_cross_entropy_v2.pth'
        self.googlenet_age.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.googlenet_age.eval()

        self.googlenet_gender = models.googlenet(pretrained=True)
        model_path = 'gender_googlenet_tsingua_cross_entropy_v2.pth'
        self.googlenet_gender.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.googlenet_gender.eval()

        self.googlenet_emotion = models.googlenet(pretrained=True)
        model_path = 'emotion_googlenet_korean_cross_entropy.pth'
        self.googlenet_emotion.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.googlenet_emotion.eval()


    def predict(self, img):
        mtcnn = MTCNN()

        frame = cv2.imread(img)
        transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            fixed_image_standardization
        ])
        boxes, _ = mtcnn.detect(frame)
        if boxes is None:  # 얼굴을 찾은 경우만 예측
            return {'gender': 'unknown', 'age': 'unknown', 'emotion': 'unknown'}
        for box in boxes:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

            # 얼굴 영역 추출 및 크기 조정
            face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = transform(face)
            face = torch.unsqueeze(face, 0)
            with torch.no_grad():
                output_age = self.googlenet_age(face)
                output_gender = self.googlenet_gender(face)
                output_emotion = self.googlenet_emotion(face)

            _, predicted_age = torch.max(output_age, 1)
            _, predicted_gender = torch.max(output_gender, 1)
            _, predicted_emotion = torch.max(output_emotion, 1)
            #default value
            age = gender = emotion = -1
            
            age = predicted_age.item()
            gender = predicted_gender.item()
            emotion = predicted_emotion.item()

        return {'gender': self.dict_gender[gender], 'age': self.dict_age[age], 'emotion': self.dict_emotion[emotion]}
