from facenet_pytorch import MTCNN as mtcnn
import torch

class AIModel:
    def __init__(self):
        self.dict_age = {0: '십대', 1: '청년', 2: '중년', 3: '노년'}
        self.dict_gender = {0: '여성', 1: '남성'}
        self.dict_emotion = {0: '화남', 1: '무표정', 2: '행복', 3: '슬픔', 4: '놀람', 5: '혐오', 6: '공포'}

    def predict(self, data):
        frame = data['image']
        boxes, _ = mtcnn.detect(frame)
        img_face = mtcnn(frame)

        if boxes is not None:  # 얼굴을 찾은 경우만 예측
            img_cropped = img_face.unsqueeze(0)
        else:
            return {'gender': "no face", 'age': "no face", 'emotion': "no face"}
        model_age = torch.load('age_googlenet_tsinghua_cross-entropy.pth')
        model_gender = torch.load('gender_googlenet_tsingua_cross_entropy_v2.pth')
        model_emotion = torch.load('emotion_googlenet_tsinghua_cross_entropy.pth')
        model_age.eval()
        model_emotion.eval()
        model_gender.eval()
        _, age = torch.max(model_age.predict(img_cropped), 1)
        -, gender = torch.max(model_gender.predict(img_cropped), 1)
        -, emotion = torch.max(model_emotion.predict(img_cropped), 1)

        return {'gender': self.dict_gender[gender], 'age': self.dict_age[age], 'emotion': self.dict_emotion[emotion]}
