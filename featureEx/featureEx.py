import torchvision.models as models
from torchvision import transforms
import torch
from torch.nn import Sequential as Sequential
from PIL import Image
import os
import cv2
from tqdm import tqdm
import numpy as np

class ResNetFeatureEx:
    def __init__(self, data_path):
        self.data_path = data_path
        model = models.resnet18(pretrained=True)
        self.model_cut = Sequential(*(list(model.children())[:-1])).eval().to(device)
        for param in self.model_cut.parameters():
            param.requires_grad = False

    def center_crop(imPIL):
        # Adapted from 6.869 Miniplaces1 Exercise
        return transforms.Compose([\
         transforms.Resize((227,227)),\
         transforms.ToTensor(),\
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(\
             imPIL)

    def get_features(self):
        included_extenstions=['webm']
        videoList=[fn for fn in os.listdir(self.data_path) if any([fn.endswith(ext) for ext in included_extenstions])]
        print(videoList)
        
        for v in videoList:
            print('Processing Video: ' + v)
            vName = v.split('.')[0]
            cap = cv2.VideoCapture(self.data_path + v)
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            n = 0
            if not os.path.exists('features/'+vName+'_'+str(nframes)):
                os.makedirs('features/'+vName+'_'+str(nframes))
            with tqdm(total=nframes) as pbar:
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    # print(frame.shape) #(720,1280,3)
                    # cv2.imwrite('tester'+str(i)+'.jpg',frame)
                    pbar.update(1)                    
                    frameTensor = ResNetFeatureEx.center_crop(Image.fromarray(np.uint8(frame)))
                    activations = self.model_cut(frameTensor.unsqueeze(0)).squeeze().numpy()
                    np.save('features/'+vName+'_'+str(nframes)+'/'+str(n), activations)
                    n += 1
                cap.release()
                cv2.destroyAllWindows()

    def get_features_batched(self):
        included_extenstions=['webm']
        videoList=[fn for fn in os.listdir(self.data_path) if any([fn.endswith(ext) for ext in included_extenstions])]
        print(videoList)
        
        for v in videoList:
            print('Processing Video: ' + v)
            vName = v.split('.')[0]
            cap = cv2.VideoCapture(self.data_path + v)
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            n = 0
            vidFrames = torch.empty((0,3,227,227)) #shape of frame batch
            vidFeatures = np.empty((0,512)) #shape of features
            with tqdm(total=nframes) as pbar:
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    pbar.update(1)                    
                    frameTensor = ResNetFeatureEx.center_crop(Image.fromarray(np.uint8(frame)))
                    # print(frameTensor.shape) #(3,227,227)
                    vidFrames = torch.cat((vidFrames,frameTensor.unsqueeze(0)), dim=0)
                    n += 1
                    if n%frameBatch == 0:
                        # print(vidFrames.shape) #(10,3,227,227)
                        vidFrames = vidFrames.to(device)
                        activations = self.model_cut(vidFrames).numpy().squeeze()
                        vidFeatures = np.append(vidFeatures,activations, axis=0)
                        # reset frame batch
                        vidFrames = torch.empty((0,3,227,227)) #shape of frame batch
                cap.release()
                cv2.destroyAllWindows()

            vidFrames = vidFrames.to(device)
            activations = self.model_cut(vidFrames).numpy().squeeze()
            vidFeatures = np.append(vidFeatures,activations, axis=0)

            print('Saving off...')
            np.save('featuresBatched/'+vName+'_'+str(nframes), vidFeatures)
            # print('frameActivations shape:', frameActivations.shape)


def main():
    videoPath = './SumMe/videos/'
    extraction = ResNetFeatureEx(data_path=videoPath)
    extraction.get_features_batched()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frameBatch = 100
    main()