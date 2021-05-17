import torchvision.models as models
from torchvision import transforms
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
        self.model_cut = Sequential(*(list(model.children())[:-2])).eval()
        for param in self.model_cut.parameters():
            param.requires_grad = False

    def generate_featuremap_unit(self,im):
        # Adapted from 6.869 Miniplaces1 Exercise
        # Extract activation from model
        # Mark the model as being used for inference
        # # Crop the image
        # im = center_crop(im_input)
        # Place the image into a batch of size 1, and use the model to get an intermediate representation
        activations = self.model_cut(im.unsqueeze(0))
        # Print the shape of our representation
        print(activations.size())
        # Extract the only result from this batch, and take just the `unit_id`th channel
        # Return this channel
        return activations.squeeze()

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
            # frameActivations = np.empty((512,8,8)) #shape of activations
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
                    # frameActivations = np.concatenate((frameActivations,activations), axis=0)
                    np.save('features/'+vName+'_'+str(nframes)+'/'+str(n), activations)
                    n += 1
                cap.release()
                cv2.destroyAllWindows()
            # print('frameActivations shape:', frameActivations.shape)


def main():
    videoPath = './SumMe/videos/'
    extraction = ResNetFeatureEx(data_path=videoPath)
    extraction.get_features()


if __name__ == '__main__':
	main()