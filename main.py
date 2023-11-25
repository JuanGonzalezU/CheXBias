import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
sys.path.append('/home/juandres/aml/CheXBias/src/')
from general_functions import *

# Class for colors
class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'

# Initialize parser
parser = argparse.ArgumentParser()

# Set argument
parser.add_argument('--mode',
                    type=str,
                    default='test',  
                    help='Choose mode (demo or test)')

parser.add_argument('--img',
                    type=str,
                    default='/home/juandres/aml/CheXBias/data_new/processed/test/patient19396_study3_view1_frontal_Female_33_Frontal_PA.jpg',  
                    help='Choose mode (demo or test)')

# Get argument
args = parser.parse_args()

# Set classes 
args.classes = ['Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture']

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model
model = CustomResNet(num_classes=len(args.classes)).to(device)

# Load weight (choose any of the selected models)
model.load_state_dict(torch.load('/home/juandres/aml/CheXBias/models/Experiment_2/age/group_selection_3/best_model.pth'))

# Get the sample image

if args.mode == 'test':
    img = Image.open('/home/juandres/aml/CheXBias/data_new/processed/test/patient19396_study3_view1_frontal_Female_33_Frontal_PA.jpg').convert('L')  # 'L' mode for grayscale
else:
    img = Image.open(args.img)

# Define pre-process
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to specified number of channels
    transforms.ToTensor()
    ])

# Pre process the image
img = preprocess(img)

# Pass it through the model 
with torch.inference_mode():

    # Turn model to eval mode
    model.eval()

    # Predict
    predictions = model(img.unsqueeze(0).to(device))

# Output predictions
print('\nCheXBias Results =D \n')
for i,pred in enumerate(torch.round(predictions).tolist()[0]):    
    if pred == 1:
        print(bcolors.WARNING + f' - {args.classes[i]} detected!' + bcolors.ENDC)
    else:
        print(bcolors.OKGREEN + f' - {args.classes[i]} not detected!' + bcolors.ENDC)
print('\nThanks for using! Verify results with professional')