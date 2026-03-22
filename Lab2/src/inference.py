import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dir = '../dataset/oxford-iiit-pet'
    test_dataset = OxfordPetDataset(test_dir, txt_name="test_unet.txt", mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load('../saved_models/best_UNet.pth'))
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for i, images in enumerate(test_loader):
            images = images.to(device)
            logits = model(images)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            pred_mask = preds.squeeze().cpu().numpy()
            rle_string = rle_encode(pred_mask)
            
            filename = test_dataset.filenames[i]
            image_id = filename.split('.')[0]
            
            results.append({
                'image_id': image_id,
                'encoded_mask': rle_string
            })
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(test_loader)} images")
                
    df = pd.DataFrame(results)
    df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()