import torch
import os
import cv2
import pandas as pd
from torch.utils.data import DataLoader
from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
import argparse
import numpy as np

def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dir = args.data_root
    txt_path = args.test_txt
    test_dataset = OxfordPetDataset(test_dir, txt_path=txt_path, mode="test")
    test_loader = DataLoader(test_dataset)
    
    model = ResNet34_UNet().to(device) if args.model == "res_unet" else UNet().to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for i, images in enumerate(test_loader):
            images = images.to(device)
            logits = model(images)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            pred_mask = preds.squeeze().cpu().numpy()
            
            filename = test_dataset.filenames[i]
            
            img_path = os.path.join(test_dir, "images", f"{filename}.jpg")
            
            orig_img = cv2.imread(img_path)
            orig_h, orig_w = orig_img.shape[:2]
            
            pred_mask_orig = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            rle_string = rle_encode(pred_mask_orig)
            
            image_id = filename.split(".")[0]
            
            results.append({
                "image_id": image_id,
                "encoded_mask": rle_string
            })
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(test_loader)} images")
                    
        df = pd.DataFrame(results)
        df.to_csv(f"submission_{args.model}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../dataset/oxford-iiit-pet", help="Path to the dataset")
    parser.add_argument("--test_txt", default="../dataset/oxford-iiit-pet/test_res_unet.txt", help="Path to test.txt")
    parser.add_argument("--model", type=str, default="res_unet", choices=["unet", "res_unet"], help="Model architecture to use")
    parser.add_argument("--model_path", default="../saved_models/best_RES_UNET.pth", help="Path to the saved model weights")
    args = parser.parse_args()
    main(args)