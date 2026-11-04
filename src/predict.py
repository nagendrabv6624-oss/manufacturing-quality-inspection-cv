"""Predict NEU-DET defect classes using the trained HOG/PCA/SVM pipeline."""
import argparse, json
from pathlib import Path
import cv2, joblib, numpy as np
CLASSES=["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--model",default="outputs/neu_det_hog_pca_svm.joblib"); args=ap.parse_args()
    model=joblib.load(args.model); p=Path(args.input); files=[p] if p.is_file() else sorted(x for x in p.rglob("*") if x.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})
    hog=cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
    for f in files:
        im=cv2.imread(str(f),cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        x=hog.compute(cv2.resize(im,(64,64),interpolation=cv2.INTER_AREA)).reshape(1,-1)
        print(f"{f}: {CLASSES[int(model.predict(x)[0])]}")
if __name__=="__main__": main()
