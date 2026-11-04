"""Train HOG + PCA + RBF-SVM on an extracted NEU-DET archive."""
import argparse, json
from pathlib import Path
import cv2, joblib, numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSES=["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]
def load_split(root, split):
    X=[]; y=[]; names=[]
    hog=cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
    for label,name in enumerate(CLASSES):
        folder=root/split/"images"/name
        if not folder.exists(): raise FileNotFoundError(folder)
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".jpg",".jpeg",".png",".bmp"}: continue
            image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if image is None: continue
            image=cv2.resize(image,(64,64),interpolation=cv2.INTER_AREA)
            X.append(hog.compute(image).ravel()); y.append(label); names.append(path.name)
    return np.asarray(X,dtype=np.float32),np.asarray(y),names
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True,help="Extracted NEU-DET directory containing train/ and validation/")
    ap.add_argument("--output-dir",default="outputs")
    args=ap.parse_args(); root=Path(args.data_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    X,y,_=load_split(root,"train"); Xv,yv,names=load_split(root,"validation")
    vi,ti=train_test_split(np.arange(len(yv)),test_size=.5,random_state=42,stratify=yv)
    model=Pipeline([("scale",StandardScaler()),("pca",PCA(n_components=128,random_state=42)),("svc",SVC(C=10,kernel="rbf",gamma="scale",class_weight="balanced"))])
    model.fit(X,y); pred=model.predict(Xv[ti]); truth=yv[ti]
    report=classification_report(truth,pred,labels=range(6),target_names=CLASSES,output_dict=True,zero_division=0)
    metrics={"dataset":"NEU-DET uploaded archive","images_total":int(len(y)+len(yv)),"train_images":int(len(y)),"validation_images":int(len(vi)),"test_images":int(len(ti)),"classes":CLASSES,"split":"Supplied train split for fitting; supplied validation split stratified 50/50 into validation and final test subsets","features":"64x64 grayscale HOG (1764 features)","model":"StandardScaler + PCA(128) + RBF SVC, C=10, class_weight=balanced","seed":42,"test_metrics":{"accuracy":float(accuracy_score(truth,pred)),"macro_precision":float(precision_score(truth,pred,average="macro")),"macro_recall":float(recall_score(truth,pred,average="macro")),"macro_f1":float(f1_score(truth,pred,average="macro")),"classification_report":report},"confusion_matrix":confusion_matrix(truth,pred).tolist()}
    joblib.dump(model,out/"neu_det_hog_pca_svm.joblib",compress=9)
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    (out/"class_names.json").write_text(json.dumps(CLASSES,indent=2))
    print(json.dumps(metrics["test_metrics"],indent=2))
if __name__=="__main__": main()
