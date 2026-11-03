from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,tensorflow as tf
ROOT=Path(__file__).resolve().parents[1]
EXTS={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True,help="Image file or directory");p.add_argument("--output",default="outputs/new_predictions.csv");p.add_argument("--image-size",type=int,default=128);a=p.parse_args()
 model_path=ROOT/"outputs"/"quality_inspection_cnn.keras";class_path=ROOT/"outputs"/"class_names.json"
 if not model_path.exists() or not class_path.exists():raise FileNotFoundError("Train first with python src/train.py")
 src=Path(a.input);files=[src] if src.is_file() else sorted(x for x in src.rglob("*") if x.is_file() and x.suffix.lower() in EXTS)
 if not files:raise ValueError("No supported images found")
 model=tf.keras.models.load_model(model_path);classes=json.loads(class_path.read_text());rows=[]
 for path in files:
  raw=tf.io.read_file(str(path));img=tf.io.decode_image(raw,channels=1,expand_animations=False);img=tf.image.resize(img,[a.image_size,a.image_size]);img=tf.cast(img,tf.float32)/255
  prob=model.predict(tf.expand_dims(img,0),verbose=0)[0];idx=int(np.argmax(prob))
  rows.append({"image":str(path),"predicted_defect":classes[idx],"confidence":float(prob[idx])})
 out=pd.DataFrame(rows);dest=Path(a.output);dest=dest if dest.is_absolute() else ROOT/dest;dest.parent.mkdir(parents=True,exist_ok=True);out.to_csv(dest,index=False);print(out.to_string(index=False))
if __name__=="__main__":main()
