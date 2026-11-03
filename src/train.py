"""Train a six-class CNN on the public NEU hot-rolled steel surface-defect dataset.

Images are downloaded at run time from the public Kaggle dataset mirror. The
dataset is not committed to this repository. Its Kaggle page currently reports
an unspecified licence; users must check the source terms before reuse.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,classification_report,
 confusion_matrix,ConfusionMatrixDisplay,f1_score,precision_score,recall_score)
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
CLASSES=["crazing","inclusion","patches","pitted_surface","rolled_in_scale","scratches"]
EXTS={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def normalize_class(value):
 s=re.sub(r"[^a-z0-9]+","_",value.lower()).strip("_")
 aliases={"rolledinscale":"rolled_in_scale","rolled_in_scale":"rolled_in_scale",
          "pittedsurface":"pitted_surface","pitted_surface":"pitted_surface"}
 return aliases.get(s,s)

def download_and_index():
 root=Path(kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database"))
 rows=[]; seen=set()
 for p in root.rglob("*"):
  if not p.is_file() or p.suffix.lower() not in EXTS: continue
  label=normalize_class(p.parent.name)
  if label not in CLASSES:
   stem=normalize_class(p.stem.split("_")[0])
   label=stem if stem in CLASSES else ""
  if label in CLASSES:
   resolved=str(p.resolve())
   if resolved not in seen: rows.append((resolved,label));seen.add(resolved)
 frame=pd.DataFrame(rows,columns=["path","class"])
 counts=frame["class"].value_counts().to_dict()
 if len(frame)<500 or set(frame["class"].unique())!=set(CLASSES):
  raise RuntimeError(f"Could not index expected six NEU classes. Found {len(frame)} images; class counts={counts}; dataset root={root}")
 if min(counts.values())<50: raise RuntimeError(f"Insufficient images in a class: {counts}")
 return frame,root

def make_ds(paths,labels,image_size,batch_size,training=False):
 ds=tf.data.Dataset.from_tensor_slices((paths.astype(str),labels.astype(np.int32)))
 if training: ds=ds.shuffle(len(paths),seed=42,reshuffle_each_iteration=True)
 def load(path,label):
  raw=tf.io.read_file(path)
  image=tf.io.decode_image(raw,channels=1,expand_animations=False)
  image.set_shape([None,None,1])
  image=tf.image.resize(image,[image_size,image_size])
  image=tf.cast(image,tf.float32)/255.0
  return image,label
 return ds.map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--epochs",type=int,default=12)
 parser.add_argument("--image-size",type=int,default=128)
 parser.add_argument("--batch-size",type=int,default=32)
 args=parser.parse_args()
 tf.keras.utils.set_random_seed(42)
 frame,source_root=download_and_index()
 label_to_id={name:i for i,name in enumerate(CLASSES)}
 y=frame["class"].map(label_to_id).to_numpy()
 paths=frame["path"].to_numpy()
 p_train,p_test,y_train,y_test=train_test_split(paths,y,test_size=.15,random_state=42,stratify=y)
 p_train,p_val,y_train,y_val=train_test_split(p_train,y_train,test_size=(.15/.85),random_state=42,stratify=y_train)
 train_ds=make_ds(p_train,y_train,args.image_size,args.batch_size,True)
 val_ds=make_ds(p_val,y_val,args.image_size,args.batch_size)
 test_ds=make_ds(p_test,y_test,args.image_size,args.batch_size)
 counts=np.bincount(y_train,minlength=len(CLASSES))
 class_weight={i:float(len(y_train)/(len(CLASSES)*counts[i])) for i in range(len(CLASSES))}
 model=tf.keras.Sequential([
  tf.keras.layers.Input(shape=(args.image_size,args.image_size,1)),
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomRotation(.06),
  tf.keras.layers.Conv2D(32,3,padding="same",activation="relu"),tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64,3,padding="same",activation="relu"),tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(128,3,padding="same",activation="relu"),tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dropout(.35),
  tf.keras.layers.Dense(96,activation="relu"),
  tf.keras.layers.Dropout(.2),
  tf.keras.layers.Dense(len(CLASSES),activation="softmax")
 ])
 model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
 callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=2,min_lr=1e-5)]
 history=model.fit(train_ds,validation_data=val_ds,epochs=args.epochs,class_weight=class_weight,callbacks=callbacks,verbose=2)
 probs=model.predict(test_ds,verbose=0);pred=probs.argmax(axis=1)
 report=classification_report(y_test,pred,labels=list(range(len(CLASSES))),target_names=CLASSES,output_dict=True,zero_division=0)
 cm=confusion_matrix(y_test,pred,labels=list(range(len(CLASSES))))
 metrics={"dataset":{"name":"NEU Surface Defect Database / NEU-DET","source":"Kaggle public mirror kaustubhdikshit/neu-surface-defect-database; original dataset by Northeastern University","source_url":"https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database","images_indexed":int(len(frame)),"class_counts":{k:int(v) for k,v in frame["class"].value_counts().to_dict().items()},"classes":CLASSES,"train_images":int(len(y_train)),"validation_images":int(len(y_val)),"test_images":int(len(y_test)),"split":"Stratified 70/15/15, random_state=42","image_size":[args.image_size,args.image_size,1]},"test_metrics":{"accuracy":float(accuracy_score(y_test,pred)),"balanced_accuracy":float(balanced_accuracy_score(y_test,pred)),"macro_precision":float(precision_score(y_test,pred,average="macro",zero_division=0)),"macro_recall":float(recall_score(y_test,pred,average="macro",zero_division=0)),"macro_f1":float(f1_score(y_test,pred,average="macro",zero_division=0)),"weighted_f1":float(f1_score(y_test,pred,average="weighted",zero_division=0)),"confusion_matrix":cm.tolist(),"classification_report":report},"training":{"epochs_requested":args.epochs,"epochs_completed":len(history.history["loss"]),"best_validation_accuracy":float(max(history.history["val_accuracy"])),"best_validation_loss":float(min(history.history["val_loss"]))},"limitations":["NEU-DET is a small controlled benchmark from one acquisition setup.","Kaggle currently reports no explicit dataset licence; check source terms and do not redistribute image files.","Random image split is not a cross-mill or cross-camera generalization test.","This classifier predicts one of six defect categories; it does not localize defect bounding boxes."]}
 out=ROOT/"outputs";out.mkdir(exist_ok=True)
 model.save(out/"quality_inspection_cnn.keras")
 (out/"class_names.json").write_text(json.dumps(CLASSES,indent=2))
 (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
 pd.DataFrame({"image":p_test,"actual_class":[CLASSES[i] for i in y_test],"predicted_class":[CLASSES[i] for i in pred],"confidence":probs.max(axis=1)}).to_csv(out/"test_predictions.csv",index=False)
 disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=CLASSES)
 fig,ax=plt.subplots(figsize=(9,8));disp.plot(ax=ax,xticks_rotation=35,cmap="Blues",colorbar=False);plt.title("NEU-DET — held-out test confusion matrix");plt.tight_layout();plt.savefig(out/"confusion_matrix.png",dpi=160);plt.close()
 plt.figure(figsize=(8,5));plt.plot(history.history["accuracy"],label="Training accuracy");plt.plot(history.history["val_accuracy"],label="Validation accuracy");plt.xlabel("Epoch");plt.ylabel("Accuracy");plt.title("CNN training history");plt.legend();plt.tight_layout();plt.savefig(out/"training_history.png",dpi=160);plt.close()
 plt.figure(figsize=(8,5));plt.bar(CLASSES,[report[c]["f1-score"] for c in CLASSES]);plt.ylabel("Test F1-score");plt.title("Per-class defect classification F1");plt.xticks(rotation=25,ha="right");plt.tight_layout();plt.savefig(out/"per_class_f1.png",dpi=160);plt.close()
 print(json.dumps({"test_metrics":metrics["test_metrics"],"training":metrics["training"]},indent=2))
if __name__=="__main__":main()
