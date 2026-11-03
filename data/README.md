# MVTec AD image data

Download MVTec AD from the [official MVTec page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). Select one category and extract it under this folder. The default category is `bottle`.

Expected category structure:

```text
<category>/
  train/good/
  test/good/
  test/<defect_type>/
  ground_truth/<defect_type>/<image_stem>_mask.png
```

The script trains only on `train/good`, calibrates thresholds using a held-out subset of those normal images, and evaluates on the test images. Defect folders and their mask folders must keep their original names.

**No images are bundled here.** MVTec AD is licensed under CC BY-NC-SA 4.0 and is for non-commercial use. Do not upload the downloaded dataset into this public repository.
