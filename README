# CLIP & KNN Proof of Concepts

## Diagram

https://excalidraw.com/#json=5St_RN3enQSrtKdrbvpeh,cnQW3K5rzaWnGftLRn7r7Q

## Env

```
python -m venv clip-knn-env
clip-knn-env\Scripts\activate.bat
```

## CLIP

### Which CUDA version

```
nvidia-smi
```

### Instalation (takes a while)
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip3 install ftfy regex tqdm
pip3 install git+https://github.com/openai/CLIP.git
```

### Execution
```
python clip-pof.py
```

### Promising alternative
https://github.com/mlfoundations/open_clip

## KNN

### Dataset

Move the COCO JSON dataset to the root of the project and call it `coco-dataset`. `coco-dataset` should have `_annotations.coco.json` and the images.

### Instalation
```
pip3 install torch torchvision scikit-learn pillow tqdm numpy
```

### Execution
```
python knn-pof.py
```
