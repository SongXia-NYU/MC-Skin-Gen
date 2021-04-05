# Minecraft Skin Generator

Generating [Minecraft](https://www.minecraft.net/en-us/) Skins with Deep Convolutional Generative Adversarial Networks 
(DCGAN)

The model is main adapted from [this tutorial](https://jovian.ai/aakashns/06b-anime-dcgan) with some modification and hyper-parameter tuning.

Data from [there](https://www.kaggle.com/alxmamaev/minecraft-skins/activity) with preprocessing: remove error files and extract images with 64*64 size
Data preprocessing was done in `extract_img.py`

### 1. Model Training (Trained model included, you can skip to #2)

```commandline
python gan_mc.py
```

### 2. Generate New Image

```commandline
python gan_gen_mc.py
```
This will generate: `mc_skin_generated/generated-images-9999.png`

### 3. Generate a movie showing the training

```commandline
python create_cv.py
```
This will generate: `mc_gen.avi`
 
