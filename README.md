# IDENTIFICATION OF DIFFERENT MEDICINAL PLANTS USING MOBILENET
***by Jayaram Bantumilli ***

# Deep Learning & Computer Vision
tensorflow==2.13.0
keras==2.13.1
numpy==1.24.4
pandas==2.1.0
scikit-learn==1.3.0
opencv-python==4.8.1.78
matplotlib==3.8.0
seaborn==0.12.2
Pillow==10.1.0

# Web Deployment
Flask==2.3.4
flask-cors==4.0.0

# Utilities
tqdm==4.66.1
h5py==3.9.0

# System Requirements
1. Operating System: Windows Only  
2. Processor: i5 and above 
3. RAM: 8GB and above 
4. Hard Disk: 128 GB 
***Make sure your system Must have atleast 8GB RAM*** Because we are training pre-trained Model May damage system with low RAM.

# To reproduce the results for the Medicinal Plant Identification project using MobileNet.

# Step-1: Clone the Repository
<pre> 
``` 
git clone https://github.com/Jayaram13690/MedicinalPlantIdentification.git 
cd MedicinalPlantIdentification
``` </pre>

# Step-2: Set Up Python Environment

1. Make sure you have Python 3.9+ installed.
2. Create a virtual environment (optional but recommended)
3. Install dependencies

# Step-3 Train Model
1. Open Model.ipynb or your training notebook.
2. You can modify Image Size, Batch Size, Epochs, and Learning Rate.
3. Start Training by executing the file if you are using Jupiter Notebook (Execute each cell).
4. Or using Python CLI. Use following Command:

Execute `python Model.ipynb`;

- MobileNetV2 is used as a pretrained base.
- Data augmentation (rotation, flip, zoom, brightness) is applied automatically.
- EarlyStopping and ReduceLROnPlateau are enabled to optimize training.

**Outputs:** Trained model (model.h5) in the Same dir folder.

# Step-4: Deployment via Flask:
1. Create `app.py`

<pre> 
```

from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('model/model.h5')
class_names = ['Class_1', 'Class_2', 'Class_3', ...]

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    img = image.load_img(img_file, target_size=(224, 224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    return jsonify({'prediction': class_names[class_idx]})

if __name__ == '__main__':
    app.run(debug=True)

```
<prev>

2. run server:
 
`python app.py`

