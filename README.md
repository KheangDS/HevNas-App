# 🧠 HevNas-App

HevNas-App is a simple and lightweight web application that uses deep learning to classify food images. Built with **Streamlit** and **PyTorch**, it allows users to upload an image and quickly get a prediction of the food category — such as Meat, Noodles/Pasta, Rice, Soup, or Vegetables/Fruits.


## ✨ Features

- 🖼 **Image Classification in Seconds**  
  Upload an image through the web interface and get instant predictions.

- 🔍 **Confidence Scores**  
  See how confident the model is about its prediction.

- 🚀 **Efficient Model Loading**  
  Uses Streamlit caching (`st.cache_resource`) to improve performance.

- ⚙️ **Custom CNN Model**  
  Powered by a custom-built `KlebJeb` architecture using PyTorch, including Batch Normalization layers.

---

## 🛠 Tech Stack

| Component         | Technology            |
|------------------|----------------------|
| Frontend         | Streamlit            |
| Deep Learning    | PyTorch, Torchvision |
| Image Processing | Pillow (PIL)         |
| Language         | Python 3             |

---

## 📥 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KheangDS/HevNas-App
   cd HevNas-App
   ```

2. Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
3. Usage
   Run the app locally with:
   ```bash
    streamlit run app.py
   ```


📂 Project Structure
├── app.py              # Streamlit app entry point
├── requirements.txt    # Dependencies
├── data/
│   └── image_utils.py  # Image preprocessing utilities
├── inference/
│   └── predict.py      # Prediction logic
├── model/
│   ├── model.py        # CNN architecture (KlebJeb)
│   └── load_model.py   # Model loading utilities
└── models/
    └── KlebJeb.pt      # Pre-trained weights
