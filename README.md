# Tinder-for-Sneakers

A prototype for a swipe-based sneaker recommendation system. Inspired by the user experience of TikTok and Tinder, users can explore sneaker options quickly by swiping through images — finding the right pair in just a few interactions.

This project uses **FashionCLIP** to convert sneaker images into embeddings and **Annoy** for fast similarity search in the vector space. The image data comes from a publicly available **Kaggle dataset**.

---

## 📁 Dataset Used

We use the following sneaker dataset from Kaggle:  
**[Nike, Adidas & Converse Image Dataset](https://www.kaggle.com/datasets/die9origephit/nike-adidas-and-converse-imaged)**

It includes sneaker images from well-known brands like Nike, Adidas, and Converse.

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/tinder-for-sneakers.git
cd tinder-for-sneakers
```

```bash
# (Optional) Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

```bash
# Install dependencies
pip install -r requirements.txt
```

### Prerequisites

- A valid Kaggle API key (`kaggle.json`) is required.  
  [Get it from your Kaggle account](https://www.kaggle.com/account) and place it in `~/.kaggle/kaggle.json`.

---

## 🚀 How to Run

### 1. Download images and create embeddings

```bash
python encode_images.py
```

This script:
- Automatically downloads the Kaggle dataset
- Converts and saves images into the `data/` directory (as `.jpg`)
- Extracts image embeddings using FashionCLIP
- Saves them in the `embeddings/` directory

### 2. Build the Annoy index

```bash
python build_index.py
```

This creates an Annoy index from the image embeddings and stores it in `annoy_index/fashion.ann`.

### 3. Search for similar images (Example)

```bash
python search_similar.py "test_images/converse/1.jpg"
```

This returns the top 5 most similar sneaker images based on the input image.

---

## 🧠 Technologies Used

- [HuggingFace Transformers](https://huggingface.co) — using `patrickjohncyh/fashion-clip`
- [Annoy](https://github.com/spotify/annoy) — for fast approximate nearest neighbor search
- [KaggleHub](https://github.com/ishant1609/KaggleHub) — for automatic dataset download
- Python 3.10+

---

## 📂 Project Structure

```
tinder-for-sneakers/
├── data/                  # Raw images (from Kaggle)
├── embeddings/            # Saved image embeddings (.npy)
├── annoy_index/           # Annoy index files (.ann) + ID map
├── test_images/           # Custom test images (e.g., your own sneaker)
│   ├── adidas/
│   ├── converse/
│   ├── nike/
│ 
├── encode_images.py       # Kaggle download + image conversion + embedding
├── build_index.py         # Build Annoy index from embeddings
├── search_similar.py      # Find similar images based on input
│
├── README.md              # Project description & instructions
├── requirements.txt       # Dependencies
├── .gitignore             # Ignored files/folders
```

---

## 📄 License

For research and prototyping purposes only.  
Data usage subject to the license of the referenced Kaggle dataset.
