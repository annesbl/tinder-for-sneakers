# Tinder-for-Sneakers (FashionCLIP Version)

A prototype for a swipe-based sneaker recommendation system. Inspired by the user experience of TikTok and Tinder, users can explore sneaker options quickly by swiping through images — finding the right pair in just a few interactions.

This project uses **FashionCLIP** to convert sneaker images into embeddings and **Annoy** for fast similarity search in the vector space. The image data comes from a publicly available **Kaggle dataset**.

---

## 📄 Documentation

You can find the original conceptual summary of the project (in German) here:

📘 [Bildbasiertes Empfehlungssystem mit FashionCLIP (PDF)](Bildbasiertes_Empfehlungssystem_mit_FashionCLIP.pdf)

---

## 📁 Dataset Used

We use the following sneaker dataset from Kaggle:  
**[Nike, Adidas & Converse Image Dataset](https://www.kaggle.com/datasets/die9origephit/nike-adidas-and-converse-imaged)**

It includes sneaker images from well-known brands like Nike, Adidas, and Converse.

---

## ⚙️ Installation

```bash
git clone https://github.com/AleemHussain/tinder-for-sneakers.git
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

This will:

- Embed your query image
- Retrieve the 5 most similar sneakers using Annoy
- Display them side-by-side with `matplotlib`

---

## 🧠 Technologies Used

- [HuggingFace Transformers](https://huggingface.co) — using `patrickjohncyh/fashion-clip`
- [Annoy](https://github.com/spotify/annoy) — for fast approximate nearest neighbor search
- Python 3.10+

---

## 📂 Project Structure

```
tinder-for-sneakers/
├── shoes/                   # Sneaker database (from Kaggle dataset)
├── query/                   # Image to search with
│   └── test_shoe.jpg        # Already given some images to test with
│
├── build_index.py           # Embeds all shoes and builds Annoy index
├── search_similar.py        # Finds similar shoes from query image
│
├── shoe_index.ann           # Generated Annoy index (auto)
├── id_to_filename.npy       # Mapping of image filenames (auto)
├── .gitignore
├── requirements.txt
├── README.md
```

---

## 📄 License

For research and prototyping purposes only.  
Data usage subject to the license of the referenced Kaggle dataset.
