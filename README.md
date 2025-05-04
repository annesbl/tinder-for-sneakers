# Tinder-for-Sneakers (FashionCLIP Version)

A prototype for a swipe-based sneaker recommendation system. Inspired by the user experience of TikTok and Tinder, users can explore sneaker options quickly by swiping through images — finding the right pair in just a few interactions.

This project uses **FashionCLIP** to convert sneaker images and text descriptions into embeddings, and **Annoy** for fast similarity search in the vector space. Users can search using an image, a natural language description, or a combination of both.

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
python build_index.py
```

This script:

* Extracts image embeddings using FashionCLIP
* Creates an Annoy index for fast similarity search

### 2. Search by image, text, or both

Edit and run:

```bash
python search_similar.py
```

You can configure:

* `query_path`: path to a sneaker image (e.g. `query/test_shoe1.jpg`)
* `text_prompt`: a natural language query (e.g. `"white sneaker with red sole"`)
* `alpha`: how to balance image vs. text (e.g. `0.5` for equal weight)

The script will:

* Generate image and/or text embeddings
* Combine them (if both provided)
* Retrieve the most similar sneakers
* Display them side-by-side with `matplotlib`

---

## 🧠 Technologies Used

* [HuggingFace Transformers](https://huggingface.co) — using `patrickjohncyh/fashion-clip`
* [Annoy](https://github.com/spotify/annoy) — for fast approximate nearest neighbor search
* Python 3.10+

---

## 📂 Project Structure

```
tinder-for-sneakers/
├── shoes/                   # Sneaker image database (from Kaggle dataset)
├── query/                   # Images to test queries with
│   └── test_shoe1.jpg       # Example input image
│
├── build_index.py           # Extracts embeddings and builds Annoy index
├── search_similar.py        # Performs image/text/mixed search
│
├── shoe_index.ann           # Saved Annoy index (auto-generated)
├── id_to_filename.npy       # Mapping of Annoy index IDs to filenames
├── requirements.txt
├── README.md
```

---

## 📄 License

For research and prototyping purposes only.
Data usage subject to the license of the referenced Kaggle dataset.

---
