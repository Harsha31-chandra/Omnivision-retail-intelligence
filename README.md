# 👁️ **ORIE: OmniVision Retail Intelligence Engine**

> A state-of-the-art, AI-powered visual search engine for e-commerce, built with CLIP and Pinecone.

![ORIE UI Concept](![Uploading Screenshot 2026-06-28 203036.png…]()
)

---

## **🚀 Overview**

Traditional e-commerce search engines rely on exact keyword matching, which fails when users search for concepts, vibes, or visual styles. **ORIE (OmniVision Retail Intelligence Engine)** solves this using **Semantic Visual Search**. 

By leveraging OpenAI's CLIP (Contrastive Language-Image Pretraining) neural network and a high-performance vector database, ORIE understands the actual mathematical meaning behind a user's text query and instantly matches it against thousands of visual product embeddings.

---

## **🧠 System Architecture**

The engine is built on a decoupled, multimodal architecture:

1. **Data Pipeline (The Brains):** Processed 6,000 raw product images through the local CLIP neural network, converting visual data into 512-dimensional mathematical vectors (embeddings).
2. **Cloud Vector Database:** Uploaded embeddings and product metadata to **Pinecone** for lightning-fast, highly scalable storage.
3. **Inference Backend:** A lightweight **FastAPI** server that vectorizes incoming user search queries in real-time and performs Cosine Similarity searches against the cloud database.
4. **Client Interface:** A vanilla HTML/JS frontend utilizing modern CSS Glassmorphism to asynchronously render product matches.

---

## **🛠️ Tech Stack**

* **AI / Deep Learning:** PyTorch, OpenAI CLIP
* **Database:** Pinecone (Vector Database)
* **Backend:** Python, FastAPI, Uvicorn
* **Data Processing:** NumPy, Pandas
* **Frontend:** HTML5, CSS3 (Glassmorphism), Vanilla JavaScript (Fetch API)

---

## **✨ Core Features**

* **Multimodal AI Search:** Search for clothing using natural language (e.g., "blue denim jeans", "red summer dress").
* **Sub-second Inference:** Real-time vectorization and cloud similarity matching.
* **Responsive UI:** A dynamic, grid-based, frosted-glass interface that requires zero page reloads.

---

## **💻 Local Setup & Installation**

Want to run ORIE on your local machine? Follow these steps:

### **1. Clone the Repository**
```git clone [https://github.com/Harsha31-chandra/Omnivision-retail-intelligence.git]``` ```(https://github.com/Harsha31-chandra/Omnivision-retail-intelligence.git)
cd Omnivision-retail-intelligence```

## **2. Environment Setup**
Ensure you have Python 3.9+ installed, then install the required dependencies:
1. ```pip install fastapi uvicorn pinecone-client torch pandas numpy```
2. ```pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)```

### 3. Add Your Visual Data
Because raw image datasets are too heavy for GitHub, you must provide your own images:

Create a folder named images in the root directory.

Place your product images inside, named by their ID (e.g., 12345.jpg). (Note: The repository includes the pre-computed .npy and .csv metadata for the default Fashion dataset).

##**4.Pinecone Configuration**
Add your Pinecone API key to the backend:

Open main.py and upload_to_pinecone.py.

Replace "YOUR_PINECONE_API_KEY" with your actual key.

##**5. Start the Engines**
You will need two terminal windows running simultaneously.

Terminal 1 (Start the Backend):

Bash
```uvicorn main:app --reload```
Terminal 2 (Start the Frontend):

Bash
```python -m http.server 8080```
Open your browser and navigate to http://localhost:8080.

###***🔮 Future Roadmap***
- [ ] Image-to-Image Search: Allow users to upload a photo of a shirt they like to find visually similar items in the catalog.

- [ ] Cloud Deployment: Containerize the backend with Docker and deploy to AWS/GCP or Vercel.

- [ ] Image-to-Image SearchAdvanced Filtering: Combine semantic search with hard metadata filters (e.g., "Show me red dresses under $50").

- [ ] Image-to-Image Search User Authentication: Implement secure login for personalized search history and saved items.

###***📄 License***
This project is open-source and available under the MIT License.
