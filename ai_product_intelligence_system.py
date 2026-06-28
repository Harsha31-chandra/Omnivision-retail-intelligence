import subprocess
import sys
import os

def install_dependencies():
    packages = [
        "ftfy",
        "regex",
        "git+https://github.com/openai/CLIP.git",
    ]
    for pkg in packages:
        print(f"[SETUP] Installing: {pkg}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print("[SETUP] All dependencies installed successfully.\n")

install_dependencies()

import os
import warnings
import time
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import clip

warnings.filterwarnings("ignore")


class FashionImageDataset(Dataset):

    def __init__(self, image_paths: List[str], preprocess: callable):
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.valid_indices = [
            i for i, p in enumerate(image_paths)
            if os.path.exists(str(p))
        ]
        invalid_count = len(image_paths) - len(self.valid_indices)
        if invalid_count > 0:
            print(f"  [DATA] Warning: {invalid_count} images not found on disk and will be skipped.")
        print(f"  [DATA] Valid images ready for embedding: {len(self.valid_indices)}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.valid_indices[idx]
        img_path = str(self.image_paths[real_idx])
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.preprocess(image)
            return image_tensor, real_idx
        except Exception as e:
            print(f"  [DATA] Error loading {img_path}: {e}")
            return None, real_idx


class AIProductIntelligenceSystem:

    def __init__(
        self,
        dataset_base_path: str = "/kaggle/input/fashion-product-images-small",
        sample_size: Optional[int] = None,
        batch_size: int = 256,
        cache_embeddings_path: str = "clip_embeddings.npy",
        cache_metadata_path: str = "cleaned_metadata.csv",
        num_catalog_clusters: int = 500,
    ):
        print("=" * 80)
        print(" AI PRODUCT INTELLIGENCE SYSTEM — Initialization")
        print("=" * 80)

        self.dataset_base_path = dataset_base_path
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.cache_embeddings_path = cache_embeddings_path
        self.cache_metadata_path = cache_metadata_path
        self.num_catalog_clusters = num_catalog_clusters

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[DEVICE] Using: {self.device}")
        if self.device.type == "cuda":
            print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[DEVICE] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        self.model = None
        self.preprocess = None
        self.df = None
        self.embeddings = None
        self.pca_embeddings = None
        self.cluster_labels = None
        self.deduplicated_catalog_df = None

    def _auto_detect_dataset_path(self) -> Tuple[str, str]:
        print("  [AUTO-DETECT] Scanning /kaggle/input/ for dataset files...")

        kaggle_input = "/kaggle/input"
        if os.path.isdir(kaggle_input):
            print(f"  [AUTO-DETECT] Contents of {kaggle_input}:")
            for item in os.listdir(kaggle_input):
                print(f"    → {item}")

        csv_path = os.path.join(self.dataset_base_path, "styles.csv")
        images_dir = os.path.join(self.dataset_base_path, "images")

        if os.path.exists(csv_path) and os.path.isdir(images_dir):
            print(f"  [AUTO-DETECT] Found at configured path: {self.dataset_base_path}")
            return csv_path, images_dir

        print(f"  [AUTO-DETECT] Not found at: {self.dataset_base_path}")
        print(f"  [AUTO-DETECT] Searching recursively under {kaggle_input}...")

        found_csv = None
        for root, dirs, files in os.walk(kaggle_input):
            if "styles.csv" in files:
                found_csv = os.path.join(root, "styles.csv")
                found_images = os.path.join(root, "images")
                if os.path.isdir(found_images):
                    print(f"  [AUTO-DETECT] ✓ Found styles.csv at: {found_csv}")
                    print(f"  [AUTO-DETECT] ✓ Found images dir at: {found_images}")
                    self.dataset_base_path = root
                    return found_csv, found_images

        if found_csv:
            parent = os.path.dirname(found_csv)
            for sibling in os.listdir(os.path.dirname(found_csv)):
                sibling_path = os.path.join(parent, sibling)
                if os.path.isdir(sibling_path) and "image" in sibling.lower():
                    print(f"  [AUTO-DETECT] ✓ Found styles.csv at: {found_csv}")
                    print(f"  [AUTO-DETECT] ✓ Found images dir at: {sibling_path}")
                    self.dataset_base_path = parent
                    return found_csv, sibling_path

        if os.path.isdir(kaggle_input):
            print(f"\n  [AUTO-DETECT] Full directory tree under {kaggle_input}:")
            for root, dirs, files in os.walk(kaggle_input):
                depth = root.replace(kaggle_input, "").count(os.sep)
                indent = "    " * depth
                print(f"  {indent}📁 {os.path.basename(root)}/")
                for f in files[:10]:
                    print(f"  {indent}  📄 {f}")
                if len(files) > 10:
                    print(f"  {indent}  ... and {len(files) - 10} more files")

        raise FileNotFoundError(
            f"Could not find 'styles.csv' + 'images/' directory anywhere under {kaggle_input}. "
            f"Please ensure the dataset 'paramaggarwal/fashion-product-images-small' is added to your Kaggle notebook."
        )

    def load_and_clean_data(self) -> pd.DataFrame:
        print("\n" + "-" * 70)
        print("[STEP 1] Loading and Cleaning Dataset")
        print("-" * 70)

        csv_path, images_dir = self._auto_detect_dataset_path()

        print(f"  [LOAD] Reading: {csv_path}")
        self.df = pd.read_csv(
            csv_path,
            on_bad_lines="skip",
            engine="python",
            dtype={"id": str},
        )
        print(f"  [LOAD] Raw rows loaded: {self.df.shape[0]}")
        print(f"  [LOAD] Columns: {list(self.df.columns)}")

        self.df["image_path"] = self.df["id"].apply(
            lambda x: os.path.join(images_dir, f"{x}.jpg")
        )

        mask_exists = self.df["image_path"].apply(os.path.exists)
        before_count = len(self.df)
        self.df = self.df[mask_exists].reset_index(drop=True)
        removed_count = before_count - len(self.df)
        print(f"  [CLEAN] Removed {removed_count} rows with missing images.")
        print(f"  [CLEAN] Valid products remaining: {len(self.df)}")

        critical_cols = ["id", "productDisplayName", "masterCategory", "subCategory"]
        available_critical = [c for c in critical_cols if c in self.df.columns]
        self.df = self.df.dropna(subset=available_critical).reset_index(drop=True)
        print(f"  [CLEAN] After dropping NaN in critical columns: {len(self.df)}")

        if self.sample_size is not None and self.sample_size < len(self.df):
            print(f"  [SAMPLE] Downsampling to {self.sample_size} items for efficiency.")
            self.df = self.df.sample(
                n=self.sample_size, random_state=42
            ).reset_index(drop=True)

        print(f"  [RESULT] Final dataset shape: {self.df.shape}")
        print(f"  [RESULT] Category distribution (top 10):")
        if "subCategory" in self.df.columns:
            print(self.df["subCategory"].value_counts().head(10).to_string(header=False))

        return self.df

    def load_clip_model(self) -> None:
        print("\n" + "-" * 70)
        print("[STEP 2] Loading CLIP Model (ViT-B/32)")
        print("-" * 70)

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        self.visual_encoder = self.model.visual
        if torch.cuda.device_count() > 1:
            print(f"  [DEVICE] Auto-detected {torch.cuda.device_count()} GPUs. Enabling DataParallel!")
            self.visual_encoder = torch.nn.DataParallel(self.visual_encoder)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  [MODEL] Architecture: ViT-B/32")
        print(f"  [MODEL] Total Parameters: {total_params:,}")
        print(f"  [MODEL] Embedding Dimension: 512")
        print(f"  [MODEL] Device: {self.device}")
        print(f"  [MODEL] Model loaded successfully.")

    def extract_embeddings(self) -> np.ndarray:
        print("\n" + "-" * 70)
        print("[STEP 3] Extracting Visual Embeddings")
        print("-" * 70)

        if (
            os.path.exists(self.cache_embeddings_path)
            and os.path.exists(self.cache_metadata_path)
        ):
            print("  [CACHE] Found cached embeddings and metadata!")
            print(f"  [CACHE] Loading embeddings from: {self.cache_embeddings_path}")
            print(f"  [CACHE] Loading metadata from: {self.cache_metadata_path}")

            cached_embeddings = np.load(self.cache_embeddings_path)
            cached_df = pd.read_csv(self.cache_metadata_path, dtype={"id": str})

            print(f"  [CACHE] Embeddings shape: {cached_embeddings.shape}")
            print(f"  [CACHE] Metadata shape: {cached_df.shape}")

            if cached_embeddings.shape[0] != len(cached_df):
                print("  [CACHE] WARNING: Shape mismatch! Re-computing embeddings...")
            else:
                self.embeddings = cached_embeddings
                self.df = cached_df
                print("  [CACHE] Cache loaded successfully. Skipping inference.")
                if self.model is None:
                    print("  [CACHE] Loading CLIP model (required for text search)...")
                    self.load_clip_model()
                return self.embeddings

        def safe_collate_fn(batch):
            valid = [(img, idx) for img, idx in batch if img is not None]
            if len(valid) == 0:
                return torch.empty(0, 3, 224, 224), torch.tensor([], dtype=torch.long)
            images, indices = zip(*valid)
            return torch.stack(images), torch.tensor(indices, dtype=torch.long)

        print(f"  [EMBED] Building batched data pipeline (batch_size={self.batch_size})...")
        dataset = FashionImageDataset(
            image_paths=self.df["image_path"].tolist(),
            preprocess=self.preprocess,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,  # Increased to 4 to feed GPUs faster
            pin_memory=True if self.device.type == "cuda" else False,
            drop_last=False,
            collate_fn=safe_collate_fn,
        )

        all_embeddings = np.zeros((len(self.df), 512), dtype=np.float32)
        valid_mask = np.zeros(len(self.df), dtype=bool)

        total_batches = len(dataloader)
        start_time = time.time()
        print(f"  [EMBED] Starting inference: {len(dataset)} images in {total_batches} batches...")

        with torch.no_grad():
            for batch_idx, (images, indices) in enumerate(dataloader):
                if images.size(0) == 0:
                    continue

                images = images.to(self.device)
                
                # Use parallelized visual encoder (mimics encode_image but supports DataParallel)
                image_features = self.visual_encoder(images.type(self.model.dtype)).float()
                
                image_features = F.normalize(image_features, p=2, dim=1)

                features_np = image_features.cpu().numpy().astype(np.float32)
                for i, idx in enumerate(indices.numpy()):
                    all_embeddings[idx] = features_np[i]
                    valid_mask[idx] = True

                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                    elapsed = time.time() - start_time
                    images_done = min((batch_idx + 1) * self.batch_size, len(dataset))
                    speed = images_done / elapsed if elapsed > 0 else 0
                    print(
                        f"  [EMBED] Batch {batch_idx + 1}/{total_batches} | "
                        f"{images_done}/{len(dataset)} images | "
                        f"{speed:.0f} img/s | "
                        f"Elapsed: {elapsed:.1f}s"
                    )

        self.df = self.df[valid_mask].reset_index(drop=True)
        self.embeddings = all_embeddings[valid_mask]

        elapsed_total = time.time() - start_time
        print(f"\n  [EMBED] Inference complete!")
        print(f"  [EMBED] Final embeddings shape: {self.embeddings.shape}")
        print(f"  [EMBED] Total time: {elapsed_total:.1f}s")
        print(f"  [EMBED] Throughput: {self.embeddings.shape[0] / elapsed_total:.0f} img/s")

        norms = np.linalg.norm(self.embeddings, axis=1)
        print(f"  [EMBED] L2 Norm — Mean: {norms.mean():.4f}, Std: {norms.std():.6f}")

        print(f"\n  [CACHE] Saving embeddings to: {self.cache_embeddings_path}")
        np.save(self.cache_embeddings_path, self.embeddings)
        print(f"  [CACHE] Saving cleaned metadata to: {self.cache_metadata_path}")
        self.df.to_csv(self.cache_metadata_path, index=False)
        print(f"  [CACHE] Cache saved successfully.")

        return self.embeddings

    def get_complementary_recommendations(
        self,
        target_product_id: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        print("\n" + "=" * 70)
        print(f"[TASK 1] Complementary Recommendations for Product: {target_product_id}")
        print("=" * 70)

        target_mask = self.df["id"].astype(str) == str(target_product_id)
        if not target_mask.any():
            print(f"  [ERROR] Product ID '{target_product_id}' not found in catalog.")
            return pd.DataFrame()

        target_idx = target_mask.idxmax()
        target_row = self.df.iloc[target_idx]
        target_embedding = self.embeddings[target_idx].reshape(1, -1)

        target_subcategory = target_row.get("subCategory", "Unknown")
        target_name = target_row.get("productDisplayName", "N/A")
        target_category = target_row.get("masterCategory", "N/A")

        print(f"  [TARGET] Name: {target_name}")
        print(f"  [TARGET] Master Category: {target_category}")
        print(f"  [TARGET] Sub-Category: {target_subcategory}")
        print(f"  [TARGET] Embedding shape: {target_embedding.shape}")

        complement_mask = self.df["subCategory"] != target_subcategory
        complement_df = self.df[complement_mask].reset_index(drop=True)
        complement_embeddings = self.embeddings[complement_mask.values]

        print(f"  [FILTER] Excluded '{target_subcategory}' — {complement_df.shape[0]} candidates remain.")

        if len(complement_df) == 0:
            print("  [WARN] No complementary products found after filtering.")
            return pd.DataFrame()

        similarities = sklearn_cosine_similarity(
            target_embedding, complement_embeddings
        ).flatten()

        print(f"  [SIMILARITY] Computed {len(similarities)} pairwise scores.")
        print(f"  [SIMILARITY] Score range: [{similarities.min():.4f}, {similarities.max():.4f}]")

        top_indices = np.argsort(similarities)[::-1][:top_n]
        top_scores = similarities[top_indices]

        results = complement_df.iloc[top_indices].copy()
        results["similarity_score"] = top_scores
        results = results.reset_index(drop=True)

        print(f"\n  {'Rank':<6} {'Score':<8} {'SubCategory':<20} {'Product Name'}")
        print("  " + "-" * 80)
        for i, (_, row) in enumerate(results.iterrows()):
            print(
                f"  {i + 1:<6} {row['similarity_score']:<8.4f} "
                f"{str(row.get('subCategory', 'N/A')):<20} "
                f"{str(row.get('productDisplayName', 'N/A'))[:50]}"
            )

        return results

    def visualize_recommendations(
        self,
        target_product_id: str,
        recommendations_df: pd.DataFrame,
    ) -> None:
        if recommendations_df.empty:
            print("  [VIZ] No recommendations to visualize.")
            return

        target_mask = self.df["id"].astype(str) == str(target_product_id)
        target_row = self.df[target_mask].iloc[0]

        n_recs = len(recommendations_df)
        fig = plt.figure(figsize=(4 * (n_recs + 1), 5))
        gs = gridspec.GridSpec(1, n_recs + 1, wspace=0.3)

        ax0 = fig.add_subplot(gs[0, 0])
        try:
            img = Image.open(target_row["image_path"])
            ax0.imshow(img)
        except Exception:
            ax0.text(0.5, 0.5, "Image\nNot Found", ha="center", va="center", fontsize=10)
        ax0.set_title(
            f"TARGET\n{str(target_row.get('productDisplayName', 'N/A'))[:30]}\n"
            f"[{target_row.get('subCategory', 'N/A')}]",
            fontsize=8, fontweight="bold", color="darkblue",
        )
        ax0.axis("off")

        for i in range(n_recs):
            ax = fig.add_subplot(gs[0, i + 1])
            rec_row = recommendations_df.iloc[i]
            try:
                img = Image.open(rec_row["image_path"])
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Image\nNot Found", ha="center", va="center", fontsize=10)
            score = rec_row.get("similarity_score", 0.0)
            ax.set_title(
                f"#{i + 1} (Score: {score:.3f})\n"
                f"{str(rec_row.get('productDisplayName', 'N/A'))[:30]}\n"
                f"[{rec_row.get('subCategory', 'N/A')}]",
                fontsize=7, color="darkgreen",
            )
            ax.axis("off")

        fig.suptitle(
            "Task 1: Smart Complementary Product Recommendations",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.savefig("task1_recommendations.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("  [VIZ] Saved visualization to: task1_recommendations.png")

    def create_deduplicated_catalog(
        self,
        n_clusters: Optional[int] = None,
        variance_threshold: float = 0.95,
    ) -> pd.DataFrame:
        print("\n" + "=" * 70)
        print("[TASK 2] Unique Product Catalog Creation (Deduplication)")
        print("=" * 70)

        if n_clusters is None:
            n_clusters = min(self.num_catalog_clusters, len(self.df))
        n_clusters = min(n_clusters, len(self.df))
        if n_clusters < 2:
            print("  [ERROR] Not enough data points for clustering.")
            return pd.DataFrame()

        print(f"\n  --- Step A: PCA Dimensionality Reduction ---")
        print(f"  [PCA] Input embeddings shape: {self.embeddings.shape}")

        max_components = min(self.embeddings.shape[0], self.embeddings.shape[1])
        pca_full = PCA(n_components=max_components, random_state=42)
        pca_full.fit(self.embeddings)

        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components_95 = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
        n_components_95 = min(n_components_95, max_components)

        print(f"  [PCA] Components for {variance_threshold * 100:.0f}% variance: {n_components_95}")
        print(f"  [PCA] Variance explained by {n_components_95} components: "
              f"{cumulative_variance[n_components_95 - 1] * 100:.2f}%")
        print(f"  [PCA] Dimensionality reduction: 512 → {n_components_95}")

        pca = PCA(n_components=n_components_95, random_state=42)
        self.pca_embeddings = pca.fit_transform(self.embeddings)
        print(f"  [PCA] Reduced embeddings shape: {self.pca_embeddings.shape}")

        print(f"  [PCA] Top-5 component variance ratios: "
              f"{pca.explained_variance_ratio_[:5].round(4).tolist()}")

        print(f"\n  --- Step B: K-Means Clustering ---")
        print(f"  [KMEANS] Number of clusters: {n_clusters}")
        print(f"  [KMEANS] Input shape: {self.pca_embeddings.shape}")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
            verbose=0,
        )
        start_time = time.time()
        self.cluster_labels = kmeans.fit_predict(self.pca_embeddings)
        kmeans_time = time.time() - start_time

        print(f"  [KMEANS] Clustering complete in {kmeans_time:.1f}s")
        print(f"  [KMEANS] Cluster labels shape: {self.cluster_labels.shape}")
        print(f"  [KMEANS] Inertia: {kmeans.inertia_:.2f}")

        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        print(f"  [KMEANS] Cluster size stats — "
              f"Min: {counts.min()}, Max: {counts.max()}, "
              f"Mean: {counts.mean():.1f}, Median: {np.median(counts):.1f}")

        print(f"\n  --- Step C: Max-Min Diversity Selection ---")

        self.df["cluster_label"] = self.cluster_labels
        selected_indices = []
        cluster_centroids = kmeans.cluster_centers_

        for cluster_id in range(n_clusters):
            cluster_member_mask = self.cluster_labels == cluster_id
            cluster_member_indices = np.where(cluster_member_mask)[0]

            if len(cluster_member_indices) == 0:
                continue

            cluster_member_embeddings = self.pca_embeddings[cluster_member_indices]
            centroid = cluster_centroids[cluster_id].reshape(1, -1)

            distances = np.linalg.norm(cluster_member_embeddings - centroid, axis=1)
            best_local_idx = np.argmin(distances)
            best_global_idx = cluster_member_indices[best_local_idx]
            selected_indices.append(best_global_idx)

        self.deduplicated_catalog_df = self.df.iloc[selected_indices].reset_index(drop=True)

        print(f"  [DIVERSITY] Selected {len(selected_indices)} unique representatives.")
        print(f"  [DIVERSITY] Original catalog size: {len(self.df)}")
        print(f"  [DIVERSITY] Deduplicated catalog size: {len(self.deduplicated_catalog_df)}")
        print(f"  [DIVERSITY] Compression ratio: "
              f"{len(self.df) / len(self.deduplicated_catalog_df):.1f}x")

        if "subCategory" in self.deduplicated_catalog_df.columns:
            print(f"\n  [DIVERSITY] Category distribution in deduplicated catalog:")
            print(
                self.deduplicated_catalog_df["subCategory"]
                .value_counts()
                .head(15)
                .to_string(header=False)
            )

        return self.deduplicated_catalog_df

    def visualize_catalog_clusters(self, n_display: int = 10) -> None:
        if self.deduplicated_catalog_df is None or self.deduplicated_catalog_df.empty:
            print("  [VIZ] No deduplicated catalog to visualize.")
            return

        sample = self.deduplicated_catalog_df.head(n_display)
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for i, (_, row) in enumerate(sample.iterrows()):
            if i >= len(axes):
                break
            try:
                img = Image.open(row["image_path"])
                axes[i].imshow(img)
            except Exception:
                axes[i].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[i].set_title(
                f"Cluster {row.get('cluster_label', '?')}\n"
                f"{str(row.get('subCategory', ''))[:15]}\n"
                f"{str(row.get('productDisplayName', ''))[:25]}",
                fontsize=7,
            )
            axes[i].axis("off")

        last_used = i if len(sample) > 0 else -1
        for j in range(last_used + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"Task 2: Deduplicated Catalog Sample ({len(self.deduplicated_catalog_df)} items total)",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig("task2_deduplicated_catalog.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("  [VIZ] Saved visualization to: task2_deduplicated_catalog.png")

    def search_catalog_by_text(
        self,
        query_text: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        print("\n" + "=" * 70)
        print(f'[TASK 3] Reverse Product Search — Query: "{query_text}"')
        print("=" * 70)

        text_tokens = clip.tokenize([query_text], truncate=True).to(self.device)
        print(f"  [TEXT] Tokenized input shape: {text_tokens.shape}")

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=1)

        text_embedding = text_features.cpu().numpy().astype(np.float32)
        print(f"  [TEXT] Text embedding shape: {text_embedding.shape}")
        print(f"  [TEXT] Text embedding L2 norm: {np.linalg.norm(text_embedding):.4f}")

        similarities = sklearn_cosine_similarity(
            text_embedding, self.embeddings
        ).flatten()

        print(f"  [SEARCH] Computed similarities against {len(similarities)} products.")
        print(f"  [SEARCH] Score range: [{similarities.min():.4f}, {similarities.max():.4f}]")
        print(f"  [SEARCH] Mean score: {similarities.mean():.4f}")

        top_indices = np.argsort(similarities)[::-1][:top_n]
        top_scores = similarities[top_indices]

        results = self.df.iloc[top_indices].copy()
        results["search_score"] = top_scores
        results = results.reset_index(drop=True)

        print(f"\n  {'Rank':<6} {'Score':<8} {'SubCategory':<20} {'Product Name'}")
        print("  " + "-" * 80)
        for i, (_, row) in enumerate(results.iterrows()):
            print(
                f"  {i + 1:<6} {row['search_score']:<8.4f} "
                f"{str(row.get('subCategory', 'N/A')):<20} "
                f"{str(row.get('productDisplayName', 'N/A'))[:50]}"
            )

        return results

    def visualize_search_results(
        self,
        query_text: str,
        results_df: pd.DataFrame,
    ) -> None:
        if results_df.empty:
            print("  [VIZ] No search results to visualize.")
            return

        n_results = len(results_df)
        fig, axes = plt.subplots(1, n_results, figsize=(4 * n_results, 5))
        if n_results == 1:
            axes = [axes]

        for i, (_, row) in enumerate(results_df.iterrows()):
            try:
                img = Image.open(row["image_path"])
                axes[i].imshow(img)
            except Exception:
                axes[i].text(0.5, 0.5, "Image\nNot Found", ha="center", va="center")
            score = row.get("search_score", 0.0)
            axes[i].set_title(
                f"#{i + 1} (Score: {score:.3f})\n"
                f"{str(row.get('productDisplayName', 'N/A'))[:30]}\n"
                f"[{row.get('subCategory', 'N/A')}]",
                fontsize=8,
            )
            axes[i].axis("off")

        fig.suptitle(
            f'Task 3: Text Search Results for "{query_text}"',
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        safe_filename = "task3_search_" + "".join(c if c.isalnum() else "_" for c in query_text[:30])
        plt.savefig(f"{safe_filename}.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  [VIZ] Saved visualization to: {safe_filename}.png")

    def run_full_pipeline(self) -> None:
        pipeline_start = time.time()

        print("\n" + "#" * 80)
        print("#" + " " * 78 + "#")
        print("#   AI PRODUCT INTELLIGENCE SYSTEM — FULL PIPELINE EXECUTION" + " " * 18 + "#")
        print("#" + " " * 78 + "#")
        print("#" * 80)

        self.load_and_clean_data()
        self.load_clip_model()
        self.extract_embeddings()

        sample_product_id = str(self.df["id"].iloc[0])
        print(f"\n  [DEMO] Using sample product ID: {sample_product_id}")

        recommendations = self.get_complementary_recommendations(
            target_product_id=sample_product_id,
            top_n=5,
        )
        self.visualize_recommendations(sample_product_id, recommendations)

        deduplicated = self.create_deduplicated_catalog(
            n_clusters=min(self.num_catalog_clusters, len(self.df)),
            variance_threshold=0.95,
        )
        self.visualize_catalog_clusters(n_display=10)

        search_queries = [
            "red summer dress",
            "black leather boots",
            "blue denim jeans",
            "formal white shirt for men",
            "casual sports sneakers",
        ]

        for query in search_queries:
            results = self.search_catalog_by_text(query_text=query, top_n=5)
            self.visualize_search_results(query, results)

        total_time = time.time() - pipeline_start
        print("\n" + "#" * 80)
        print(" PIPELINE EXECUTION COMPLETE")
        print("#" * 80)
        print(f"  Total products processed: {len(self.df)}")
        print(f"  Embedding dimensions: {self.embeddings.shape[1]}")
        print(f"  Task 1 — Recommendations generated: {len(recommendations)} items")
        print(f"  Task 2 — Deduplicated catalog size: {len(deduplicated)} items")
        print(f"  Task 3 — Search queries executed: {len(search_queries)}")
        print(f"  Total pipeline time: {total_time:.1f}s")
        print("#" * 80)


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    UNIVERSAL_DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")

    CONFIG = {
        "dataset_base_path": UNIVERSAL_DATASET_PATH,
        "sample_size": 6000,
        "batch_size": 256,
        "cache_embeddings_path": "clip_embeddings.npy",
        "cache_metadata_path": "cleaned_metadata.csv",
        "num_catalog_clusters": 500,
    }

    print("=" * 80)
    print(" CONFIGURATION")
    print("=" * 80)
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 80)

    system = AIProductIntelligenceSystem(**CONFIG)
    system.run_full_pipeline()

    print("\n" + "=" * 80)
    print(" ADDITIONAL STANDALONE DEMONSTRATIONS")
    print("=" * 80)

    if len(system.df) > 10:
        random_product_id = str(system.df["id"].iloc[10])
        print(f"\n[EXTRA DEMO] Complementary recs for product: {random_product_id}")
        extra_recs = system.get_complementary_recommendations(random_product_id, top_n=3)
        if not extra_recs.empty:
            print(extra_recs[["id", "productDisplayName", "subCategory", "similarity_score"]].to_string())

    print("\n[EXTRA DEMO] Custom text search: 'elegant evening gown'")
    evening_results = system.search_catalog_by_text("elegant evening gown", top_n=3)
    if not evening_results.empty:
        print(evening_results[["id", "productDisplayName", "subCategory", "search_score"]].to_string())

    if system.deduplicated_catalog_df is not None:
        print("\n[EXTRA DEMO] Deduplicated Catalog Statistics:")
        print(f"  Total unique items: {len(system.deduplicated_catalog_df)}")
        if "masterCategory" in system.deduplicated_catalog_df.columns:
            print("  Master Category Distribution:")
            print(
                system.deduplicated_catalog_df["masterCategory"]
                .value_counts()
                .to_string(header=False)
            )

    print("\n" + "=" * 80)
    print(" ALL TASKS COMPLETED SUCCESSFULLY")
    print("=" * 80)
