import argparse
import ast
import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from tqdm import tqdm
from utils import load_config, parse_concept_list

import numpy as np
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


EXTRA_STOPWORDS: Set[str] = {
    "using", "based", "use", "used", "approach", "novel", "new",
    "improved", "propose", "proposed", "technique", "techniques",
    "method", "methods", "study", "analysis", "paper", "work",
    "result", "results", "show", "demonstrate", "present",
    "investigate", "explore", "evaluate", "experiment",
    "experimental", "performance", "effective", "efficient",
    "also", "various", "different", "several", "many",
}

PROTECTED_WORDS: Set[str] = {
    "deep", "machine", "neural", "network", "networks", "learning",
    "natural", "language", "processing", "computer", "vision",
    "reinforcement", "transfer", "federated", "generative",
    "adversarial", "attention", "self", "supervised", "unsupervised",
    "semi", "few", "shot", "zero", "pre", "fine", "training",
    "inference", "classification", "detection", "segmentation",
    "recognition", "generation", "extraction", "embedding",
    "representation", "knowledge", "graph", "ontology", "model",
    "optimization", "gradient", "loss", "function", "distribution",
    "latent", "feature", "layer", "encoder", "decoder", "diffusion",
    "contrastive", "metric", "reward", "policy", "agent", "state",
    "action", "environment", "image", "text", "speech", "audio",
    "video", "point", "cloud", "sequence", "token", "prompt",
    "retrieval", "memory", "reasoning", "alignment", "robustness",
    "fairness", "privacy", "safety", "bias", "calibration",
}

DEFAULT_ACRONYMS: Dict[str, str] = {
    "nlp": "natural language processing",
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "lstm": "long short-term memory",
    "gru": "gated recurrent unit",
    "gpt": "generative pre-trained transformer",
    "bert": "bidirectional encoder representations from transformers",
    "svm": "support vector machine",
    "knn": "k nearest neighbors",
    "pca": "principal component analysis",
    "gan": "generative adversarial network",
    "vae": "variational autoencoder",
    "rl": "reinforcement learning",
    "rlhf": "reinforcement learning from human feedback",
    "dpo": "direct preference optimization",
    "ner": "named entity recognition",
    "pos": "part of speech",
    "iot": "internet of things",
    "llm": "large language model",
    "rag": "retrieval augmented generation",
    "tf-idf": "term frequency inverse document frequency",
    "lda": "latent dirichlet allocation",
    "hmm": "hidden markov model",
    "sgd": "stochastic gradient descent",
    "auc": "area under the curve",
    "rmse": "root mean square error",
    "mri": "magnetic resonance imaging",
    "fmri": "functional magnetic resonance imaging",
    "eeg": "electroencephalography",
    "mae": "mean absolute error",
    "mse": "mean squared error",
    "elbo": "evidence lower bound",
    "kl": "kullback leibler",
    "mcmc": "markov chain monte carlo",
    "mlp": "multilayer perceptron",
    "vit": "vision transformer",
}


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[""„‟]", '"', text)
    text = re.sub(r"[''‛]", "'", text)
    text = re.sub(r"[–—―]", "-", text)
    text = re.sub(r"[^\w\s\-/().+#&]", "", text)
    text = re.sub(r"^\s*[-•·]\s*", "", text)
    text = re.sub(r"\s*\(\s*\)\s*", "", text)
    return text.strip(" -.,;:")


def expand_acronym(text: str) -> str:
    """Expand known acronyms; strip parenthetical acronyms."""
    m = re.match(r"^(.+?)\s*\(([a-zA-Z\-]+)\)\s*$", text)
    if m:
        text = m.group(1).strip()
    key = text.lower().replace(" ", "").replace("-", "")
    return DEFAULT_ACRONYMS.get(key, text)


class Lemmatizer:
    """spaCy-based lemmatizer (loaded once, reused)."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name, disable=["parser", "ner"])
        except OSError:
            logger.info(f"Downloading spaCy model '{model_name}'...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name, disable=["parser", "ner"])

    def lemmatize(self, text: str) -> str:
        if not text or len(text) < 2:
            return text
        doc = self.nlp(text.lower())
        tokens = []
        for tok in doc:
            if tok.is_punct or tok.is_space:
                continue
            lemma = tok.lemma_.lower()
            is_stop = tok.is_stop or lemma in EXTRA_STOPWORDS
            is_protected = tok.text in PROTECTED_WORDS or lemma in PROTECTED_WORDS
            if is_stop and not is_protected and len(doc) > 1:
                continue
            tokens.append(tok.text if "-" in tok.text else lemma)
        result = " ".join(tokens)
        result = re.sub(r"^(the|a|an|of|for|in|on|to|with|and|or)\s+", "", result)
        return result.strip()


def preprocess_concept(concept: str, lemmatizer: Lemmatizer) -> str:
    """Full preprocessing: clean -> lemmatize -> expand acronyms."""
    c = clean_text(concept)
    c = lemmatizer.lemmatize(c)
    c = expand_acronym(c)
    return c.strip()


class NormalizationState:
    """
    Persisted state that enables incremental normalization.

    Stores:
        - concept_to_canonical: {preprocessed_concept: canonical_form}
        - canonical_embeddings: {canonical_form: np.ndarray}
        - cluster_members:      {canonical_form: set of preprocessed members}
        - next_cluster_id:      int
        - canonical_to_cid:     {canonical_form: cluster_id}
    """

    def __init__(self):
        self.concept_to_canonical: Dict[str, str] = {}
        self.canonical_embeddings: Dict[str, np.ndarray] = {}
        self.cluster_members: Dict[str, set] = defaultdict(set)
        self.canonical_to_cid: Dict[str, int] = {}
        self.next_cluster_id: int = 0

    def is_known(self, concept: str) -> bool:
        return concept in self.concept_to_canonical

    def get_canonical(self, concept: str) -> Optional[str]:
        return self.concept_to_canonical.get(concept)

    def add_to_cluster(self, concept: str, canonical: str, embedding: np.ndarray):
        """Add a concept to an existing or new cluster."""
        self.concept_to_canonical[concept] = canonical
        self.cluster_members[canonical].add(concept)
        # Update embedding only if this is a new canonical
        if canonical not in self.canonical_embeddings:
            self.canonical_embeddings[canonical] = embedding
            self.canonical_to_cid[canonical] = self.next_cluster_id
            self.next_cluster_id += 1

    def get_cluster_id(self, canonical: str) -> int:
        return self.canonical_to_cid.get(canonical, -1)

    @property
    def n_clusters(self) -> int:
        return len(self.canonical_embeddings)

    @property
    def n_concepts(self) -> int:
        return len(self.concept_to_canonical)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"State saved: {self.n_concepts} concepts, "
                     f"{self.n_clusters} clusters -> {path}")

    @classmethod
    def load(cls, path: str) -> "NormalizationState":
        with open(path, "rb") as f:
            state = pickle.load(f)
        logger.info(f"State loaded: {state.n_concepts} concepts, "
                     f"{state.n_clusters} clusters <- {path}")
        return state

    @classmethod
    def load_or_create(cls, path: Optional[str]) -> "NormalizationState":
        if path and Path(path).exists():
            return cls.load(path)
        logger.info("No existing state found — starting fresh")
        return cls()


def select_canonical(concepts: List[str]) -> str:
    """Pick the best representative from a group of similar concepts."""
    if len(concepts) == 1:
        return concepts[0]

    def score(c: str) -> tuple:
        nw = len(c.split())
        nc = len(c)
        return (
            -abs(nw - 3),                                   # prefer 2-4 words
            -abs(nc - 25),                                   # prefer moderate length
            -10 if c.isupper() else 0,                       # penalize ALL CAPS
            2 if c.islower() else 0,                         # prefer lowercase
            -5 * len(re.findall(r"[^a-zA-Z0-9\s\-]", c)),  # penalize special chars
        )

    return max(concepts, key=score)


class ConceptNormalizer:
    """
    Incremental concept normalizer.

    - On first run: preprocesses, embeds, clusters everything, saves state.
    - On later runs: loads state, only processes NEW concepts, assigns them
      to the nearest existing cluster (if above threshold) or creates new ones.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        cols = cfg.get("columns", {})
        self.id_col = cols.get("id", "id")
        self.raw_col = cols.get("raw_concepts", "raw_concepts")
        self.threshold = cfg.get("similarity_threshold", 0.85)
        self.return_details = cfg.get("return_details", False)

        self.lemmatizer = Lemmatizer(cfg.get("spacy_model", "en_core_web_sm"))

        model_name = cfg.get("embedding_model", "all-MiniLM-L6-v2")
        logger.info(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.batch_size = cfg.get("batch_size", 256)

        state_path = cfg.get("state_file")
        self.state = NormalizationState.load_or_create(state_path)

    def _encode(self, concepts: List[str]) -> np.ndarray:
        return self.encoder.encode(
            concepts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def _assign_new_concepts(
        self, new_concepts: List[str], new_embeddings: np.ndarray
    ):
        """
        For each new concept, find the closest existing cluster. If similarity >= threshold, assign to that cluster. Otherwise, create a new cluster.
        After assigning all individually, do a second pass to merge new singletons among themselves.
        """
        state = self.state

        if state.n_clusters == 0:
            # Cold start: cluster all new concepts from scratch
            self._initial_cluster(new_concepts, new_embeddings)
            return

        # Get existing canonical embeddings as matrix
        existing_canonicals = list(state.canonical_embeddings.keys())
        existing_matrix = np.stack(
            [state.canonical_embeddings[c] for c in existing_canonicals]
        )

        # Compute similarity of each new concept to all existing canonicals
        sim_matrix = cosine_similarity(new_embeddings, existing_matrix)

        unassigned_concepts = []
        unassigned_embeddings = []

        for i, concept in tqdm(enumerate(new_concepts), total=len(new_concepts)):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_sim = sim_matrix[i, best_idx]

            if best_sim >= self.threshold:
                canonical = existing_canonicals[best_idx]
                state.add_to_cluster(concept, canonical, new_embeddings[i])
            else:
                unassigned_concepts.append(concept)
                unassigned_embeddings.append(new_embeddings[i])

        logger.info(
            f"Assigned {len(new_concepts) - len(unassigned_concepts)} "
            f"to existing clusters, {len(unassigned_concepts)} unassigned"
        )

        # Cluster unassigned among themselves
        if unassigned_concepts:
            self._initial_cluster(
                unassigned_concepts, np.stack(unassigned_embeddings)
            )

    def _initial_cluster(
        self, concepts: List[str], embeddings: np.ndarray
    ):
        """Cluster concepts from scratch using greedy nearest-neighbor."""
        state = self.state
        # Simple greedy clustering: iterate, assign to nearest existing or create new
        for i, concept in tqdm(enumerate(concepts), total=len(concepts)):
            emb = embeddings[i]

            if state.n_clusters > 0:
                canonicals = list(state.canonical_embeddings.keys())
                canon_matrix = np.stack(
                    [state.canonical_embeddings[c] for c in canonicals]
                )
                sims = cosine_similarity(emb.reshape(1, -1), canon_matrix)[0]
                best_idx = int(np.argmax(sims))
                if sims[best_idx] >= self.threshold:
                    state.add_to_cluster(concept, canonicals[best_idx], emb)
                    continue

            # New cluster — concept is its own canonical
            state.add_to_cluster(concept, concept, emb)

    def _merge_singular_plural(self):
        """Post-hoc: if 'X' and 'Xs' are both canonicals, merge them."""
        state = self.state
        canonicals = list(state.canonical_embeddings.keys())
        canonical_set = set(canonicals)
        merges: Dict[str, str] = {}

        for c in canonicals:
            if c.endswith("s") and c[:-1] in canonical_set:
                merges[c] = c[:-1]  # plural -> singular

        for old_canon, new_canon in merges.items():
            if old_canon == new_canon:
                continue
            # Move all members
            members = state.cluster_members.pop(old_canon, set())
            state.cluster_members[new_canon].update(members)
            for m in members:
                state.concept_to_canonical[m] = new_canon
            # Reassign cluster id
            old_cid = state.canonical_to_cid.pop(old_canon, None)
            # Remove old embedding
            state.canonical_embeddings.pop(old_canon, None)

        if merges:
            logger.info(f"Merged {len(merges)} singular/plural pairs")

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize concepts in the DataFrame.
        Processes only concepts not already in state.
        """
        df = df.copy()

        # ── Parse concept lists ────────────────────────────
        logger.info("Parsing concept lists...")

        concept_lists = df[self.raw_col].map(parse_concept_list)

        # Collect all unique raw concepts
        all_raw: Set[str] = set()
        for lst in concept_lists:
            all_raw.update(lst)
        all_raw.discard("")
        logger.info(f"Total unique raw concepts: {len(all_raw)}")

        # ── Preprocess ─────────────────────────────────────
        logger.info("Preprocessing concepts...")
        raw_to_preprocessed: Dict[str, str] = {}
        for raw in all_raw:
            raw_to_preprocessed[raw] = preprocess_concept(raw, self.lemmatizer)

        # Filter out empty after preprocessing
        raw_to_preprocessed = {
            r: p for r, p in raw_to_preprocessed.items() if len(p) >= 2
        }
        unique_preprocessed = set(raw_to_preprocessed.values())
        logger.info(f"Unique preprocessed concepts: {len(unique_preprocessed)}")

        # ── Find new concepts ──────────────────────────────
        new_concepts = [c for c in unique_preprocessed if not self.state.is_known(c)]
        known_count = len(unique_preprocessed) - len(new_concepts)
        logger.info(f"Already known: {known_count}, new: {len(new_concepts)}")

        # ── Encode & cluster new concepts ──────────────────
        if new_concepts:
            logger.info(f"Encoding {len(new_concepts)} new concepts...")
            new_embeddings = self._encode(new_concepts)
            logger.info("Clustering new concepts...")
            self._assign_new_concepts(new_concepts, new_embeddings)
            self._merge_singular_plural()

        # ── Build output ───────────────────────────────────
        logger.info("Building output...")

        normalized_lists = []
        cluster_id_lists = []
        preprocessed_lists = []

        for concept_list in concept_lists:
            norm_row = []
            cid_row = []
            prep_row = []
            for raw in concept_list:
                pp = raw_to_preprocessed.get(raw)
                if pp is None or len(pp) < 2:
                    norm_row.append(None)
                    cid_row.append(-1)
                    prep_row.append(None)
                    continue
                canonical = self.state.get_canonical(pp)
                cid = self.state.get_cluster_id(canonical) if canonical else -1
                norm_row.append(canonical)
                cid_row.append(cid)
                prep_row.append(pp)

            normalized_lists.append(norm_row)
            cluster_id_lists.append(cid_row)
            preprocessed_lists.append(prep_row)

        df["normalized_concepts"] = normalized_lists
        df["cluster_ids"] = cluster_id_lists
        if self.return_details:
            df["preprocessed_concepts"] = preprocessed_lists

        # Stats
        all_canonical = set()
        for lst in normalized_lists:
            all_canonical.update(c for c in lst if c is not None)
        logger.info(
            f"Done: {len(unique_preprocessed)} preprocessed -> "
            f"{len(all_canonical)} canonical forms "
            f"({self.state.n_clusters} total clusters in state)"
        )

        return df

    def build_cluster_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build cluster summary from current state."""
        rows = []
        for canonical, members in self.state.cluster_members.items():
            cid = self.state.get_cluster_id(canonical)
            for member in sorted(members):
                rows.append({
                    "cluster_id": cid,
                    "canonical_concept": canonical,
                    "member_concept": member,
                })
        return (
            pd.DataFrame(rows)
            .sort_values(["cluster_id", "canonical_concept", "member_concept"])
            .reset_index(drop=True)
        )

    def save_state(self):
        path = self.cfg.get("state_file")
        if path:
            self.state.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize extracted scientific concepts (incremental)"
    )
    parser.add_argument(
        "--config",
        default="configs/concept_normalization.yaml",
        help="Path to normalization YAML config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load data
    input_file = cfg["input_file"]
    logger.info(f"Reading {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} rows")

    # Normalize
    normalizer = ConceptNormalizer(cfg)
    df_out = normalizer.normalize(df)

    # Save outputs
    output_file = cfg["output_file"]
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_file, index=False)
    logger.info(f"Saved -> {output_file}")

    clusters_file = cfg.get("clusters_output_file")
    if clusters_file:
        cluster_df = normalizer.build_cluster_table(df_out)
        Path(clusters_file).parent.mkdir(parents=True, exist_ok=True)
        cluster_df.to_csv(clusters_file, index=False)
        logger.info(f"Clusters -> {clusters_file} ({len(cluster_df)} entries)")

    # Save state for future incremental runs
    normalizer.save_state()
