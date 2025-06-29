**SongFinder** is an ML-powered search tool that helps users discover songs based on textual descriptions or lyrics snippets. Using the **Genius Song Lyrics dataset**, the system leverages **semantic search** (via SBERT embeddings and FAISS) to retrieve the top 5 most relevant songs, even when queries don’t match lyrics exactly (e.g., "breakup song with guitar solo" → _November Rain_). Designed for low latency (<500ms), the tool prioritizes accuracy and speed. Ideal for music fans who recall lyrics or themes but not song titles.

---

### **1. Problem Formulation**

#### Clarifying Questions:
- _User Intent:_ Are queries descriptive ("song about lost love") or literal ("lyrics: 'hello darkness'")?
	Answer: Queries can be formulated using the combination of song description and the exact words.
- _Output Scope:_ Strictly top-5 matches, no ranking/explanation needed?
	Answer: The system returns a variable number of results (1–5) based on a similarity threshold. The output is ranked from the most relevant to the least. Relevant lyrics are highlighted.
- _Languages:_ Support multilingual queries (dataset includes language tags)?
	Answer: Queries are expected to be in one of the following languages: English or Russian. (No)

#### Business Goal:
- Enable users to find songs when they recall lyrics or general themes but not titles/artists.

#### Use Cases:
##### 1. Primary Use Case: "I Know the Lyrics, But Not the Title"
- **Scenario:**  
	- A user remembers a _fragment_ of song lyrics (e.g., "hello darkness my old friend") but not the title/artist.

- **How LyricFinder Helps:**
	- _Input:_ `"hello darkness my old friend"` → Output: **Top 5 matches** (e.g., _The Sound of Silence_ by Simon & Garfunkel).
	- _Value:_ Solves the universal "tip-of-the-tongue" problem for music.

##### 2. Use Case: Discovering Songs by Conceptual Themes

- **Scenario:**  
	- A user searches for songs about a _theme_ (e.g., "long-distance relationships") without exact lyrics.

- **How LyricFinder Helps:**
	- SBERT captures semantics:
	    - Query: `"sad song about missing someone far away"` → Matches lyrics like _"I'm here without you baby..."_ (even if keywords don’t overlap).

##### 3. Use Case: Music Trivia & Content Creation

- **Scenario:**  
	- A podcast host needs songs with _specific lyrical content_ (e.g., "songs mentioning 'California'").

- **How LyricFinder Helps:**
	- Acts as a **lyrical search engine** for researchers/creators.

##### Requirements:
- **Input:** Free-text query (e.g., "romantic 80s pop song").
- **Output:** Ranked list of matching songs.
- **Constraints:** Low latency (<500ms), scalable to thousands of songs.

#### ML Objective:
- **Category:** Semantic Search.
- **Retrieval Task:** Find songs with lyrics semantically closest to the query.
- **I/O:** Text → Song embeddings → Nearest-neighbor search.
- **Constraints:**
    - No metadata (mood/theme) → Pure text-based matching.
    - Low latency (real-time search).


---

### **2. Metrics**

#### Offline:
- **Recall@5:** % of true matches in top 5 results (need test queries with ground truth).
- **nDCG@5:** Ranking quality of top 5.
- **Query Latency:** <500ms.

#### Online:
- **Click-through Rate (CTR):** Do users click top results?
- **Mean Reciprocal Rank (MRR):** Quality of top result.

---

### **3. Architecture (MVP)**

#### Non-ML:
- **Frontend:** Query input, results display.
- **Backend:** FastAPI server.
- **Database:** PostgreSQL (song metadata: title, artist, language).

#### ML-Components
1. **Embedding Model:**
    - Convert **query** and **lyrics** to vectors (e.g., `all-MiniLM-L6-v2` for fast embeddings).
2. **Vector Database:**
    - Precompute lyrics embeddings → Store in FAISS/Pinecone.
3. **Search Service:**
    - Query → Embedding → FAISS nearest-neighbor lookup → Return top 5 songs.
4. **Ranker:**
	- Order generated candidates by relevance.

---

### **4. Data Preparation**

#### Lyrics Dataset:
- **Source**: Genious Lyrics
- **Data Types:** numerical (year, views), categorical (tag, artist, language), unstructured (tytle, lyrics)
- **Limitations:**
	- No mood/theme → Rely solely on lyrics text.
	- Possible noise (repetition, non-lyric text).
- **Storage:** PostgreSQL
- **Data preparation steps:**
	1. Filter non-English and non-Russian songs.
	2. Get top 50K songs based on their veiws.
	3. Clean lyrics (remove brackets, annotations, normalize case).
	4. Split into train/test.

#### Queries Dataset (for SBERT fine-tuning)
- **Source**: Manually generated or using weak supervision (Use artist/genre metadata to auto-generate query-lyric pairs (e.g., "rap song about love" for Drake lyrics))
- **Data Types:** Triplets
```
	{
	  "query": "sad breakup song with piano",  # User search query
	  "positive_lyrics": "...",  # Relevant lyrics (from target song)
	  "negative_lyrics": "..."  # Irrelevant lyrics (random or hard negative)
	}
```
- **Limitations:**
	- **Bias**: Manual queries may overrepresent popular songs.
	- **Coverage**: Limited to lyrics in the Genius dataset (no obscure/underground tracks).
- **Storage:** PostgreSQL
- **Data preparation steps:**
	1. Generate queries.
	2. Extract Positive Lyrics.
	3. Sample Negative Lyrics.
	4. Validation (Ensure 10% of triplets are verified by a second annotator. Remove queries matching both positive and negative lyrics)
	5. Split into train/test.

---

### **5. Feature Engineering**

#### Features:
- **Lyrics Embeddings:** 384-dim vectors from SBERT.
- **Artist names:** one-hot encoded.
- **Tag (genre):** one-hot encoded.

#### Representation:
- Precompute all lyrics embeddings offline → FAISS index.

---

### **6. Model Development**

#### Approaches:
1. **Baseline:** TF-IDF + Cosine Similarity (fast but no semantics).
2. **MVP:** Pretrained SBERT (semantic, no training needed).
3. **Advanced:** Fine-tune SBERT with triplet loss (query, positive lyrics, negative lyrics=random).

#### Training (if fine-tuning):
- **Loss:** Triplet loss (margin=1.0).
- **Optimizer:** AdamW (lr=2e-5).

#### Evaluation:
- Test set of 1K diverse queries with human-labeled matches.

---

### **7. Prediction Service**

#### Serving:
- **Online:** Query → Embedding → FAISS lookup → Return top-5 in <500ms.
- **Caching:** Memcached for frequent queries.

#### Edge Cases:
- No matches → Return "Try a different query" + popular songs.

---

### **8. Online Testing**

#### A/B Test:
- **Control:** TF-IDF.
- **Test:** SBERT.
- **Metric:** CTR@5.

#### Shadow Mode:
- Log SBERT results without exposing to users initially.

---

### **9. Scaling & Monitoring**

#### Scaling:
- **FAISS:** Shard index by artist/tag/language.
- **Load Balancing:** Distribute query load (NGINX + Kubernetes).

#### Monitoring:
- **Alerts:** Recall@5 drops <70% or latency >1s.
- **Drift Detection:** Track embedding distribution shifts (KL divergence).
