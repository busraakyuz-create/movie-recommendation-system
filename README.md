# Movie Recommendation System: Content-Based, Collaborative & Hybrid

A movie recommender built on the **MovieLens** dataset (9,724 rated movies, 610 users). For a given movie, it finds the most similar titles using three approaches and compares them.

## Approach
1. **Data cleaning:** merged movies, user tags and ratings, and kept only rated movies
2. **Content-based filtering:** combined each movie's genres and tags into one `metadata` text → `CountVectorizer` (~1,700 terms) → **TruncatedSVD** with 25 components (>80% of the variance explained)
3. **Collaborative filtering (item-to-item):** movie × user rating matrix (9,724 × 610) → TruncatedSVD with 200 components
4. **Cosine similarity** in each latent space, plus a **hybrid score** that averages the content and collaborative similarities

## Example: movies most similar to *Toy Story (1995)* (hybrid)

| Rank | Movie | Hybrid similarity |
|---|---|---|
| 1 | Toy Story 2 (1999) | 0.79 |
| 2 | Monsters, Inc. (2001) | 0.78 |
| 3 | Shrek (2001) | 0.75 |
| 4 | A Bug's Life (1998) | 0.71 |
| 5 | Finding Nemo (2003) | 0.71 |
| 6 | Antz (1998) | 0.70 |
| 7 | Toy Story 3 (2010) | 0.70 |

Each method has blind spots: content similarity alone returns animated films that share tags (e.g. *Turbo*), and ratings alone bring in popular but unrelated titles (e.g. *Independence Day*). The hybrid combines both signals and gives the most realistic recommendations.

## Tech
Python · scikit-learn (CountVectorizer, TruncatedSVD, cosine_similarity) · pandas · NumPy · Matplotlib

## Files
- `recommender-system.ipynb` – full pipeline
- `tests/` – automated checks
- `data/` – data is downloaded inside the notebook

Built during the Workintech Data Science & Machine Learning program.
