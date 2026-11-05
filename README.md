# Multimodal Temporal Dynamics in Music Genres

This master's thesis investigates how musical characteristics and relationships between genres change over time, using the **Music4All-Onion** dataset. The analysis focuses on interpretable features, genre similarity, and comparisons across modalities.

## Dataset

The original Music4All-Onion dataset contains **109,269 songs** and **685 candidate genres**. Tracks available in all three selected modalities were assigned to at most one genre using Google OR-Tools.

The final selected dataset contains:

- **87,858 songs** assigned to **50 genres**
- Release years **1955–2019**
- **61 overlapping five-year windows**, shifted by one year
- At least **20 songs per genre-window** for a valid comparison
- **Spotify metadata:** e.g., energy, danceability, tempo, valence, duration
- **Essentia audio features:** e.g., loudness, spectral and rhythmic descriptors
- **Lyrics TF-IDF:** word-based lyrical features

## Research Questions

- **RQ1:** How do interpretable musical features change within genres, both in absolute terms and relative to the overall music landscape?
- **RQ2:** Do music genres become more or less similar to one another over time?
- **RQ3:** To what extent are changes in genre similarity consistent across Spotify metadata, Essentia audio features, and lyrics TF-IDF?

## Planned Analysis

- Track the development of individual features, such as energy, loudness, tempo, danceability, duration, and selected lyrical terms.
- Compare genre-specific trajectories with the overall music landscape in the same time windows.
- Where reliable sources exist, compare observed genre characteristics with published genre conventions.
- Measure changes in pairwise genre similarity to identify convergence and divergence.
- Compare genre-similarity trends across metadata, audio, and lyrics.
- Check robustness to window overlap, temporal coverage, and genre assignments.

## Data Notes

Genre labels come from the dataset's genre scores and an optimization procedure; they are not verified record-label classifications. Overlapping windows share tracks and are not independent. Early decades have lower genre coverage, so comparisons require appropriate support and careful interpretation.
