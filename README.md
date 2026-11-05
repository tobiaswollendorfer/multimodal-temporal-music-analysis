[README.md](https://github.com/user-attachments/files/34426766/README.md)
# Multimodal Temporal Dynamics in Music Genres

This project investigates how music genres have evolved over time using the **Music4All-Onion** dataset. It examines changes in musical characteristics, differences between genres, and whether these changes are consistent across audio, metadata, and lyrics.

## Dataset

The original Music4All-Onion dataset contains **109,269 songs** and **685 candidate genres**. For this study, songs were matched across three modalities and assigned to genres using Google OR-Tools, with each song assigned to at most one selected genre.

The final analysis dataset contains:

- **87,858 songs** assigned to **50 genres**
- Release years from **1955 to 2019**
- **61 overlapping five-year windows**, moving forward one year at a time
- A minimum of **20 songs** for a genre-window to be considered valid
- **Spotify metadata:** e.g., energy, danceability, tempo, valence, and duration
- **Essentia audio features:** e.g., loudness, spectral properties, and rhythmic characteristics
- **Lyrics TF-IDF:** word-based features describing lyrical content

## Research Questions

- **RQ1:** How do the characteristics of music genres evolve over time across different modalities?
- **RQ2:** How do interpretable musical features change within genres, both in absolute terms and relative to the overall music landscape?
- **RQ3:** Do music genres become more or less similar to one another over time?
- **RQ4:** To what extent are changes in genre similarity consistent across metadata, audio, and lyrics?

## Planned Analysis

- Track changes in features such as energy, loudness, tempo, danceability, and lyrical vocabulary.
- Compare genre trends with overall music trends to identify distinctive developments.
- Compare published genre conventions with the characteristics observed in the dataset.
- Measure how much individual genres change over time and which genre pairs converge or diverge.
- Examine whether acoustic and lyrical trends tell similar or different stories.
- Check whether findings are robust to genre assignments, temporal coverage, and window selection.

## Data Notes

Genre assignments are based on the dataset's genre scores and an optimization procedure, not on verified record-label classifications. Because five-year windows overlap, neighboring windows are not statistically independent. The dataset is less complete in earlier decades, so temporal comparisons must account for coverage.
