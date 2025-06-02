DaTacos:

The DaTacos presents several challenges related to its dataset and compatibility with existing models, leading to the development of a separate, custom model.
Dataset Incompatibility and Limitations

    Incompatible with Existing Models: The DaTacos dataset isn't directly compatible with existing models within the project, such as Lyricover.
    Need for Physical Audio Files: For compatibility, a physical audio file is required. DaTacos, however, only provides extracted features in HDF5 format, not the raw audio.
    Lack of Song Information: The dataset doesn't provide identifying song information (e.g., "Song X by Artist Y," or a YouTube ID). This prevents models that rely on such metadata or need to process the original audio from working.
    Impact on Lyric-Based Models: Because of the lack of raw audio and song metadata, models like Lyricover, which are based on separate tonal and lyrical features (requiring audio for lyric transcription), cannot operate on this dataset.

Attempts to Overcome Limitations

    SecondHandSongs API: One potential workaround was to request access to the SecondHandSongs API (which is the source of the DaTacos data). This API does provide the necessary song information.
    Limited API Access: Unfortunately, access to the SecondHandSongs API is very rarely granted. In 2025, for example, it was granted only once.
    Decision to Create a Custom Model: Due to the difficulty in obtaining API access, we decided to create a separate, custom model specifically designed to utilize the DaTacos dataset for learning.

Feature Evolution and Challenges with the Custom Model

    Initial Feature Set (HPCP only): Initially, a simple subset of features, HPCP (Harmonic Pitch Class Profile) alone, was used.

    Unsatisfactory HPCP Results: The results were highly unsatisfactory. It was concluded that determining whether something is a cover or not based solely on HPCP is very difficult.

    Full Feature Set Implemented: Consequently, the full set of available features was incorporated:
        chroma_cens: numpy.ndarray
        crema: numpy.ndarray
        hpcp: numpy.ndarray
        key_extractor:
            key: numpy.str_
            scale: numpy.str_
            strength: numpy.float64
        madmom_features:
            tempos: numpy.ndarray
        mfcc_htk: numpy.ndarray

    Key and Scale Feature Enhancement: The key (e.g., C, D, E) and scale (major/minor) information were combined into a single feature. Previously, this was calculated imprecisely by simple binary comparison. Now, by merging them, the model gains more useful information as the actual "distance" or similarity between keys and scales is measured.

    Major Challenges with Array-Based Features:
        The biggest problems were encountered with chroma_cens, crema, hpcp, and mfcc_htk.
        Using simple statistical calculations like max, min, median, or mean on these large arrays (e.g., 12x2000) did not provide sufficient information.
        DWT (Discrete Wavelet Transform) Attempts: Attempts were made using DWT, which yielded the best results so far, but these were still not satisfactory enough.
        Similarity Matrix Computation: Efforts were also made using similarity matrix computation. While this appears to be the most reasonable approach, it's a very memory-intensive process (both RAM and disk space). This process needs significant optimization to run much faster and require fewer resources.

Current Evaluation Status

    Unsatisfactory Evaluation Results: At this moment, the evaluation results are unsatisfactory. The model performs only slightly better than random guessing.
    Bias Towards "Cover": The model currently has a greater tendency to classify songs as covers (assigning a '1' label more often than '0').
