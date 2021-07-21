# Anime_Recommender

This is a project for CS608.

Use user ratings, synopsis and posters from anime to form a user preference matrix for recommending anime.

[data_scraper_and_preprocessing.ipynb](data_scraper_and_preprocessing.ipynb) is to collect the data from https://myanimelist.net/ and preprocess the data for models.

[anime_pic_download.sh](anime_pic_download.sh) download the posters from the collected list.

[rating_only.ipynb](rating_only.ipynb) build recommenders with only the rating data.

[image_feature_extraction.py](image_feature_extraction.py) and [synopsis_feature_extraction.py](synopsis_feature_extraction.py) is to extract the visual and textual features for multimodality models.

[multimodality_vbpr.ipynb](multimodality_vbpr.ipynb) runs recommenders with visual and textual features.
