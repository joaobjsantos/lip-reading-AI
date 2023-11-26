from split_video_into_frames import split_video_by_words
from crop import crop_dataset
from train import split_and_train
from test import test_cnn_model


def get_frequent_words(word_counts, min_word_count):
    frequent_words = []
    for word, count in word_counts.items():
        if count >= min_word_count:
            frequent_words.append(word)
    return frequent_words


if __name__ == "__main__":
    video_path = ""
    subtitles_path = ""

    word_counts, word_decoder = split_video_by_words(video_path, subtitles_path)
    print("Subtitles processed")

    crop_dataset(allowed_words=get_frequent_words(word_counts, 10))
    print("Dataset cropped")

    model = split_and_train()
    print("Model trained")

    test_cnn_model(model)
    print("Model tested")

