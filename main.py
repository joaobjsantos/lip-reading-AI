from split_video_into_frames import split_video_by_words, get_word_counts_from_file
from crop import crop_dataset
from train import split_and_train
from test import get_model_and_test


def get_frequent_words(word_counts, min_word_count):
    frequent_words = []
    for word, count in word_counts.items():
        if count >= min_word_count:
            frequent_words.append(word)
    return frequent_words


if __name__ == "__main__":
    # video_path = "ric2.mp4"
    # subtitles_path = "ric2.srt"

    # word_counts, word_decoder = split_video_by_words(video_path, subtitles_path)
    # print("Subtitles processed")

    # word_counts = get_word_counts_from_file()
    # for w,c in word_counts.items():
    #     print(w, c)

    # crop_dataset(allowed_words=get_frequent_words(word_counts, 5))
    # print("Dataset cropped")

    model = split_and_train(gpu=True)
    print("Model trained")

    get_model_and_test(model)
    print("Model tested")

