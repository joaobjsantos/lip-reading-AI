from split_video_into_frames import get_word_counts, split_video_by_words, get_word_counts_from_file
from crop import crop_dataset
from train import split_and_train, split_only
from test import get_model_and_test
import pyphen


def get_frequent_words(word_counts, min_word_count, min_syllables = None, min_length = None):
    frequent_words = []
    pt_dic = pyphen.Pyphen(lang="pt_PT")
    for word, count in word_counts.items():
        if count >= min_word_count and \
        (min_syllables is None or pt_dic.inserted(word).count("-") + 1 >= min_syllables) and \
        (min_length is None or len(word) >= min_length):
            frequent_words.append(word)
    return frequent_words


if __name__ == "__main__":
    video_path = "ric3.mp4"
    subtitles_path = "ric3.srt"

    word_counts, word_decoder = split_video_by_words(video_path, subtitles_path, output_dir="output_frames_syl_2_3")
    print("Subtitles processed")

    word_counts = get_word_counts("output_frames_len_s_2")
    # for w,c in word_counts.items():
    #     print(w, c)

    print({word: word_counts[word] for word in get_frequent_words(word_counts, min_word_count=40, min_syllables=2)})

    # print(get_frequent_words(word_counts, min_word_count=20, min_syllables=2))
    # print(get_frequent_words(word_counts, min_word_count=5, min_syllables=2))
    # print(len(get_frequent_words(get_word_counts("output_frames_len_2"), min_word_count=5, min_syllables=2)))

    crop_dataset(dataset_folder="output_frames_syl_2_3", allowed_words=get_frequent_words(word_counts, min_word_count=40, min_syllables=2))
    # crop_dataset(dataset_folder="output_frames_len_s_2", allowed_words=get_frequent_words(word_counts, min_word_count=20, min_syllables=2))
    # crop_dataset(dataset_folder="output_frames_len_s_2", allowed_words=get_frequent_words(word_counts, 5, min_syllables=2))
    print("Dataset cropped")

    # split_only()
    # split_only("cropped_5_10_len_s")

    model = split_and_train(cropped_dir="cropped_40_syllables", gpu=True)
    print("Model trained")

    get_model_and_test(model)
    print("Model tested")

