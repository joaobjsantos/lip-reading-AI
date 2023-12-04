from split_video_into_frames import get_word_counts, split_video_by_words, get_word_counts_from_file
from crop import crop_dataset
from train import split_and_train, split_only
from test import get_model_and_test
import pyphen, sys, os
import shutil

MIN_WORD_COUNT = 10

def get_frequent_words(word_counts, min_word_count, min_syllables = None, min_length = None):
    """
    Generate a list of frequent words based on their counts and optional filters.

    Parameters:
        word_counts (dict): A dictionary mapping words to their respective counts.
        min_word_count (int): The minimum count required for a word to be considered frequent.
        min_syllables (int, optional): The minimum number of syllables required for a word to be considered frequent. Defaults to None.
        min_length (int, optional): The minimum length required for a word to be considered frequent. Defaults to None.

    Returns:
        list: A list of frequent words that satisfy the given conditions.
    """
    frequent_words = []
    pt_dic = pyphen.Pyphen(lang="pt_PT")
    for word, count in word_counts.items():
        if count >= min_word_count and \
        (min_syllables is None or pt_dic.inserted(word).count("-") + 1 >= min_syllables) and \
        (min_length is None or len(word) >= min_length):
            frequent_words.append(word)
    return frequent_words


if __name__ == "__main__":

    if len(sys.argv) != 5:
        sys.exit("The format should be: video_path subtitles_path gpu{True/False} model{3_layer_CNN/4_layer_CNN/CNN_LSTM}\n")

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        sys.exit("Video path does not exist.\n")

    if not video_path.endswith(".mp4"):
        sys.exit("Video path must be a .mp4 file.\n")

    subtitles_path = sys.argv[2]

    if not os.path.exists(subtitles_path):
        sys.exit("Subtitles path does not exist.\n")

    if not subtitles_path.endswith(".srt"):
        sys.exit("Subtitles path must be a .srt file.\n")

    #video_path = "ric3.mp4"
    #subtitles_path = "ric3.srt"

    use_gpu = sys.argv[3]

    if use_gpu != "True" and use_gpu != "False":
        sys.exit("Use GPU must be True or False.\n")

    use_gpu = True if use_gpu == "True" else False

    model_name = sys.argv[4]

    if model_name != "3_layer_CNN" and model_name != "4_layer_CNN" and model_name != "CNN_LSTM":
        sys.exit("The model must be one of these three: 3_layer_CNN , 4_layer_CNN, CNN_LSTM")


    # delete output_frames folder, cropped folder and train_val_test.npy file
    if os.path.exists("output_frames"):
        shutil.rmtree("output_frames")

    if os.path.exists("cropped"):
        shutil.rmtree("cropped")

    if os.path.exists("train_val_test.npy"):
        os.remove("train_val_test.npy")


    word_counts, word_decoder = split_video_by_words(video_path, subtitles_path)
    print("Subtitles processed")

    word_counts = get_word_counts()
    # for w,c in word_counts.items():
    #     print(w, c)

    print({word: word_counts[word] for word in get_frequent_words(word_counts, min_word_count=MIN_WORD_COUNT, min_syllables=2)})

    # print(get_frequent_words(word_counts, min_word_count=20, min_syllables=2))
    # print(get_frequent_words(word_counts, min_word_count=5, min_syllables=2))
    # print(len(get_frequent_words(get_word_counts("output_frames_len_2"), min_word_count=5, min_syllables=2)))

    crop_dataset(allowed_words=get_frequent_words(word_counts, min_word_count=MIN_WORD_COUNT, min_syllables=2), max_word_instances=MIN_WORD_COUNT)
    # crop_dataset(dataset_folder="output_frames_len_s_2", allowed_words=get_frequent_words(word_counts, min_word_count=20, min_syllables=2))
    # crop_dataset(dataset_folder="output_frames_len_s_2", allowed_words=get_frequent_words(word_counts, 5, min_syllables=2))
    print("Dataset cropped")

    # split_only()
    # split_only("cropped_5_10_len_s")

    model = split_and_train(gpu=use_gpu, model_name=model_name, max_word_count=MIN_WORD_COUNT)
    print("Model trained")

    get_model_and_test(model)
    print("Model tested")

