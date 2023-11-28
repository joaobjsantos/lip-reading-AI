import cv2
import os
import unidecode
from get_word_file import time_subtitles

def split_video_by_words_file(video_path, words_file):
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    with open(words_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            word, start_frame, end_frame = parts
            start_frame, end_frame = int(float(start_frame)), int(float(end_frame))

            word_count = 1
            while os.path.exists(os.path.join(output_dir, f"{word}", f"{video_id}_{word_count}")):
                word_count += 1

            word_output_dir = os.path.join(output_dir, f"{word}", f"{video_id}_{word_count}")
            os.makedirs(word_output_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            # fps = cap.get(cv2.CAP_PROP_FPS)

            frame_number = start_frame
            while frame_number <= end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    break

                frame_path = os.path.join(word_output_dir, f"frame_{frame_number}.jpg")
                cv2.imwrite(frame_path, frame)

                frame_number += 1

            cap.release()


def split_video_by_words(video_path, subtitles_path, allowed_words = None):
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    lines = time_subtitles(subtitles_path, fps=fps)

    print(fps)

    word_counts = {}

    word_decoder = {}
    inverse_word_decoder = {}
    decoded_word_counts = {}

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            word, start_frame, end_frame = parts
            start_frame, end_frame = int(float(start_frame)), int(float(end_frame))

            if allowed_words and word not in allowed_words:
                continue

            if not word.isascii():
                if word in inverse_word_decoder.keys():
                    word = inverse_word_decoder[word]
                else:
                    decoded_word = unidecode.unidecode(word)
                    original_word = word

                    decoded_word_counts[decoded_word] = decoded_word_counts.get(decoded_word, 0) + 1

                    word = f"{decoded_word}_{decoded_word_counts[decoded_word]}"
                    word_decoder[word] = original_word
                    inverse_word_decoder[original_word] = word


            word_count = 1
            while os.path.exists(os.path.join(output_dir, f"{word}", f"{video_id}_{word_count}")):
                word_count += 1

            word_counts[word] = word_count

            word_output_dir = os.path.join(output_dir, f"{word}", f"{video_id}_{word_count}")
            os.makedirs(word_output_dir, exist_ok=True)  

            frame_number = start_frame
            while frame_number <= end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    print("not ret " + word)
                    break

                frame_path = os.path.join(word_output_dir, f"frame_{frame_number}.jpg")
                if not cv2.imwrite(frame_path, frame):
                    print("not saved " + word)

                frame_number += 1
                
            print(frame_number/cap.get(cv2.CAP_PROP_FRAME_COUNT)*100, "%")
    cap.release()

    with open("word_counts.txt", "w", encoding='utf-8') as f:
        for word, count in word_counts.items():
            f.write(f"{word} {count}\n")

    with open("ascii_decoder.txt", "w", encoding='utf-8') as f:
        for word, original_word in word_decoder.items():
            f.write(f"{word} {original_word}\n")

    return word_counts, word_decoder


def get_word_counts_from_file(words_file="word_counts.txt"):
    word_counts = {}
    with open(words_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                word, count = parts
                word_counts[word] = int(count)
    return word_counts


if __name__ == "__main__":
    video_path = "ric1.mp4"
    subtitle_path = "ric1.srt"
    
    split_video_by_words(video_path, subtitle_path)
