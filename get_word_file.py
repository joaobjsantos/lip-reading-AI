import os
import re
import pyphen

def parse_srt(file_path):
    #print(os.listdir("./"))
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    subtitles = []
    current_subtitle = None

    for line in lines:
        line = line.strip()
        if re.match(r'\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+', line):
            if current_subtitle:
                if 'text' in current_subtitle and current_subtitle['text'].strip():
                    subtitles.append(current_subtitle)
            current_subtitle = {'text': '', 'start_time': None, 'end_time': None}
            start, end = map(lambda x: re.sub(',', '.', x), line.split(' --> '))
            current_subtitle['start_time'] = timestamp_to_seconds(start)
            current_subtitle['end_time'] = timestamp_to_seconds(end)
        elif line == '' or re.match(r'\d+', line) or re.match(r'\[.*\]', line):
            continue
        elif current_subtitle:
            current_subtitle['text'] += line.replace(":"," ")

    if current_subtitle and 'text' in current_subtitle and current_subtitle['text'].strip():
        subtitles.append(current_subtitle)

    return subtitles

def timestamp_to_seconds(timestamp):
    h, m, s = map(float, re.split('[:,]', timestamp))
    return 3600 * h + 60 * m + s

def split_words_spaces_len(subtitles, fps, write_path=None):
    processed_subtitles = []

    for subtitle in subtitles:
        words = subtitle['text'].split()
        total_words = len(words)
        if total_words > 0:
            total_duration = subtitle['end_time'] - subtitle['start_time']
            total_word_duration = sum(len(word) / len(subtitle['text']) * total_duration for word in words)

            silence_duration = total_duration - total_word_duration
            silence_interval = silence_duration / (total_words - 1) if total_words > 1 else 0

            current_time = subtitle['start_time']
            offset = 0.1
            for word in words:
                word = word.lower()
                word_duration = len(word) / len(subtitle['text']) * total_duration
                word_end_time = current_time + word_duration

                start_time = round((current_time + offset) * fps)
                end_time = round((word_end_time) * fps)
                processed_subtitles.append(f"{word} {start_time} {end_time}")
                # write_to_file(word, round((current_time + offset) * 30), round((word_end_time + offset) * 30))
                current_time = word_end_time + silence_interval

    if write_path != None:
        with open(write_path, "w", encoding="utf-8") as output_file:
            output_file.writelines(processed_subtitle + '\n' for processed_subtitle in processed_subtitles)
    return processed_subtitles


def split_words_len(subtitles, fps, write_path=None):
    processed_subtitles = []

    for subtitle in subtitles:
        words = subtitle['text'].split()
        total_words = len(words)
        if total_words > 0:
            total_duration = subtitle['end_time'] - subtitle['start_time']
            total_word_duration = sum(len(word) / len(subtitle['text']) * total_duration for word in words)

            current_time = subtitle['start_time']

            for word in words:
                word = word.lower()
                word_duration = len(word) / len(subtitle['text']) * total_word_duration
                word_end_time = current_time + word_duration

                start_time = round(current_time * fps + 1)
                end_time = round(word_end_time * fps)
                processed_subtitles.append(f"{word} {start_time} {end_time}")
                # write_to_file(word, round((current_time + offset) * 30), round((word_end_time + offset) * 30))
                current_time = word_end_time

    if write_path != None:
        with open(write_path, "w", encoding="utf-8") as output_file:
            output_file.writelines(processed_subtitle + '\n' for processed_subtitle in processed_subtitles)
    return processed_subtitles


def split_words_syllables(subtitles, fps=30.0, write_path=None):
    processed_subtitles = []
    dic = pyphen.Pyphen(lang="pt_PT")

    for subtitle in subtitles:
        words = subtitle['text'].split()
        word_syllables = {word.lower(): dic.inserted(word.lower()).split("-") for word in words}
        total_syllables = sum(len(syllables) for syllables in word_syllables.values())
        if total_syllables > 0:
            total_duration = subtitle['end_time'] - subtitle['start_time']
            # total_word_duration = sum(len(word_syllables[word]) / len(subtitle['text']) * total_duration for word in words)

            current_time = subtitle['start_time']

            for word in words:
                word = word.lower()
                word_duration = len(word_syllables[word]) / total_syllables * total_duration
                word_end_time = current_time + word_duration

                start_time = round(current_time * fps + 1)
                end_time = round(word_end_time * fps)
                processed_subtitles.append(f"{word} {start_time} {end_time}")
                # write_to_file(word, round((current_time + offset) * 30), round((word_end_time + offset) * 30))
                current_time = word_end_time

    if write_path != None:
        with open(write_path, "w", encoding="utf-8") as output_file:
            output_file.writelines(processed_subtitle + '\n' for processed_subtitle in processed_subtitles)
    return processed_subtitles

def time_subtitles(substitles_path, fps=30.0):
    # return split_words_syllables(parse_srt(substitles_path), fps, write_path="processed_subtitles.txt")
    return split_words_spaces_len(parse_srt(substitles_path), fps, write_path="processed_subtitles.txt")

if __name__ == "__main__":
    srt_file_path = "teste.srt"
    time_subtitles(srt_file_path, 30)
