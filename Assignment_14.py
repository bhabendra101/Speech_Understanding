import gtts
import speech_recognition as sr
import librosa
import soundfile as sf
import os

def make_a_corpus(texts, languages, filenames):
    recognized_texts = []
    recognizer = sr.Recognizer()

    for text, lang, name in zip(texts, languages, filenames):
        mp3_file = name + ".mp3"
        wav_file = name + ".wav"

        # 1. Text → MP3
        tts = gtts.gTTS(text=text, lang=lang)
        tts.save(mp3_file)

        # 2. MP3 → WAV
        audio, sr_rate = librosa.load(mp3_file, sr=None)
        sf.write(wav_file, audio, sr_rate)

        # 3. Speech Recognition
        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)
            try:
                recognized = recognizer.recognize_google(audio_data, language=lang)
            except sr.UnknownValueError:
                recognized = ""

        recognized_texts.append(recognized)

    return recognized_texts