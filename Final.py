import pronouncing
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}")

# Load the Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.to(device)

# Open a file dialog for the user to select a file
def select_audio_file():
    Tk().withdraw()  # Close the root window
    file_path = askopenfilename(filetypes=[("Audio Files", "*.wav *.flac")])
    return file_path

audio_file = select_audio_file()
if not audio_file:
    print("No file selected. Exiting...")
    exit()

# Load and preprocess the audio file
waveform, sample_rate = torchaudio.load(audio_file)

# Resample if necessary (Wav2Vec2 requires 16kHz)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    sample_rate = 16000

# Process the audio file
input_values = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt").input_values

# Perform speech-to-text using the model
with torch.no_grad():
    logits = model(input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)

# Decode the predicted ids to text
transcription = processor.batch_decode(predicted_ids)[0]
print("Transcription:", transcription)

# Define diphthongs with ARPAbet phoneme sequences
diphthong_phonemes = {
    'ɑι': ['AY'],         # Example: "high"
    'ɑʊ': ['AW'],         # Example: "cow"
    'ɔι': ['OY'],         # Example: "boy"
    'eι': ['EY'],         # Example: "say"
    'əʊ': ['OW'],         # Example: "go"
    'eə': ['EH', 'R'],    # For "lair" or "bear"
    'ιə': ['IH', 'R'],    # Approximation for ιə (e.g., "idea", "jeer")
    'ʊə': ['UH', 'R'],    # Approximation for ʊə (e.g., "sure", "cure")
}

# Define custom function to check for vowels
def is_vowel(phoneme):
    return any(char.isdigit() for char in phoneme)

# Function to strip stress markers (e.g., AY1 -> AY)
def strip_stress(phoneme):
    return ''.join([char for char in phoneme if not char.isdigit()])

# Function to detect syllables and diphthongs in a word
def detect_syllables_and_diphthongs(word):
    pronunciations = pronouncing.phones_for_word(word)
    if not pronunciations:
        return {"Word": word, "Syllables": [], "Diphthongs": []}

    detected_syllables = []
    detected_diphthongs = []

    for pronunciation in pronunciations:
        phonemes = pronunciation.split()
        syllables = [phoneme for phoneme in phonemes if is_vowel(phoneme)]
        detected_syllables.extend(syllables)

        phonemes_no_stress = [strip_stress(phoneme) for phoneme in phonemes]
        syllables_no_stress = [strip_stress(syllable) for syllable in syllables]

        for diphthong, phoneme_sequence in diphthong_phonemes.items():
            if ' '.join(phoneme_sequence) in ' '.join(phonemes_no_stress):
                detected_diphthongs.append(diphthong)

    detected_diphthongs = list(set(detected_diphthongs))

    return {
        "Word": word,
        "Syllables": syllables_no_stress,
        "Diphthongs": detected_diphthongs
    }

def process_sentence(transcription):
    words = transcription.split()
    results = [detect_syllables_and_diphthongs(word.lower()) for word in words]
    return results

# Display results
results = process_sentence(transcription)
for result in results:
    print(f"Word: {result['Word']}")
    print(f"  Syllables: {', '.join(result['Syllables'])}")
    print(f"  Diphthongs Detected: {', '.join(result['Diphthongs']) if result['Diphthongs'] else 'None'}")
