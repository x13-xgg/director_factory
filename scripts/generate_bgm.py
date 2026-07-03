"""Generate BGM (background music) files for Director Factory.

Each track is a short looping ambient/musical piece with distinct emotional character.
"""
import math
import random
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 44100
BITS = 16
MAX_AMP = 32767
BGM_DIR = Path(__file__).parent.parent / "assets" / "bgm"


def write_wav(path: Path, samples: list[float], sr: int = SAMPLE_RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        clamped = [max(-1.0, min(1.0, s)) for s in samples]
        packed = b"".join(struct.pack("<h", int(s * MAX_AMP)) for s in clamped)
        wf.writeframes(packed)


def sine(freq: float, duration: float) -> list[float]:
    n = int(duration * SAMPLE_RATE)
    return [math.sin(2 * math.pi * freq * i / SAMPLE_RATE) for i in range(n)]


def saw(freq: float, duration: float) -> list[float]:
    n = int(duration * SAMPLE_RATE)
    return [2 * ((freq * i / SAMPLE_RATE) % 1) - 1 for i in range(n)]


def square(freq: float, duration: float, duty: float = 0.5) -> list[float]:
    n = int(duration * SAMPLE_RATE)
    out = []
    for i in range(n):
        phase = (freq * i / SAMPLE_RATE) % 1
        out.append(0.6 if phase < duty else -0.6)
    return out


def envelope(samples: list[float], attack: float = 0.01, release: float = 0.05) -> list[float]:
    n = len(samples)
    a_s = int(attack * SAMPLE_RATE)
    r_s = int(release * SAMPLE_RATE)
    out = list(samples)
    for i in range(min(a_s, n)):
        out[i] *= i / a_s
    for i in range(min(r_s, n)):
        out[n - 1 - i] *= i / max(r_s, 1)
    return out


def reverb_lite(samples: list[float], decay: float = 0.3, delay_ms: int = 80) -> list[float]:
    """Simple delay-based pseudo-reverb."""
    delay_samples = int(delay_ms / 1000 * SAMPLE_RATE)
    n = len(samples)
    out = list(samples)
    for i in range(delay_samples, n):
        out[i] += out[i - delay_samples] * decay
    peak = max(abs(s) for s in out) or 1
    return [s / peak * 0.85 for s in out]


def pad_synth(freqs: list[float], duration: float, waveform="sine") -> list[float]:
    """Layered pad synth."""
    n = int(duration * SAMPLE_RATE)
    result = [0.0] * n
    for freq in freqs:
        vol = 0.3 / len(freqs)
        for i in range(n):
            t = i / SAMPLE_RATE
            # Slow LFO on amplitude
            lfo = 0.7 + 0.3 * math.sin(2 * math.pi * 0.2 * t + freq)
            if waveform == "sine":
                result[i] += math.sin(2 * math.pi * freq * t) * vol * lfo
            elif waveform == "saw":
                result[i] += (2 * ((freq * t) % 1) - 1) * vol * 0.4 * lfo
    return result


def arpeggiate(notes: list[float], tempo_bpm: float, duration: float, waveform="sine") -> list[float]:
    """Simple arpeggiator."""
    beat = 60 / tempo_bpm
    n = int(duration * SAMPLE_RATE)
    result = [0.0] * n
    for j, freq in enumerate(notes):
        start = int(j * beat * SAMPLE_RATE)
        note_len = int(beat * 0.9 * SAMPLE_RATE)
        for i in range(note_len):
            idx = start + i
            if idx >= n:
                break
            t = i / SAMPLE_RATE
            env = math.exp(-t * 3)
            if waveform == "sine":
                result[idx] += math.sin(2 * math.pi * freq * t) * env * 0.25
            elif waveform == "square":
                phase = (freq * t) % 1
                result[idx] += (0.6 if phase < 0.5 else -0.6) * env * 0.15
    return result


def note_to_freq(note: str) -> float:
    """Convert note name to frequency. A4 = 440Hz."""
    notes = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4,
             "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    name = note.rstrip("0123456789")
    octave = int(note[len(name):]) if note[len(name):] else 4
    semitone = notes.get(name, 0)
    semitones_from_a4 = (octave - 4) * 12 + semitone
    return 440 * (2 ** (semitones_from_a4 / 12))


def generate_all():
    print("Generating BGM files...")
    DURATION = 12.0  # seconds per track

    # 1. tension_bed.wav — dark ambient, low drones, minor key
    print("  tension_bed.wav")
    n = int(DURATION * SAMPLE_RATE)
    tension = [0.0] * n
    # Bass drone
    for i in range(n):
        t = i / SAMPLE_RATE
        lfo = 0.6 + 0.4 * math.sin(2 * math.pi * 0.12 * t)
        tension[i] += math.sin(2 * math.pi * 55 * t) * 0.25 * lfo
        tension[i] += math.sin(2 * math.pi * 110 * t) * 0.08 * lfo
        # High tension shimmer
        tension[i] += math.sin(2 * math.pi * 800 * t) * 0.03 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * t))
    tension = reverb_lite(tension, 0.4)
    write_wav(BGM_DIR / "tension_bed.wav", envelope(tension, attack=0.8, release=1.5))

    # 2. hope_theme.wav — orchestral, bright, major key arpeggios
    print("  hope_theme.wav")
    hop = [0.0] * n
    pad = pad_synth([note_to_freq("C4"), note_to_freq("E4"), note_to_freq("G4")], DURATION, "sine")
    arp_notes = [note_to_freq(n) for n in ["C4", "E4", "G4", "C5", "G4", "E4"]]
    arp = arpeggiate(arp_notes * 4, 80, DURATION, "sine")
    for i in range(n):
        hop[i] = pad[i] * 0.5 + arp[i] * 0.5
    hop = reverb_lite(hop, 0.35, 120)
    write_wav(BGM_DIR / "hope_theme.wav", envelope(hop, attack=1.0, release=2.0))

    # 3. sad_strings.wav — solo strings feel, slow melody in minor
    print("  sad_strings.wav")
    sad = [0.0] * n
    melody = [note_to_freq(n) for n in ["A3", "C4", "E4", "D4", "A3", "G3", "F3", "E3"]]
    pad = pad_synth([note_to_freq("A3"), note_to_freq("C4")], DURATION, "sine")
    for j, freq in enumerate(melody):
        start = int(j * 1.5 * SAMPLE_RATE)
        note_len = int(1.4 * SAMPLE_RATE)
        for i in range(note_len):
            idx = start + i
            if idx >= n:
                break
            t = i / SAMPLE_RATE
            env = math.exp(-t * 1.5)
            vibrato = 1 + 0.005 * math.sin(2 * math.pi * 5 * t)
            sad[idx] += math.sin(2 * math.pi * freq * vibrato * t) * env * 0.35
    for i in range(n):
        sad[i] += pad[i] * 0.2
    sad = reverb_lite(sad, 0.5, 150)
    write_wav(BGM_DIR / "sad_strings.wav", envelope(sad, attack=1.0, release=2.0))

    # 4. joy_theme.wav — upbeat, major key, bright
    print("  joy_theme.wav")
    joy = [0.0] * n
    pad = pad_synth([note_to_freq("G3"), note_to_freq("B3"), note_to_freq("D4")], DURATION, "sine")
    arp = arpeggiate([note_to_freq(n) for n in ["G3", "B3", "D4", "G4", "D4", "B3"]] * 6, 120, DURATION, "sine")
    for i in range(n):
        joy[i] = pad[i] * 0.4 + arp[i] * 0.6
    joy = reverb_lite(joy, 0.25, 80)
    write_wav(BGM_DIR / "joy_theme.wav", envelope(joy, attack=0.5, release=1.5))

    # 5. fear_drone.wav — low drone, dissonant, unsettling
    print("  fear_drone.wav")
    fear = [0.0] * n
    for i in range(n):
        t = i / SAMPLE_RATE
        # Tritone interval (dissonant)
        fear[i] += math.sin(2 * math.pi * 60 * t) * 0.2
        fear[i] += math.sin(2 * math.pi * 82.5 * t) * 0.15
        # Slow filter sweep
        mod = 0.3 + 0.7 * abs(math.sin(2 * math.pi * 0.08 * t))
        fear[i] *= mod
        # Occasional sting
        fear[i] += math.sin(2 * math.pi * 400 * t) * 0.04 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.15 * t))
    fear = reverb_lite(fear, 0.6)
    write_wav(BGM_DIR / "fear_drone.wav", envelope(fear, attack=1.5, release=2.0))

    # 6. action_beat.wav — electronic, bass-heavy, rhythmic
    print("  action_beat.wav")
    action = [0.0] * n
    bpm = 140
    beat = 60 / bpm
    # Kick
    for b in range(int(DURATION / beat)):
        start = int(b * beat * SAMPLE_RATE)
        for i in range(int(0.15 * SAMPLE_RATE)):
            idx = start + i
            if idx >= n:
                break
            t = i / SAMPLE_RATE
            action[idx] += math.sin(2 * math.pi * 60 * t) * math.exp(-t * 30) * 0.6
    # Bass line
    bass_notes = [note_to_freq(n) for n in ["E2", "E2", "G2", "A2"]] * 4
    for j, freq in enumerate(bass_notes):
        start = int(j * beat * SAMPLE_RATE)
        for i in range(int(beat * 0.7 * SAMPLE_RATE)):
            idx = start + i
            if idx >= n:
                break
            t = i / SAMPLE_RATE
            action[idx] += (2 * ((freq * t) % 1) - 1) * 0.12 * math.exp(-t * 2)
    # Hi-hat
    for b in range(int(DURATION / (beat / 2))):
        start = int(b * beat / 2 * SAMPLE_RATE)
        for i in range(int(0.03 * SAMPLE_RATE)):
            idx = start + i
            if idx >= n:
                break
            action[idx] += random.uniform(-0.15, 0.15) * math.exp(-i / SAMPLE_RATE * 200)
    write_wav(BGM_DIR / "action_beat.wav", envelope(action, attack=0.3, release=0.5))

    # 7. lonely_piano.wav — sparse piano notes, minor key, lots of space
    print("  lonely_piano.wav")
    lonely = [0.0] * n
    piano_notes = [note_to_freq(n) for n in ["F3", "Ab3", "C4", "Eb4", "Bb3", "Ab3", "F3", "Eb3"]]
    for j, freq in enumerate(piano_notes):
        start = int(j * 2.0 * SAMPLE_RATE)
        note_len = int(2.5 * SAMPLE_RATE)
        for i in range(note_len):
            idx = start + i
            if idx >= n:
                break
            t = i / SAMPLE_RATE
            # Piano-like: fundamental + harmonics with fast decay
            env = math.exp(-t * 2)
            val = math.sin(2 * math.pi * freq * t) * env * 0.4
            val += math.sin(2 * math.pi * freq * 2 * t) * env * 0.15
            val += math.sin(2 * math.pi * freq * 3 * t) * env * 0.07
            lonely[idx] += val
    lonely = reverb_lite(lonely, 0.45, 200)
    write_wav(BGM_DIR / "lonely_piano.wav", envelope(lonely, attack=0.3, release=3.0))

    # 8. serene_pad.wav — warm ambient pad, major 7th chords
    print("  serene_pad.wav")
    serene = pad_synth(
        [note_to_freq("Eb3"), note_to_freq("G3"), note_to_freq("Bb3"), note_to_freq("D4")],
        DURATION, "sine"
    )
    for i in range(n):
        t = i / SAMPLE_RATE
        lfo = 0.7 + 0.3 * math.sin(2 * math.pi * 0.08 * t)
        serene[i] *= lfo
    serene = reverb_lite(serene, 0.4, 150)
    write_wav(BGM_DIR / "serene_pad.wav", envelope(serene, attack=2.0, release=3.0))

    # 9. wistful_guitar.wav — acoustic feel, gentle arpeggios
    print("  wistful_guitar.wav")
    wist = [0.0] * n
    chord_seq = [
        [note_to_freq("D3"), note_to_freq("F#3"), note_to_freq("A3")],
        [note_to_freq("B2"), note_to_freq("D3"), note_to_freq("F#3")],
        [note_to_freq("G2"), note_to_freq("B2"), note_to_freq("D3")],
        [note_to_freq("A2"), note_to_freq("C#3"), note_to_freq("E3")],
    ]
    for c_idx, chord in enumerate(chord_seq * 2):
        start = int(c_idx * 1.5 * SAMPLE_RATE)
        for f in chord:
            for i in range(int(1.5 * SAMPLE_RATE)):
                idx = start + i
                if idx >= n:
                    break
                t = i / SAMPLE_RATE
                env = math.exp(-t * 3)
                # Simulate plucked string
                val = math.sin(2 * math.pi * f * t) * env * 0.2
                val += math.sin(2 * math.pi * f * 2 * t) * env * 0.06
                wist[idx] += val
    wist = reverb_lite(wist, 0.35, 100)
    write_wav(BGM_DIR / "wistful_guitar.wav", envelope(wist, attack=0.1, release=1.5))

    # 10. neutral_bed.wav — light ambient, unobtrusive
    print("  neutral_bed.wav")
    neutral = pad_synth(
        [note_to_freq("C3"), note_to_freq("G3"), note_to_freq("C4")],
        DURATION, "sine"
    )
    for i in range(n):
        t = i / SAMPLE_RATE
        lfo = 0.8 + 0.2 * math.sin(2 * math.pi * 0.1 * t + i * 0.0001)
        neutral[i] *= lfo
    neutral = reverb_lite(neutral, 0.3, 100)
    write_wav(BGM_DIR / "neutral_bed.wav", envelope(neutral, attack=1.5, release=2.0))

    print(f"\nDone! 10 BGM files generated in {BGM_DIR}")


if __name__ == "__main__":
    generate_all()
