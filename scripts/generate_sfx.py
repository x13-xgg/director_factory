"""Generate 20 procedural sound effect WAV files for the SFX library."""
import math
import random
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 44100
BITS = 16
MAX_AMP = 32767
ASSETS_DIR = Path(__file__).parent.parent / "sfx"


def write_wav(path: Path, samples: list[float], sr: int = SAMPLE_RATE):
    """Write mono 16-bit PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        clamped = [max(-1.0, min(1.0, s)) for s in samples]
        packed = b"".join(struct.pack("<h", int(s * MAX_AMP)) for s in clamped)
        wf.writeframes(packed)


def dur(seconds: float) -> int:
    return int(seconds * SAMPLE_RATE)


def sine(freq: float, length_samples: int, sr: int = SAMPLE_RATE) -> list[float]:
    return [math.sin(2 * math.pi * freq * i / sr) for i in range(length_samples)]


def noise(length_samples: int) -> list[float]:
    return [random.uniform(-1, 1) for _ in range(length_samples)]


def white_noise(length_samples: int) -> list[float]:
    return noise(length_samples)


def pink_noise(length_samples: int) -> list[float]:
    """Approximate pink noise (1/f spectrum)."""
    white = noise(length_samples)
    b = [0.0] * 7
    out = []
    for w in white:
        b[0] = 0.99886 * b[0] + w * 0.0555179
        b[1] = 0.99332 * b[1] + w * 0.0750759
        b[2] = 0.96900 * b[2] + w * 0.1538520
        b[3] = 0.86650 * b[3] + w * 0.3104856
        b[4] = 0.55000 * b[4] + w * 0.5329522
        b[5] = -0.7616 * b[5] - w * 0.0168980
        out.append(b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362)
        b[6] = w * 0.115926
    scale = max(abs(s) for s in out) if out else 1
    return [s / scale * 0.7 for s in out]


def envelope(samples: list[float], attack: float = 0.01, release: float = 0.05) -> list[float]:
    """Apply simple attack/release envelope."""
    n = len(samples)
    attack_samples = int(attack * SAMPLE_RATE)
    release_samples = int(release * SAMPLE_RATE)
    out = []
    for i, s in enumerate(samples):
        env = 1.0
        if i < attack_samples:
            env = i / attack_samples
        elif i >= n - release_samples:
            env = (n - i - 1) / max(release_samples, 1)
        out.append(s * env)
    return out


def generate_all():
    print("Generating SFX files...")

    # 1. footstep_concrete.wav — short percussive thuds
    steps = []
    for _ in range(4):
        tick = []
        for i in range(dur(0.08)):
            t = i / SAMPLE_RATE
            tick.append(math.sin(2 * math.pi * 800 * t) * math.exp(-t * 60) * 0.7)
        steps.extend(tick)
        steps.extend([0] * dur(0.15))
    write_wav(ASSETS_DIR / "footstep_concrete.wav", envelope(steps))
    print("  footstep_concrete.wav")

    # 2. footstep_gravel.wav — crunchier steps
    steps = []
    for _ in range(4):
        crunch = []
        for i in range(dur(0.12)):
            t = i / SAMPLE_RATE
            nz = random.uniform(-0.5, 0.5)
            tone = math.sin(2 * math.pi * 500 * t) * math.exp(-t * 40)
            crunch.append((nz + tone) * 0.5)
        steps.extend(crunch)
        steps.extend([0] * dur(0.18))
    write_wav(ASSETS_DIR / "footstep_gravel.wav", envelope(steps))
    print("  footstep_gravel.wav")

    # 3. wind_ambient.wav — filtered noise with slow modulation
    wind_len = dur(3.0)
    wind = []
    for i in range(wind_len):
        t = i / SAMPLE_RATE
        mod = 0.5 + 0.5 * math.sin(2 * math.pi * 0.3 * t)
        nz = pink_noise(1)[0] * 0.5
        wind.append(nz * mod)
    write_wav(ASSETS_DIR / "wind_ambient.wav", envelope(wind, attack=0.5, release=0.5))
    print("  wind_ambient.wav")

    # 4. wind_howl_gust.wav — wind with occasional gusts
    gust_len = dur(2.5)
    gust = []
    for i in range(gust_len):
        t = i / SAMPLE_RATE
        base = pink_noise(1)[0] * 0.3
        burst = 0
        if 0.5 < t < 1.2:
            burst = math.sin(2 * math.pi * 200 * t) * 0.4 * (t - 0.5) * (1.2 - t) * 3
        gust.append(base + burst)
    write_wav(ASSETS_DIR / "wind_howl_gust.wav", envelope(gust, attack=0.3, release=0.3))
    print("  wind_howl_gust.wav")

    # 5. rain_light.wav — high-frequency hiss
    rain_len = dur(3.0)
    rain = []
    for i in range(rain_len):
        t = i / SAMPLE_RATE
        # Simulate individual drops hitting at random intervals
        base_hiss = pink_noise(1)[0] * 0.15
        drop = 0
        # Occasional louder drops
        drop_phase = (t * 7.3) % 1.0
        if drop_phase < 0.02:
            drop = math.exp(-drop_phase * 200) * 0.6
        rain.append(base_hiss + drop)
    write_wav(ASSETS_DIR / "rain_light.wav", envelope(rain, attack=0.3, release=0.5))
    print("  rain_light.wav")

    # 6. thunder_distant.wav — low rumble with peak
    th_len = dur(2.0)
    thunder = []
    for i in range(th_len):
        t = i / SAMPLE_RATE
        low_rumble = pink_noise(1)[0] * 0.3
        # Low frequency sweep
        freq = 80 + 40 * math.sin(2 * math.pi * 2 * t)
        low_tone = math.sin(2 * math.pi * freq * t) * 0.2
        # Crack at 0.4s
        crack = 0
        if 0.3 < t < 0.6:
            crack = pink_noise(1)[0] * 0.6 * math.exp(-(t - 0.3) * 10)
        thunder.append(low_rumble + low_tone + crack)
    write_wav(ASSETS_DIR / "thunder_distant.wav", envelope(thunder, attack=0.1, release=0.4))
    print("  thunder_distant.wav")

    # 7. door_metal_creak.wav — slow frequency sweep
    creak_len = dur(1.5)
    creak = []
    for i in range(creak_len):
        t = i / SAMPLE_RATE
        freq = 400 + 600 * (t / 1.5)
        tone = math.sin(2 * math.pi * freq * t) * 0.3
        nz = pink_noise(1)[0] * 0.1
        # Squeak harmonics
        harm = math.sin(2 * math.pi * freq * 2.3 * t) * 0.1
        creak.append(tone + nz + harm)
    write_wav(ASSETS_DIR / "door_metal_creak.wav", envelope(creak))
    print("  door_metal_creak.wav")

    # 8. door_slam.wav — short impact
    slam_len = dur(0.5)
    slam = []
    for i in range(slam_len):
        t = i / SAMPLE_RATE
        impact = pink_noise(1)[0] * math.exp(-t * 20) * 0.8
        low = math.sin(2 * math.pi * 100 * t) * math.exp(-t * 15) * 0.4
        slam.append(impact + low)
    write_wav(ASSETS_DIR / "door_slam.wav", envelope(slam, attack=0.002, release=0.1))
    print("  door_slam.wav")

    # 9. glass_shatter.wav — high-frequency burst
    glass_len = dur(0.8)
    glass = []
    for i in range(glass_len):
        t = i / SAMPLE_RATE
        burst = white_noise(1)[0] * math.exp(-t * 8) * 0.7
        high = math.sin(2 * math.pi * 3000 * t) * math.exp(-t * 6) * 0.4
        glass.append(burst + high)
    write_wav(ASSETS_DIR / "glass_shatter.wav", envelope(glass, attack=0.001, release=0.2))
    print("  glass_shatter.wav")

    # 10. metal_impact.wav — resonant metallic ping
    metal_len = dur(1.0)
    metal = []
    for i in range(metal_len):
        t = i / SAMPLE_RATE
        ping1 = math.sin(2 * math.pi * 880 * t) * math.exp(-t * 5) * 0.35
        ping2 = math.sin(2 * math.pi * 1200 * t) * math.exp(-t * 7) * 0.25
        ping3 = math.sin(2 * math.pi * 2100 * t) * math.exp(-t * 10) * 0.15
        nz = pink_noise(1)[0] * math.exp(-t * 12) * 0.2
        metal.append(ping1 + ping2 + ping3 + nz)
    write_wav(ASSETS_DIR / "metal_impact.wav", envelope(metal, attack=0.002, release=0.3))
    print("  metal_impact.wav")

    # 11. engine_hum.wav — low constant hum with harmonics
    eng_len = dur(3.0)
    engine = []
    for i in range(eng_len):
        t = i / SAMPLE_RATE
        # Varying RPM
        rpm = 60 + 20 * math.sin(2 * math.pi * 0.5 * t)
        base = math.sin(2 * math.pi * rpm * t) * 0.3
        h2 = math.sin(2 * math.pi * rpm * 2 * t) * 0.15
        h3 = math.sin(2 * math.pi * rpm * 4 * t) * 0.08
        nz = pink_noise(1)[0] * 0.05
        engine.append(base + h2 + h3 + nz)
    write_wav(ASSETS_DIR / "engine_hum.wav", envelope(engine, attack=0.5, release=0.5))
    print("  engine_hum.wav")

    # 12. machinery_low_rumble.wav — industrial low rumble
    mach_len = dur(3.0)
    mach = []
    for i in range(mach_len):
        t = i / SAMPLE_RATE
        low1 = math.sin(2 * math.pi * 40 * t) * 0.3
        low2 = math.sin(2 * math.pi * 55 * t) * 0.25
        # Occasional clank
        clank = 0
        clank_phase = (t * 1.7) % 1.0
        if clank_phase < 0.03:
            clank = math.sin(2 * math.pi * 300 * t) * math.exp(-clank_phase * 200) * 0.4
        nz = pink_noise(1)[0] * 0.1
        mach.append(low1 + low2 + clank + nz)
    write_wav(ASSETS_DIR / "machinery_low_rumble.wav", envelope(mach, attack=0.5, release=0.5))
    print("  machinery_low_rumble.wav")

    # 13. electric_hum.wav — 50/60Hz buzz
    buzz_len = dur(2.0)
    buzz = []
    for i in range(buzz_len):
        t = i / SAMPLE_RATE
        h50 = math.sin(2 * math.pi * 50 * t) * 0.3
        h100 = math.sin(2 * math.pi * 100 * t) * 0.2
        h150 = math.sin(2 * math.pi * 150 * t) * 0.12
        h200 = math.sin(2 * math.pi * 200 * t) * 0.08
        nz = pink_noise(1)[0] * 0.05
        buzz.append(h50 + h100 + h150 + h200 + nz)
    write_wav(ASSETS_DIR / "electric_hum.wav", envelope(buzz, attack=0.2, release=0.2))
    print("  electric_hum.wav")

    # 14. fire_crackle.wav — random crackling
    fire_len = dur(3.0)
    fire = []
    for i in range(fire_len):
        t = i / SAMPLE_RATE
        base_crackle = pink_noise(1)[0] * 0.15
        crack = 0
        # Random pops
        pop_chance = random.random()
        if pop_chance < 0.08:
            decay = 0.05
            crack = white_noise(1)[0] * 0.5 * (pop_chance / 0.08)
        fire.append(base_crackle + crack)
    write_wav(ASSETS_DIR / "fire_crackle.wav", envelope(fire, attack=0.1, release=0.2))
    print("  fire_crackle.wav")

    # 15. water_drip_echo.wav — spaced drips with reverb tail
    drip_len = dur(2.5)
    drip = []
    # Drips at 0.3, 0.9, 1.6, 2.2 seconds
    drip_times = [0.3, 0.9, 1.6, 2.2]
    for i in range(drip_len):
        t = i / SAMPLE_RATE
        val = pink_noise(1)[0] * 0.02
        for dt in drip_times:
            delay = t - dt
            if 0 < delay < 0.8:
                freq = 1200
                ring = math.sin(2 * math.pi * freq * delay) * math.exp(-delay * 8) * 0.4
                val += ring
        drip.append(val)
    write_wav(ASSETS_DIR / "water_drip_echo.wav", envelope(drip, attack=0.05, release=0.1))
    print("  water_drip_echo.wav")

    # 16. silence_room_tone.wav — very quiet ambient
    room_len = dur(3.0)
    room = pink_noise(room_len)
    room = [s * 0.04 for s in room]  # Very quiet
    write_wav(ASSETS_DIR / "silence_room_tone.wav", envelope(room, attack=0.5, release=0.5))
    print("  silence_room_tone.wav")

    # 17. crowd_murmur.wav — filtered noise with speech-like modulation
    crowd_len = dur(3.0)
    crowd = []
    for i in range(crowd_len):
        t = i / SAMPLE_RATE
        nz = pink_noise(1)[0] * 0.2
        # Speech-like modulation at ~4-8 Hz
        mod = 0.5 + 0.5 * math.sin(2 * math.pi * 6 * t)
        # Occasional louder "word"
        word_burst = 0
        if random.random() < 0.03:
            word_burst = pink_noise(1)[0] * 0.3
        crowd.append(nz * mod + word_burst)
    write_wav(ASSETS_DIR / "crowd_murmur.wav", envelope(crowd, attack=0.2, release=0.3))
    print("  crowd_murmur.wav")

    # 18. birds_ambient.wav — chirping patterns
    bird_len = dur(3.0)
    birds = []
    chirp_schedule = [
        (0.2, 1500, 0.15), (0.6, 1800, 0.12), (1.0, 2200, 0.1),
        (1.5, 1600, 0.13), (2.0, 2000, 0.11), (2.5, 1400, 0.14),
    ]
    for i in range(bird_len):
        t = i / SAMPLE_RATE
        val = pink_noise(1)[0] * 0.02
        for chirp_t, chirp_freq, chirp_dur in chirp_schedule:
            delay = t - chirp_t
            if 0 < delay < chirp_dur:
                # Frequency sweep down
                sweep_freq = chirp_freq * (1 - delay / chirp_dur * 0.3)
                tone = math.sin(2 * math.pi * sweep_freq * delay) * math.exp(-delay * 20) * 0.15
                val += tone
        birds.append(val)
    write_wav(ASSETS_DIR / "birds_ambient.wav", envelope(birds, attack=0.1, release=0.1))
    print("  birds_ambient.wav")

    # 19. heartbeat_deep.wav — low-frequency thumps
    hb_len = dur(3.0)
    heartbeat = []
    beat_times = [0.2, 0.9, 1.6, 2.3]
    for i in range(hb_len):
        t = i / SAMPLE_RATE
        val = 0.0
        for bt in beat_times:
            delay = t - bt
            if 0 < delay < 0.15:
                # Double-thump (lub-dub)
                thump1 = math.sin(2 * math.pi * 50 * delay) * math.exp(-delay * 20) * 0.6
                thump2 = 0
                if delay > 0.05:
                    d2 = delay - 0.05
                    thump2 = math.sin(2 * math.pi * 60 * d2) * math.exp(-d2 * 15) * 0.4
                val += thump1 + thump2
        heartbeat.append(val)
    write_wav(ASSETS_DIR / "heartbeat_deep.wav", envelope(heartbeat))
    print("  heartbeat_deep.wav")

    # 20. static_radio.wav — white noise with occasional signal
    static_len = dur(2.5)
    static = []
    for i in range(static_len):
        t = i / SAMPLE_RATE
        nz = white_noise(1)[0] * 0.3
        # Occasional "signal" — radio-like tones
        signal = 0
        signal_phase = (t * 0.7) % 1.0
        if signal_phase < 0.15:
            signal = math.sin(2 * math.pi * 1000 * t) * 0.15
        static.append(nz + signal)
    write_wav(ASSETS_DIR / "static_radio.wav", envelope(static, attack=0.1, release=0.1))
    print("  static_radio.wav")

    print(f"\nDone! 20 SFX files generated in {ASSETS_DIR}")


if __name__ == "__main__":
    generate_all()
