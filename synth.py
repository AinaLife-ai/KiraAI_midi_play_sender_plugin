"""
Pure Python MIDI synthesizer - anti-aliased, click-free, simplified.
移植自 midi-player skill 的 scripts/synth.py（原样保留）。
"""
import numpy as np

PRESETS = {
    "piano":   ([(1,1.0),(2,0.42),(3,0.18),(5,0.07)], 0.004, 2.0, 0.9, 0.2),
    "guitar":  ([(1,1.0),(2,0.48),(3,0.15),(4,0.06)],  0.003, 2.2, 0.80, 0.3),
    "nylon":   ([(1,1.0),(2,0.40),(3,0.10)],            0.005, 2.5, 0.75, 0.35),
    "organ":   ([(1,1.0),(2,0.6),(4,0.3),(6,0.12)],     0.02,  0.5, 0.85, 0.1),
    "8bit":    ([],                                      0.002, 0.7, 0.9, 0.0),
    "square":  ([(1,1.0),(3,0.25),(5,0.12),(7,0.07)],   0.002, 0.6, 0.75, 0.0),
    "strings": ([(1,1.0),(2,0.30),(3,0.12)],             0.08,  1.5, 0.65, 0.25),
    "bell":    ([(1,1.0),(2,0.7),(3,0.4),(4,0.2),(5.2,0.3)], 0.001, 0.2, 1.2, 0.05),
    "flute":   ([(1,1.0),(2,0.15)],                       0.03,  0.6, 1.0, 0.15),
    "default": ([(1,1.0),(2,0.25),(3,0.08)],              0.005, 1.0, 0.9, 0.15),
}


class SineSynth:
    """Polyphonic synthesizer with smooth envelopes and anti-aliasing."""

    def __init__(self, sample_rate: int = 44100, preset: str = "default"):
        self.sr = sample_rate
        self.voices: dict = {}
        self.harmonics, self.attack, self.decay, self.brightness, self.release = PRESETS.get(
            preset, PRESETS["default"]
        )
        self.preset_name = preset

    @staticmethod
    def note_to_freq(note: int) -> float:
        return 440.0 * (2 ** ((note - 69) / 12.0))

    def noteon(self, channel: int, note: int, velocity: int):
        freq = self.note_to_freq(note)
        self.voices[(channel, note)] = {
            'vel': velocity / 127.0,
            'phase': 0.0,
            'freq': freq,
            'age': 0,
            'releasing': False,
            'rel_age': 0,
        }

    def noteoff(self, channel: int, note: int):
        key = (channel, note)
        if key in self.voices and not self.voices[key]['releasing']:
            self.voices[key]['releasing'] = True
            self.voices[key]['rel_age'] = 0

    def render(self, n_samples: int) -> np.ndarray:
        buf = np.zeros((n_samples, 2), dtype=np.float32)
        t = np.arange(n_samples, dtype=np.float32) / self.sr
        to_remove = []
        release_samples = int(self.release * self.sr)
        f_nyq = self.sr * 0.47

        for key, v in list(self.voices.items()):
            vel = v['vel']
            phase = v['phase']
            freq = v['freq']
            age = v['age']
            releasing = v['releasing']
            rel_age = v['rel_age']

            # Generate waveform for this block
            wave = np.zeros(n_samples, dtype=np.float32)

            if self.preset_name in ("8bit", "square"):
                for hnum in [1, 3, 5, 7]:
                    fh = freq * hnum
                    if fh >= f_nyq: continue
                    amp = (1.0 / hnum) * (1.0 - (fh / f_nyq))
                    wave += amp * np.sin(2 * np.pi * fh * (t + phase), dtype=np.float32)
                wave *= 0.55
            else:
                for hnum, amp in self.harmonics:
                    fh = freq * hnum
                    if fh >= f_nyq: continue
                    wave += amp * np.sin(2 * np.pi * fh * (t + phase), dtype=np.float32)

            wave *= vel * 0.22 * self.brightness

            # Envelope
            if releasing:
                rel_progress = rel_age / release_samples if release_samples > 0 else 1.0
                env = np.exp(-(rel_progress + t * self.sr / release_samples) * 6.0, dtype=np.float32)
                env *= np.exp(-age / (self.sr * self.decay * 0.5))
            else:
                env = np.ones(n_samples, dtype=np.float32)
                atk_s = int(self.attack * self.sr)
                if age < atk_s:
                    for i in range(min(n_samples, atk_s - age)):
                        r = (age + i) / max(atk_s, 1)
                        env[i] = 0.5 - 0.5 * np.cos(np.pi * r)
                sustain_env = np.exp(-(age + np.arange(n_samples)) / (self.sr * self.decay), dtype=np.float32)
                sustain_env = np.maximum(sustain_env, 0.25)
                env *= sustain_env

            wave *= env
            buf[:, 0] += wave * 0.65
            buf[:, 1] += wave * 0.65

            # Update state
            v['phase'] = phase + n_samples / self.sr
            v['age'] = age + n_samples
            if releasing:
                v['rel_age'] = rel_age + n_samples
                if v['rel_age'] >= release_samples:
                    to_remove.append(key)

        for key in to_remove:
            self.voices.pop(key, None)

        # Soft saturation
        buf = np.tanh(buf * 1.2) / 1.2
        return buf.astype(np.float32)
