import soundfile as sf
import numpy as np

from .segment import Segment
from .log import logging
from .label import Label
from modules.pe import initialize_pe

def combine_labels(labels): # combining labels that pau boundaries
    if len(labels) == 1:
        return labels[0]

    res = []
    for l in labels:
        res = res[:-1]
        res.extend(l)
    return Segment(res)

def label_from_line(line): # yeah..
    s, e, p = line.strip().split()
    s = float(s) / 10000000
    e = float(e) / 10000000
    return Label(s, e, p)

def read_label(path): # yeah..
    labels = []
    for line in open(path).readlines():
        labels.append(label_from_line(line))

    return Segment(labels)

def write_label(path, label, isHTK=True): # write label with start offset
    offset = label[0].start
    with open(path, 'w', encoding='utf8') as f:
        for l in label:
            if isHTK:
                f.write(f'{int(10000000 * (l.start - offset))} {int(10000000 * (l.end - offset))} {l.phone}\n')
            else:
                f.write(f'{l.start - offset}\t{l.end - offset}\t{l.phone}\n')

def get_pitch(x, fs, hparams): # parselmouth F0
    pe_cls = initialize_pe(hparams)
    f0 = pe_cls.get_pitch(x, fs, hparams)

    return f0

# From MakeDiffSinger/variance-temp-solution/get_pitch.py
def norm_f0(f0):
    f0 = np.log2(f0)
    return f0

def denorm_f0(f0, uv, pitch_padding=None):
    f0 = 2 ** f0
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0

def interp_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0)
    if sum(uv) == len(f0):
        f0[uv] = -np.inf
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None), uv


def write_ds(loc, wav, fs, pitch='parselmouth', time_step=0.005, f0_min=40, f0_max=1100, voicing_threshold=0.45, **kwargs):
    res = {'offset' : 0}
    res['text'] = kwargs['ph_seq']
    res['ph_seq'] = kwargs['ph_seq']
    res['ph_dur'] = kwargs['ph_dur']
    if 'ph_num' in list(kwargs.keys()):
        res['ph_num'] = kwargs['ph_num']
        if 'note_seq' in list(kwargs.keys()):
            res['note_seq'] = kwargs['note_seq']
            res['note_dur'] = kwargs['note_dur']
            res['note_slur'] = ' '.join(['0'] * len(kwargs['note_dur']))
    f0 = pitch
    if isinstance(pitch, str):
        f0 = get_pitch(wav, fs, hparams)
    timestep = time_step
    f0, _ = interp_f0(f0)
    res['f0_seq'] = ' '.join([str(round(x, 1)) for x in f0])
    res['f0_timestep'] = str(timestep)

    with open(loc, 'w', encoding='utf8') as f:
        json.dump([res], f, indent=4)

def process_lab_wav_pair(segment_loc, lab, wav, hparams, language_def, lang=None):
    logging.info(f'Reading {wav}.')
    x, fs = sf.read(wav)

    if x.ndim > 1:
        x = np.mean(x, axis=1)

    if hparams.audio_sample_rate != 0 and fs != hparams.audio_sample_rate:
        x = librosa.resample(x, orig_sr=fs, target_sr=hparams.audio_sample_rate)
        fs = hparams.audio_sample_rate

    pitch = hparams.pe # precalculate midi pitch
    if hparams.estimate_midi or hparams.write_ds:
        logging.info(f'Estimating pitch for {wav}')
        pitch = get_pitch(x, fs, hparams)
    
    logging.info(f'Segmenting {lab}.')
    fname = lab.stem

    segments = read_label(lab).segment_label(max_length=hparams.max_length, max_silences=hparams.max_silences, length_relax=hparams.max_length_relaxation_factor)
    logging.info('Splitting wave file and preparing transcription lines.')
    transcripts = []
    for i in range(len(segments)):
        segment = segments[i]
        segment_name = f'{fname}_seg{i:03d}'
        logging.info(f'Segment {i+1} / {len(segments)}')

        s = int(fs * segment.start)
        e = int(fs * segment.end)
        p_s = int(segment.start / hparams.time_step)
        p_e = int(segment.end / hparams.time_step)
        segment_wav = x[s:e]
        segment_pitch = pitch
        if hparams.estimate_midi or hparams.write_ds:
            segment_pitch = pitch[p_s:p_e]

        if hparams.detect_breaths:
            segment.detect_breath(segment_wav, fs, hparams)

        transcript_row = {
            'name' : segment_name,
            'ph_seq' : segment.to_phone_string(),
            'ph_dur' : segment.to_lengths_string()
            }

        if language_def:
            transcript_row['ph_num'], split_pos = segment.to_phone_nums_string(lang=lang)
            dur = transcript_row['ph_dur'].split()
            num = [int(x) for x in transcript_row['ph_num'].split()]
            assert len(dur) == sum(num), 'Ops'
            if hparams.estimate_midi:
                note_seq, note_dur = segment.to_midi_strings(segment_wav, fs, split_pos, hparams)
                transcript_row['note_seq'] = note_seq
                transcript_row['note_dur'] = note_dur

        all_pau = np.all(np.fromiter(map(lambda x : x in ['SP', 'AP'], transcript_row['ph_seq'].split()), bool))
        all_rest = False
        if hparams.estimate_midi:
            all_rest = np.all(np.fromiter(map(lambda x : x == 'rest', transcript_row['note_seq'].split()), bool))

        if not (all_pau or all_rest):
            sf.write(segment_loc / (segment_name + '.wav'), segment_wav, fs)
            transcripts.append(transcript_row)
            if hparams.write_labels:
                isHTK = hparams.write_labels.lower() == 'htk'
                write_label(segment_loc / (segment_name + ('.lab' if isHTK else '.txt')), segment, isHTK)
            
            if hparams.write_ds:
                write_ds(segment_loc / (segment_name + '.ds'), segment_wav, fs, hparams **transcript_row)
        else:
            logging.warning('Detected pure silence either from segment label or note sequence. Skipping.')
    
    return transcripts