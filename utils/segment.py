from copy import deepcopy
import numpy as np
import math
import parselmouth as pm
import librosa

class Segment: # should've been named segment in hindsight...
    def __init__(self, labels):
        self.labels = deepcopy(labels) # list of Labels
        self.pauses = ['sil', 'pau', 'SP']

    def __sub__(self, other):
        labels = []
        for lab in self.labels:
            labels.append(lab - other)
        return Segment(labels)
    
    def __add__(self, other):
        labels = []
        for lab in self.labels:
            labels.append(lab + other)
        return Segment(labels)

    def length(self): # total length in seconds
        lens = [x.length() for x in self.labels]
        return math.fsum(lens)

    def to_phone_string(self): # space separated phonemes
        phones = []
        for l in self.labels:
            p = l.phone.replace('pau', 'SP').replace('sil', 'SP').replace('br', 'AP')
            phones.append(p)
        return ' '.join(phones)

    def to_lengths_string(self): # space separated lengths
        return ' '.join([str(round(x.length(), 12)) for x in self.labels])
    
    def to_phone_nums_string(self, lang): # phoneme separations
        # Find all vowel positions
        vowel_pos = []
        if self.labels[0].phone not in self.pauses + lang['vowels'] + ['br', 'AP']:
            vowel_pos.append(0)

        for i in range(len(self.labels)):
            l = self.labels[i]
            if l.phone in lang['vowels']:
                prev_l = self.labels[i-1]
                if prev_l.phone in lang['liquids'].keys(): # check liquids before vowel.
                    # if the value for the liquid is true, move position for any consonant, else, move position for specified consonants.
                    liquid = lang['liquids'][prev_l.phone]
                    if liquid == True:
                        if self.labels[i-2].phone not in self.pauses + lang['vowels']:
                            vowel_pos.append(i-1)
                        else:
                            vowel_pos.append(i)
                    elif self.labels[i-2].phone in liquid:
                        vowel_pos.append(i-1)
                    else:
                        vowel_pos.append(i)
                else:
                    vowel_pos.append(i)
            elif l.phone in self.pauses + ['br', 'AP']:
                vowel_pos.append(i)
        vowel_pos.append(len(self))

        # use diff to calculate ph_num
        ph_num = np.diff(vowel_pos)
        return ' '.join(map(str, ph_num)), vowel_pos
    
    def to_midi_strings(self, x, fs, split_pos, hparams): # midi estimation
        # init hparams
        pitch = hparams.pe
        time_step = hparams.time_step
        f0_min = hparams.f0_min
        f0_max = hparams.f0_max
        voicing_threshold = hparams.voicing_threshold_midi
        cents = hparams.use_cents

        pps = 1 / time_step
        f0 = hparams.pe
        if isinstance(hparams.pe, str):
            f0 = get_pitch(x, fs, hparams) # get pitch
        midi_pitch = np.copy(f0)
        midi_pitch[midi_pitch > 0] = librosa.hz_to_midi(midi_pitch[midi_pitch > 0])

        if midi_pitch.size < self.length() * pps:
            pad = math.ceil(self.length() * pps) - midi_pitch.size
            midi_pitch = np.pad(midi_pitch, [0, pad], mode='edge')

        note_seq = []
        note_dur = []

        temp_label = deepcopy(self) - self.labels[0].start # offset label to have it start at 0 because this receives segmented wavs
        for i in range(len(split_pos) - 1): # for each split
            s = split_pos[i]
            e = split_pos[i+1]

            note_lab = Segment(temp_label[s:e]) # temp label
            p_s = math.floor((note_lab.labels[0].start) * pps)
            p_e = math.ceil((note_lab.labels[-1].end) * pps)

            # check for rests
            is_rest = False
            note_lab_phones = [x.phone for x in note_lab.labels]
            for pau in self.pauses:
                if pau in note_lab_phones:
                    is_rest = True
                    break
            
            if is_rest:
                note_seq.append('rest') 
            else: # get modal pitch
                note_pitch = midi_pitch[p_s:p_e]
                note_pitch = note_pitch[note_pitch > 0]
                if note_pitch.size > 0:
                    counts = np.bincount(np.round(note_pitch).astype(np.int64))
                    midi = counts.argmax()
                    if hparams.use_cents:
                        midi = np.mean(note_pitch[(note_pitch >= midi - 0.5) & (note_pitch < midi + 0.5)])
                    note_seq.append(librosa.midi_to_note(midi, cents=hparams.use_cents, unicode=False))
                else:
                    note_seq.append('rest')

            note_dur.append(note_lab.length())
        
        return ' '.join(note_seq), ' '.join(map(lambda x : str(round(x, 12)), note_dur))

    def detect_breath(self, x, fs, hparams):
        # Initialize hparams
        time_step = hparams.time_step
        f0_min = hparams.f0_min
        f0_max = hparams.f0_max
        voicing_threshold = hparams.voicing_threshold_breath
        window = hparams.breath_window_size
        min_len = hparams.breath_min_length
        min_db = hparams.breath_db_threshold
        min_centroid = hparams.breath_centroid_threshold

        # Referenced from MakeDiffSinger/acoustic_forced_alignment/enhance_tg.py
        # Features needed
        sound = pm.Sound(x, sampling_frequency=fs)
        f0 = get_pitch(x, fs, hparams) # VUVs
        hop_size = int(time_step * fs)
        centroid = librosa.feature.spectral_centroid(y=x, sr=fs, hop_length=hop_size).squeeze(0) # centroid
        rms = librosa.amplitude_to_db(librosa.feature.rms(y=x, hop_length=hop_size).squeeze(0)) # RMS for peak/dip searching

        ap_ranges = [] # all AP ranges
        temp_label = deepcopy(self) - self.labels[0].start
        for lab in temp_label.labels: 
            if lab.phone not in self.pauses: # skip non pause phonemes
                continue

            if lab.length() < min_len: # skip pauses shorter than min breath len
                continue
 
            temp_ap_ranges = [] 
            br_start = None
            win_pos = lab.start
            while win_pos + window <= lab.end: # original algorithm from reference code
                all_unvoiced = (f0[int(win_pos / time_step) : int((win_pos + window) / time_step)] < f0_min).all()
                rms_db = 20 * np.log10(np.clip(sound.get_rms(from_time=win_pos, to_time=win_pos + window), 1e-12, 1))

                if all_unvoiced and rms_db >= min_db:
                    if br_start is None:
                        br_start = win_pos
                else:
                    if br_start is not None:
                        br_end = win_pos + window - time_step
                        if br_end - br_start >= min_len:
                            mean_centroid = centroid[int(br_start / time_step):int(br_end / time_step)].mean()
                            if mean_centroid >= min_centroid:
                                temp_ap_ranges.append((br_start, br_end))
                win_pos += time_step
            if br_start is not None:
                br_end = win_pos + window - time_step
                if br_end - br_start >= min_len:
                    mean_centroid = centroid[int(br_start / time_step):int(br_end / time_step)].mean()
                    if mean_centroid >= min_centroid:
                        temp_ap_ranges.append((br_start, br_end))
            
            if len(temp_ap_ranges) == 0: # skip if no AP was found
                continue
            
            # combine AP ranges with similar starts
            clean_ap_ranges = [list(temp_ap_ranges[0])]
            for ap_start, ap_end in temp_ap_ranges: 
                if clean_ap_ranges[-1][0] == ap_start:
                    clean_ap_ranges[-1][1] = ap_end
                else:
                    clean_ap_ranges.append([ap_start, ap_end])

            resized_ap_ranges = []
            # resize AP ranges by finding the peak, finding the dips and finding the deepest dip closest to the peak on both sides
            for ap_start, ap_end in clean_ap_ranges:
                s = int((ap_start + window) / time_step) # add window to remove potential energy spike from voiced section before the pause
                e = int(ap_end / time_step)

                if s >= e: # breath too short, can't analyze
                    resized_ap_ranges.append((ap_start, ap_end))
                    continue
                
                peak = np.argmax(rms[s:e]) + s
                peaks = signal.find_peaks_cwt(rms[s:e], np.arange(6, 10)) + s # if successful, it finds the breath peak better than argmax
                if peaks.size != 0:
                    peak = peaks[np.argmax(rms[peaks])]

                dips = signal.find_peaks_cwt(-rms[s:e], np.arange(1, 10)) + s
                
                if dips.size == 0: # can't resize if there are no dips
                    resized_ap_ranges.append((ap_start, ap_end))
                    continue
                
                # binary search nearby dips from peak
                L = 0
                R = len(dips) - 1

                while L != R:
                    m = math.ceil((L + R) / 2)
                    if dips[m] > peak:
                        R = m - 1
                    else:
                        L = m
                
                R = min(L + 1, len(dips) - 1)

                # find dips to the left and right until the dip before or after is higher than the current dip
                ss = dips[L]
                ee = dips[R]
                L_break = False
                for i in range(L, 0, -1):
                    ss = dips[i]
                    if rms[dips[i]] < rms[dips[i-1]]:
                        L_break = True
                        break
                
                R_break = False
                for i in range(R, len(dips)-1):
                    ee = dips[i]
                    if rms[dips[i]] < rms[dips[i+1]]:
                        R_break = True
                        break
                
                # if the end of the detected dips arrays were reached, it's probably better to use the original range
                if not L_break:
                    ss = ap_start
                else:
                    ss *= time_step

                if not R_break:
                    ee = ap_end
                else:
                    ee *= time_step

                resized_ap_ranges.append((ss, ee))
            ap_ranges.extend(resized_ap_ranges)

        # insert AP ranges into label
        for ap_start, ap_end in ap_ranges:
            pos = temp_label.binary_search(ap_start) # find position in array
            curr = temp_label.labels[pos]
            if curr.phone not in self.pauses or curr.start == ap_start: # if it wasn't a pause it's most likely before the detection
                if curr.start < ap_start: # index change not needed for curr.start == ap_start
                    pos += 1
                curr = deepcopy(temp_label.labels[pos])
                ap_start = curr.start
                del temp_label.labels[pos] # delete old label and replace with new
                temp_label.labels.insert(pos, Label(ap_start, min(ap_end, curr.end), 'AP'))
                if ap_end < curr.end: # add SP at the end if needed
                    temp_label.labels.insert(pos+1, Label(ap_end, curr.end, 'SP'))
            else:
                sp_end = curr.end
                curr.end = ap_start # push pause end for AP
                curr.phone = 'SP'
                temp_label.labels.insert(pos+1, Label(ap_start, min(ap_end, sp_end), 'AP')) # add AP
                if ap_end < sp_end: # add SP at the end if needed
                    temp_label.labels.insert(pos+2, Label(ap_end, sp_end, 'SP'))

        # cleanup labels from short pauses
        for i in range(len(temp_label) - 1, 0, -1):
            curr = temp_label.labels[i]
            if curr.length() < time_step and curr.phone in self.pauses: # good enough temporary short threshold
                temp_label.labels[i-1].end = curr.end
                del temp_label.labels[i]
        # short label cleanup for start label
        curr = temp_label.labels[0]
        if curr.length() < time_step and curr.phone in self.pauses:
            temp_label.labels[1].start = curr.start
            del temp_label.labels[0]
        
        self.labels = (temp_label + self.labels[0].start).labels

    def binary_search(self, time):
        L = 0
        R = len(self.labels) - 1

        while L != R:
            m = math.ceil((L + R) / 2)
            if self.labels[m].start > time:
                R = m - 1
            else:
                L = m
        
        return L

    def __len__(self): # number of labels
        return len(self.labels)

    def __getitem__(self, key): # when i needed Segment[x] back then
        return self.labels[key]

    @property
    def start(self): # segment start
        return self.labels[0].start

    @property
    def end(self): # segment end
        return self.labels[-1].end

    def shorten_label(self, max_length = 15):
        self_len = self.length()
        if self_len > max_length: # If label length is bigger
            # Calculate shortest possible length
            pau_len = self.labels[0].length() + self.labels[-1].length()
            shortest = self_len - pau_len
            
            if shortest > max_length: # can't be shortened anymore. just make it not exist !
                return None
            elif shortest == max_length: # extreme shortening
                return Segment(self.labels[1:-1])
            else: # shorten pau boundaries to crop to max length
                labels = deepcopy(self.labels)
                a = labels[0]
                b = labels[-1]
                short_pau = min(a.length(), b.length()) # get shorter length pau
                same_length_pau = shortest + short_pau * 2 

                if same_length_pau < max_length: # shorten long pau first if the sample with similar length paus is shorter
                    long_pau = short_pau + max_length - same_length_pau
                    if a.length() > b.length():
                        a.start = a.end - long_pau
                    else:
                        b.end = b.start + long_pau
                else: # shorten both paus by the shorter length
                    k = (max_length - shortest) / (2 * short_pau)
                    a.start = a.end - k * short_pau
                    b.end = b.start + k * short_pau
                
                return Segment(labels)
        else:
            return deepcopy(self) # no need to do anything

    def segment_label(self, max_length = 15, max_silences = 0, length_relax=0.1): # label splitting...
        # Split by silences first
        labels = []
        pau_pos = []
        # Find all pau positions
        for i in range(len(self.labels)):
            l = self.labels[i]
            if l.phone in self.pauses:
                pau_pos.append(i)

        # segment by pau positions
        for i in range(len(pau_pos)-1):
            s = pau_pos[i]
            e = pau_pos[i+1]+1
            labels.append(Segment(self.labels[s:e]))

        resegment = []
        if max_silences > 0: # concatenate labels for resegmentation
            s = 0
            e = 1
            while s < len(labels): # combine labels one by one until you reach max silences or max length
                curr = combine_labels(labels[s:e])
                if e - s - 1 >= max_silences:
                    temp = curr.shorten_label(max_length=max_length)
                    if temp:
                        resegment.append(temp)
                    else:
                        e -= 1
                        resegment.append(combine_labels(labels[s:e]))
                    s = e
                elif curr.length() > max_length:
                    logging.debug('long len: %f', curr.length())
                    temp = curr.shorten_label(max_length=max_length)
                    if temp:
                        resegment.append(temp)
                        logging.debug('cut down')
                    else:
                        e -= 1
                        shorter = labels[s:e]
                        if len(shorter) > 0:
                            resegment.append(combine_labels(shorter))
                            logging.debug('shorter segment: %d', len(shorter))
                        else:
                            logging.warning('A segment could not be shortened to the given maximum length, this sample might be slightly longer than the maximum length you desire.')
                            k = 1
                            while temp is None:
                                temp = curr.shorten_label(max_length=max_length + k*length_relax)
                                k += 1
                            resegment.append(temp)
                            e += 1
                    s = e
                e += 1
        else: # first segmentation pass already left it with no pau in between
            for l in labels:
                curr = l.shorten_label()
                if curr:
                    resegment.append(curr)

        return resegment

# no clue how else to solve the circular imports lmao
from .conv_tools import *