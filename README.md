# nnsvs-db-converter
This fork of the nnsvs-db-converter is meant to coexist with the original to provide a more advanced experience, while sacrificing portability & ease of use. All guides assume you understand how to use Python and/or Conda. If anything in this guide does not make sense, please refer to the original!
## Changes from the original version
- Uses a config over CLI
- A few different libraries, like loguru and click over logging and argparser
- Code is easier to deploy into other tools and such
## Planned/WIP features
- Implement RMVPE PE
- Implement direct SOME inference (for less steps, basically)
- Implement some audio cleanup stuff (fade beginning and end of segments very slightly, etc)
- Update/add more documentation

## Main Usage

### Through Python (Conda recommeneded for Windows)
 1. Create an environment: `conda create -n db_converter python=3.12 -y`
 2. Install the requirements: `pip install -r requirements.txt`
 3. Run the script as follows (this is an example script): `python db_converter.py -L en {PATH_TO_NNSVS_DB}`

All of the following is from the original and has not been updated for this version yet.

# Information
 This is a Python script for converting singing databases with HTS mono labels into the DiffSinger format. This Python script deals with sample segmentation to mostly ensure that samples have a maximum given length and a maximum amount of pauses in between. The recognized silence phonemes by this script are `sil`, `pau`, and `SP`. `AP` and `br` are special symbols considered as breaths. All silences turn into `SP` and all breaths turn into `AP`. It also assumes the labels have silences labeled at the start and end.
 
 This Python script only requires six external libraries to work, unlike the existing one which needs the NNSVS library, which might be hard to install for most people.
# How to Use

## Example commands

 Anything within the square brackets is optional. Read more about them in [the help text.](#help-text-from-the-file-itself)
 
 These example commands assume that your terminal's current working directory is where `db_converter.py` or `db_converter.bat` is in. If you're using the embeddable version, replace `python db_converter.py` to `db_converter.bat`.
 
 **Tip:** You can drag and drop a file or a folder over the terminal window to automatically add the path in the arguments.
 
### Basic Conversion

 If you want to use [MakeDiffSinger](https://github.com/openvpi/MakeDiffSinger) still to do all the extra variance data needed.
 
 **Requirements:** NNSVS-style Database (.wav and .lab only)
 
```cmd
python db_converter.py [-l max_length -s max_silences -w htk|aud] path/to/nnsvs/db 
```

### Conversion with variance duration support

 If you want to use a DiffSinger variance model for timing prediction only.
 
 **Requirements:** NNSVS-style Database (.wav and .lab only), [Language Definition](#language-definition)
 
```cmd
python db_converter.py [-l max_length -s max_silences -w htk|aud] -L path/to/language-def.json path/to/nnsvs/db
```

### Conversion with variance duration and pitch support

 If you want to use a DiffSinger variance model for timing and pitch prediciton.
 
 **Requirements:** NNSVS-style Database (.wav and .lab only), [Language Definition](#language-definition)

```cmd
python db_converter.py [-l max_length -s max_silences -w htk|aud] -L path/to/language-def.json -m [-c] path/to/nnsvs/db
```

### Conversion with variance duration and pitch support (exporting with .ds files)

 If you want to use a DiffSinger variance model for timing and pitch prediciton.
 
 **Requirements:** NNSVS-style Database (.wav and .lab only), [Language Definition](#language-definition)

```cmd
python db_converter.py [-l max_length -s max_silences -w htk|aud] -L path/to/language-def.json -mD [-c] path/to/nnsvs/db
```

# Features

## Language Definition

 Language definition is a `.json` file containing the vowels and the "liquids" of the phoneme system the labels are on. This `.json` file will then be passed on through the `--language-def` argument. This is used for DiffSinger variance duration support. Vowels are where the splits will be made. Liquids are for special cases where a consonant is wanted to be the first part of the split. The liquids will be pushed if the phoneme before it is a specified consonant in the given array or any consonant if it is set to `true`. Here's a quick example `.json` file and a sample split.

```json
{
    "vowels" : ["a", "i", "u", "e", "o", "N", "A", "I", "U", "E", "O"], // All vowels in the phoneme set
    "liquids" : {
        "w" : ["k", "g"], // Only pushes timing for these consonants
        "y" : true // Pushes timing for all consonants
    }
}
```

```
ph_seq    |  SP | k | a | a | k | a | w | a | k | w | a | b | w | a |... 
ph_num    |    2    | 1 |   2   |   2   |   2   |       4       |    ... 
```

## MIDI Estimation
 `nnsvs-db-converter` can estimate MIDI information using the results of the [Language Definition](#language-definition) for note timing and pitch detection for the pitch information. This is used for DiffSinger variance pitch support. Read more about how to use it in [the help text.](#help-text-from-the-file-itself)

 As of 01/16/2024, you now have the ability to choose which pitch estimator to use for MIDI estimation. Only `parselmouth` and `harvest` are available, which is Praat's To Pitch (ac) function and WORLD's Harvest algorithm respectively. A new argument `--pitch-extractor` has been added for this.

 **Note:** Harvest is very CPU-intensive and might be a bit more dangerous to use multithreading on compared to parselmouth.

## Automatic Breath Detection

 As of 12/09/2023, `nnsvs-db-converter` now supports automatic breath detection, using a similar algorithm from [MakeDiffSinger's Acoustic Auto Aligner.](https://github.com/openvpi/MakeDiffSinger/blob/main/acoustic_forced_alignment/enhance_tg.py#L105-L172) It is modified to better capture the breaths, although it may mislabel them still.

 A new argument `--detect-breaths` or `-B` is now added for this, with other arguments to further control the detection. Read more about them in [the help text.](#help-text-from-the-file-itself)

## Multiprocessing

 As of 12/09/2023, `nnsvs-db-converter` now supports multiprocessing to split the workload throughout your computer's cores. Python multiprocessing is known for hogging up CPU, so **please use with caution.**

 A new argument `--num-processes` or `-T` is added for this. This sets the number of processes used. Setting it to 0 uses all of your cores. **Be careful when using this option**

## Sampling Rate Checking
 As of 12/30/2023, `nnsvs-db-converter` now adds a way to check for the sampling rates of each audio file. It will automatically change the sampling rate to the one specified if this does not match. By default, it will make sure all audio files are 44100 Hz. This setting can be turned off by passing `-r 0`.

## .ds file and label export

 `nnsvs-db-converter` supports exporting `.ds` files and label files. `.ds` files can be exported for usage with [SlurCutter,](https://github.com/openvpi/dataset-tools/releases) and label files can be exported for checking the segmented labels or generally for segmenting an NNSVS database. You can export in Audacity format or HTK format. Read more about how to use this in [the help text.](#help-text-from-the-file-itself)

 **Note:** Labels and DiffSinger phoneme transcriptions might be slightly different because of how phoneme replacement is only done for the DiffSinger phoneme transcriptions and not the regular labels.

## Help Text from the file itself
```
usage: db_converter.py [-h] [--num-processes int] [--debug] [--max-length float]
                       [--max-length-relaxation-factor float] [--max-silences int] [--audio-sample-rate int]
                       [--language-def path] [--estimate-midi] [--use-cents] [--pitch-extractor parselmouth | harvest]
                       [--time-step float] [--f0-min float] [--f0-max float] [--voicing-threshold-midi float]
                       [--detect-breaths] [--voicing-threshold-breath float] [--breath-window-size float]
                       [--breath-min-length float] [--breath-db-threshold float] [--breath-centroid-threshold float]
                       [--write-ds] [--write-labels htk | aud]
                       path

Converts a database with mono labels (NNSVS Format) into the DiffSinger format and saves it in a new folder in the
path supplemented.

positional arguments:
  path                  The path of the folder of the database.

options:
  -h, --help            show this help message and exit
  --num-processes int, -T int
                        Number of processes to run for faster segmentation. Enter 0 to use all cores. (default: 1)
  --debug, -d           Show debug logs. (default: False)

segmentation options:
  Options related to segmentation.

  --max-length float, -l float
                        The maximum length of the samples in seconds. (default: 15)
  --max-length-relaxation-factor float, -R float
                        This length in seconds will be continuously added to the maximum length for segments that are
                        too long for the maximum length to cut. (default: 0.1)
  --max-silences int, -s int
                        The maximum amount of silences (pau) in the middle of each segment. Set to a big amount to
                        maximize segment lengths. (default: 0)
  --audio-sample-rate int, -r int
                        The sampling rate in Hz to put the audio files in. If the sampling rates do not match it will
                        be converted to the specified sampling rate. Enter 0 to ignore sample rates. (default: 44100)

MIDI estimation options:
  Options related to MIDI Estimation. MIDI estimation requires a language definition.

  --language-def path, -L path
                        The path of the language definition .json file. If present, phoneme numbers will be added.
                        (default: None)
  --estimate-midi, -m   Whether to estimate MIDI or not. Only works if a language definition is added for note
                        splitting. (default: False)
  --use-cents, -c       Add cent offsets for MIDI estimation. (default: False)
  --pitch-extractor parselmouth | harvest, -p parselmouth | harvest
                        Pitch extractor used for MIDI estimation. Only parselmouth reads voicing-threshold-midi.
                        (default: parselmouth)
  --time-step float, -t float
                        The time step used for all frame-by-frame analysis functions. (default: 0.005)
  --f0-min float, -f float
                        The minimum F0 to detect in Hz. Used in MIDI estimation and breath detection. (default: 40)
  --f0-max float, -F float
                        The maximum F0 to detect in Hz. Used in MIDI estimation and breath detection. (default: 1100)
  --voicing-threshold-midi float, -V float
                        The voicing threshold used for MIDI estimation. (default: 0.45)

breath detection options:
  Options for breath detection. Enabled with --detect-breaths.

  --detect-breaths, -B  Detect breaths within all pauses. (default: False)
  --voicing-threshold-breath float, -v float
                        The voicing threshold used for breath detection. (default: 0.6)
  --breath-window-size float, -W float
                        The size of the window in seconds for breath detection. (default: 0.05)
  --breath-min-length float, -b float
                        The minimum length of a breath in seconds. (default: 0.1)
  --breath-db-threshold float, -e float
                        The threshold in the RMS of the signal in dB to detect a breath. (default: -60)
  --breath-centroid-threshold float, -C float
                        The threshold in the spectral centroid of the signal in Hz to detect a breath. (default: 2000)

output options:
  Options related to output DiffSinger database.

  --write-ds, -D        Write .ds files for usage with SlurCutter or for preprocessing. (default: False)
  --write-labels htk | aud, -w htk | aud
                        Write labels if you want to check segmentation labels. "htk" gives HTK style labels, "aud"
                        gives Audacity style labels. (default: None)
```
