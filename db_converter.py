import os, sys
import scipy.signal as signal # find peaks
import click
from pathlib import Path # path fiddling
import traceback # errors
import csv # csv
import json # json
import time # timer
import itertools # repeat
import concurrent.futures as futures # threading

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

from utils.conv_tools import process_lab_wav_pair
from utils.hparams import hparam
from utils.log import logging, set_logger

@click.command()
@click.argument('path', type=str, metavar='path')
@click.option('--config', '-c', required=False, default='configs/base.yaml', type=str, metavar='path', help='Path to configuration file.')
@click.option('--language-def', '-L', required=False, type=str, metavar='path', help='The path of the language definition .json file.')
@click.option('--db_name', '-n', required=False, type=str, default='diffsinger_db', help='The name of the folder output.')
@click.option('--debug', '-d', is_flag=True, help='Show debug logs.')
def convert_db(path: str, config: str, language_def: str, db_name: str, debug: bool):
    try:
        set_logger(debug)

        hparams = hparam(config=config)
        if debug:
            hparams.print_keys()

        # find proper language json
        if language_def:
            if language_def is not os.path.isfile(language_def):
                language_path = f"languages/{language_def}.json"
            else:
                language_path = language_def

        # Prepare locations
        base_path = Path(path)
        diffsinger_loc = base_path / f'{db_name}'
        segment_loc = diffsinger_loc / 'wavs'
        transcript_loc = diffsinger_loc / 'transcriptions.csv'

        # Label finding
        logging.info('Finding all labels.')
        lab_locs = list(base_path.glob('**/*.lab'))
        lab_locs.sort(key=lambda x : x.name)
        lab_locs = [Path(x) for x in lab_locs]
        logging.info(f'Found {len(lab_locs)} label' + ('.' if len(lab_locs) == 1 else 's.'))
        
        # wave equivalent finding
        lab_wav = {}
        for i in lab_locs:
            file = i.name
            wav_name = i.with_suffix('.wav').name
            temp = list(base_path.glob(f'**/{wav_name}'))
            if len(temp) == 0:
                raise FileNotFoundError(f'No wave file equivalent of {file} was found.')
            if len(temp) > 1:
                logging.warning(f'Found more than one instance of a wave file equivalent for {file}. Picking {temp[0]}.')
            lab_wav[i] = temp[0]

        # check for language definition
        lang = None
        transcript_header = ['name', 'ph_seq', 'ph_dur']
        if language_path:
            with open(language_path) as f:
                lang = json.load(f)
            if isinstance(lang['liquids'], list): # support for old lang spec
                lang['liquids'] = {x : True for x in lang['liquids']}
            transcript_header.append('ph_num')

        if hparams.estimate_midi:
            transcript_header.extend(['note_seq', 'note_dur'])

        # actually make the directories
        logging.info('Making directories and files.')
        diffsinger_loc.mkdir(exist_ok=True)
        segment_loc.mkdir(exist_ok=True)

        # prepare transcript.csv
        transcript_f = open(transcript_loc, 'w', encoding='utf8', newline='')
        transcript = csv.DictWriter(transcript_f, fieldnames=transcript_header)
        transcript.writeheader()

        # go through all of it.

        t0 = time.perf_counter()
        if hparams.num_processes == 1:
            transcripts = []
            for lab, wav in lab_wav.items():
                transcripts.extend(process_lab_wav_pair(segment_loc, lab, wav, hparams, language_path, lang))
            logging.info('Writing all transcripts.')
            transcript.writerows(transcripts)
        else:
            workers = hparams.num_processes
            if workers == 0:
                workers = None
                logging.info('Starting process pool with default number of threads.')
            else:
                logging.info(f'Starting process pool with {workers} threads.')
            with futures.ProcessPoolExecutor(max_workers=workers) as executor:
                results = executor.map(process_lab_wav_pair, itertools.repeat(segment_loc), lab_wav.keys(), lab_wav.values(), itertools.repeat(hparams), itertools.repeat(language_path), itertools.repeat(lang))
            logging.info('Writing all transcripts.')
            for res in results:
                transcript.writerows(res)
        runtime = time.perf_counter() - t0
        logging.info(f'Took {runtime} seconds')

        # close the file. very important <3
        transcript_f.close()
            
    except Exception as e:
        for i in traceback.format_exception(e.__class__, e, e.__traceback__):
            print(i, end='')
        os.system('pause')

if __name__ == "__main__":
    convert_db()