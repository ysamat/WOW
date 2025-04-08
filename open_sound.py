# -*- coding: utf-8 -*-
"""

Created on 2019-10-29
@author: atoultaro
"""
import os
import re
import datetime
import sys
sys.path.append(os.path.dirname("C:/Users/samat/Downloads/dsp-20250105T061136Z-001/dsp/SampleStream.py"))

from soundfile import SoundFile
from SampleStream import SampleStream, Streams


def make_sound_stream(day_sound_path, format_str="%Y%m%d_%H%M%S"):
    #year is four digits, month is two digits, day is two digits, hour is two digits, minute is two digits, second is two digits
    #eg. 20191029_010203
    '''
    Function for making a soundstream capable of iterating through lots of
    files.

    Input:
        day_sound_path - List of folder location(s) containing soundfiles
        format_str - format string of the date default is"%Y%m%d_%H%M%S"
            for Cornell
    Returns
        Returns a soundstrem of all wav or aif files listed in the folder
    '''

    # Declare the stream
    stream_elements = Streams()

    # if it's not a list make it a list
    if isinstance(day_sound_path, (list,)) is False:
        day_sound_path = [day_sound_path]

    for ii in range(len(day_sound_path)):

        # get the file director
        file_dir = day_sound_path[ii]

        # Iterate through the folders and extract associated
        for filename in os.listdir(file_dir):

            # if soundfile add it to the stream
            if filename.endswith(".wav") or filename.endswith(".aif") or \
                    filename.endswith("flac"):
                sound_fullfile = file_dir + '/' + filename
                start = get_start_timestamp(filename, format_str)
                aa = SoundFile(sound_fullfile)
                stream_elements.add_file(sound_fullfile, [len(aa)],
                                         [start], aa.samplerate)
                # print(os.path.join(directory, filename)) # debugging
            else:
                continue

    # Combine streams into a sample stream
    stream = SampleStream(stream_elements)

    print(stream)

    return stream

def get_sample_rate(day_sound_path):
    '''
    Function for getting the sample rate of a soundfile

    Input:
        day_sound_path - List of folder location(s) containing soundfiles

    Returns
        Returns the sample rate of the soundfile
    '''

    # if it's not a list make it a list
    if isinstance(day_sound_path, (list,)) is False:
        day_sound_path = [day_sound_path]

    for ii in range(len(day_sound_path)):

        # get the file director
        file_dir = day_sound_path[ii]

        # Iterate through the folders and extract associated
        for filename in os.listdir(file_dir):

            # if soundfile add it to the stream
            if filename.endswith(".wav") or filename.endswith(".aif") or \
                    filename.endswith("flac"):
                sound_fullfile = file_dir + '/' + filename
                aa = SoundFile(sound_fullfile)
                sample_rate = aa.samplerate
                break

    return sample_rate


def get_start_timestamp(f, format_str="%Y%m%d_%H%M%S"):
    ''' returns a datetime object (start time) given a soundfile/directory
    name in the standard Cornell format

    input:
        f- filename from which to extract the timestamp
        format_str - format string of the timestamp default "%Y%m%d_%H%M%S"
            Cornell format
    '''

    fname = os.path.split(f)[1]

    match_date = re.search(r'\d{8}_\d{6}', fname)
    try:
        # Cornell Format
        start_time = datetime.datetime.strptime(match_date.group(), format_str)
    except AttributeError:
        #  not cornell format try scripps
        match_date = re.search(r'\d{6}_\d{6}', fname)
        try:
            start_time = datetime.datetime.strptime(match_date.group(),
                                                    format_str)
        except AttributeError:
            # Also not scripts, try NOAA
            start_time = datetime.datetime.strptime(fname[-16:-4], format_str)

    return start_time
