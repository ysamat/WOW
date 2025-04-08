# -*- coding: utf-8 -*-
"""
Prepare data, including sound clips and info files, for John's localization
program

Created on 2019-09-27
@author: atoultaro
"""

import os
import chardet
import pandas as pd #yash samat 1/22/2025: version 2.2.3
import numpy as np
import soundfile as sf #yash samat 1/22/2025: imported soundfile as librosa.output is outdated
# from math import floor, ceil
# import matplotlib.pyplot as plt
#import sys
#sys.path.append(os.path.dirname("C:/Users/samat/Downloads/open_sound.py"))
import ast
from open_sound import make_sound_stream
from open_sound import get_sample_rate
import librosa
import read_data_prepare_inputs as rdpi

with open('data_prepare_inputs.txt', 'r') as file: # yash samat 1/30/2025: read data_prepare_inputs.txt file
    content = file.readlines() # yash samat 1/30/2025: convert each line of file to list

def datetime_to_list(datetime):
    '''
    Convert datetime to list of date strings
    :param datetime:
    :return:
    '''
    date_str_list = []
    return date_str_list


def write_time(file_to_write, timestamp):
    file_to_write.write(f'{timestamp.year:04d} {timestamp.month:02d} '
                        f'{timestamp.day:02d} {timestamp.hour:02d} '
                        f'{timestamp.minute:02d} {timestamp.second:02d}.'
                        f'{timestamp.microsecond:06d}\n')
    return None


# first arrival logs / selection tables
# first_arrival_seltab = r'85941_CCB27_seltab_20190301_TBLC_short.txt'
# first_arrival_seltab = r'85941_CCB27_seltab_20190301_TBLC.txt'
# first_arrival_seltab = r'85941_ORStateUCCB_002k_M05_multi_UTCz_20190220_000000.Table.1.selections_three_short.txt'
# first_arrival_seltab = r'85941_ORStateUCCB_002k_M05_multi_UTCz_20190220_test.txt'
# first_arrival_seltab = r'85941_ORStateUCCB_002k_M05_multi_UTCz_20190220_20200702_v1.csv'
inputs = rdpi.read_inputs_from_file(content) #yash samat 1/30/2025: list of inputs to read from file
first_arrival_seltab = rdpi.get_data_selection_path(inputs) #yash samat 1/30/2025: get data selection path from list of inputs
print(first_arrival_seltab)

# input directory
sound_path = rdpi.get_folder_path(inputs) #yash samat 1/30/2025: get sound path folder from list of inputs
print(sound_path)

seltab_file = first_arrival_seltab
# output directory
output_path = os.path.join(rdpi.get_output_path(inputs)) #yash samat 1/31/2025: get output path folder from list of inputs
if not os.path.exists(output_path):
    os.mkdir(output_path)

# channel mapping
#chan_map_old2new = {1: 2, 2: 3, 3: 4, 4: 5, 5: 1}
chan_map_old2new = ast.literal_eval(rdpi.get_old2new(inputs)) #yash samat 1/31/2025: changed channel mapping
#chan_map_new2old = {1: 5, 2: 1, 3: 2, 4: 3, 5: 4}
chan_map_new2old = ast.literal_eval(rdpi.get_new2old(inputs)) #yash samat 1/31/2025: changed channel mapping

# reference channel of this hydrophone geometry
# specify the raven pro channel number, channel_ref, which will become the sound file receiver number 1 for sbe
channel_ref = int(rdpi.get_channel_ref(inputs)) #yash samat 1/31/2025: changed channel mapping

# filler gap time
# gap = 1.0 #yash samat 2/2/2025: commented out - not used in code
# row 3: maximum duration (s) of emitted call
# max_dur = 2.0
# row 4: maximum distance in meters
max_dis = float(rdpi.get_max_dis(inputs)) #yash samat 2/2/2025: get maximum distance from list of inputs
# row 5: number of sound channels
num_chan = int(rdpi.get_num_channels(inputs)) #yash samat 2/2/2025: get number of channels from list of inputs
# row 6: channel IDs
chan_id = list(range(1, num_chan+1))
#chan_id = []

# row 10: sampling rate
sample_rate = get_sample_rate(rdpi.get_folder_path(inputs)) #yash samat 2/19/2025: get sample rate from soundfile
# row 9: number of samples
# dur_wav = 30.0  # amount of read wave duration #yash samat 2/2/2025: commented out - not used in code
# num_samples = int(dur_wav*sample_rate) #yash samat 2/2/2025: commented out - not used in code

# row 11: minimum and maximum vertical z coordinate
vertical_bound = [int(rdpi.get_min_z(inputs)), int(rdpi.get_max_z(inputs))] #yash samat 2/2/2025: get min and max z from list of inputs
# row 12: swim speed
swim_speed = [int(rdpi.get_min_speed(inputs)), int(rdpi.get_max_speed(inputs))] #yash samat 2/2/2025: get min and max swim speed from list of inputs
# row 13-17: Time stamp for first sample of sound_i.  Six numbers
# time_ref = ['2009', '02', '19', '00', '01', '45', '.0']
# row 18-22: Time stamp for start of sound of interest for receiver recn(1).
# time_start = ['2009', '02', '19', '00', '01', '45', '.0']
# row 23-27: Time stamp for end of sound of interest for receiver recn(1).
# time_stop  = ['2009', '02', '19', '00', '01', '45', '10.0']
# row 28-32: min & max 3d effective speed of sound (m/s)
sound_speed = [float(rdpi.get_min_sound_speed(inputs)), float(rdpi.get_max_sound_speed(inputs))] #yash samat 2/2/2025: get min and max sound speed from list of inputs
# row 33: same object
same_object = int(rdpi.get_same_object(inputs)) #yash samat 2/2/2025: get same_object parameter from list of inputs

# read first-arrival logs
df_seltab = pd.read_csv(seltab_file, delimiter='\t')
print(df_seltab)
df_seltab = df_seltab[df_seltab['Call ID'].notna()]  # remove rows without Call ID
"""Remove rows with no Before Noise and After Noise - essentially remove rows with no surrounding noise"""
df_seltab = df_seltab[df_seltab['Before Noise (s)'].notna()]
df_seltab = df_seltab[df_seltab['After Noise (s)'].notna()]
df_seltab['Call ID'] = df_seltab['Call ID'].astype(int)
# sort by call ID first and then by begin time
df_seltab = df_seltab.sort_values(by=['Call ID', 'Begin Time (s)'])
print(df_seltab)

id_unique = df_seltab['Call ID'].unique()
id_map = dict()
for ii in range(id_unique.shape[0]):
    id_map.update({id_unique[ii]: ii+1})
df_seltab['Call ID new'] = df_seltab['Call ID'].apply(lambda x: id_map[x])
num_id = id_unique.shape[0]

# hydrophone id mapping
df_seltab['Channel new'] = df_seltab['Channel'].apply(lambda x: chan_map_old2new[x])

# open sound stream
stream = make_sound_stream(sound_path)
stream_starttime = stream.stream.times[0].start

# for row0 in df_seltab.iterrows():
channel_ref -= 1  # channel numbers used here start with 0. E.g. chan 5 => chan 4
file_count = 1
df_one_call_list = []

chan_list = df_seltab['Channel'].tolist()
count = 0
for val in chan_list:
    if val == channel_ref:
            count += 1
if count == 0:
    exit('There are no detected sounds on the reference receiver = ' + str(rdpi.get_channel_ref(inputs)) + ' selected in input file')

print(df_seltab['Channel new'].tolist())

for ii in range(num_id):
    df_one_call_0 = df_seltab.loc[df_seltab['Call ID new'] == ii+1]

    # Check existence of the reference channel;
    # if yes, sort the channels in increasing manner
    # if no, move to the next row
    chan_id = (df_one_call_0['Channel new']-1).tolist()
    print(chan_id)

    # check if there's any duplicate channels, such as 1, 2, 2, 5
    chan_count = np.array(df_one_call_0['Channel new'].value_counts())

    print(chan_count)

    """if not (channel_ref in chan_id):
        print('No reference ID')
        continue"""
    if df_one_call_0.shape[0] <= 2:
        print('The number of hydrophone is smaller than three.')
        continue
    elif any(chan_count >= 2):
        print('Duplicate channels; one channel exists more than once.')
        continue
    else:
        print('New ID: '+str(ii))
        # sort channels
        df_one_call = df_one_call_0.sort_values(by=['Channel new'])
        chan_id.sort()
        
        df_one_call['Before Noise (s)'] = df_one_call['Before Noise (s)'].astype(float)
        df_one_call['After Noise (s)'] = df_one_call['After Noise (s)'].astype(float)
        df_one_call['Channel new'] = df_one_call['Channel new'].astype(int)

        #low_freq is the maximum of the min frequencies for this Call ID = file_count
        low_freq = df_one_call['Low Freq (Hz)'].max()
        
        #high_freq is the minimum of the max frequencies for this Call ID = file_count
        high_freq = df_one_call['High Freq (Hz)'].min()

        """if low_freq is greater than or equal to high_freq, then set low_freq to the minimum of the min frequencies for this Call ID = file_count
        and set high_freq to the maximum of the max frequencies for this Call ID = file_count."""
        if low_freq >= high_freq:
            low_freq = df_one_call['Low Freq (Hz)'].min()
            high_freq = df_one_call['High Freq (Hz)'].max()

        """if low_freq is still greater than or equal to high_freq, then exit the program with an error message."""
        if low_freq >= high_freq:
            exit('Error: low_freq >= high_freq for Call ID = '+str(file_count))
        
        # max_dur = (df_one_call['End Time (s)']-df_one_call['Begin (s)']).max() #yash samat 2/2/2025: commented out
        nrec_sf = (np.array(chan_id) >= 0).sum()

        """yash samat 2/2/2025: adding a column called num_samples which is equal to number of samples of the 
        detected signal plus noise before and after detected signal, all read in from inputed data selection table from raven pro."""
        df_one_call['num_samples'] = (np.floor((df_one_call['End Time (s)'] - df_one_call['Begin Time (s)'] + df_one_call['Before Noise (s)'].astype(float)
                                                + df_one_call['After Noise (s)'].astype(float))*sample_rate)).astype(int)
        df_one_call['first_arr_time'] = df_one_call['Begin Time (s)'] - df_one_call['Before Noise (s)']
        df_one_call['File ID'] = (np.ones(df_one_call.shape[0]) * file_count).astype(int)

        num_sample_max = df_one_call['num_samples'].max()
        sample_dict = dict()
        for index, row in df_one_call.iterrows():
            first_arr_time = stream_starttime + pd.DateOffset(seconds=row['first_arr_time'])
            stream.set_time(first_arr_time)
            samples, samples_info1, samples_info2, _, _ = stream.read(row['num_samples'])
            call_samples = np.concatenate([samples[:, row['Channel']-1], np.zeros(num_sample_max-samples.shape[0])])
            sample_dict.update({row['Channel new']: call_samples})

        sample_list = []
        for cc in chan_id:
            sample_list.append(sample_dict[cc+1])
        sample_multi_chan = np.stack(sample_list).T

        print(sample_multi_chan)

        # row 7: gains
        gains = [1]*nrec_sf
        # row 8: absolute levels
        levels = [-1]*nrec_sf

        # write to info_q.txt where q = file_count = Call ID in Raven pro selection table
        with open(os.path.join(output_path, "info_"+str(file_count)), "w") as f:
            unique_call_types = df_one_call["Call Type"].dropna().unique()
            call_type = unique_call_types[0] if len(unique_call_types) > 0 else "Unknown"
            f.write(f"{call_type}\n")  # row 1: call id
            f.write(f'{low_freq:.2f} {high_freq:.2f}\n')  # row 2: low & high frequencies
            f.write(f'{float(rdpi.get_max_dur(inputs)):.2f}\n')  # row 3: max duration
            f.write(f'{max_dis:.2f}\n')  # row 4: max distance

            f.write(f'{nrec_sf:d}\n')  # row 5: nrec_sf

            for cc in chan_id:  # row 6: channel ids
                f.write(str(cc+1)+' ')
            f.write('\n')

            for cc in range(len(gains)):  # row 7: gains
                f.write(str(gains[cc])+' ')
            f.write('\n')
            for cc in range(len(levels)):  # row 8: levels
                f.write(str(levels[cc])+' ')
            f.write('\n')

            # for ii in range(nrec_sf):  # row 9: num of samples
            #     f.write(f'{num_samples:d}'+' ')
            for cc in chan_id:  # row 6: channel ids
                row = df_one_call[df_one_call['Channel new'] == cc+1]
                f.write(f'{row["num_samples"].values[0]:d} ')
            f.write('\n')
            f.write(f'{sample_rate:d}\n')  # row 10: sample rate
            f.write(f'{vertical_bound[0]:.2f} {vertical_bound[1]:.2f}\n')  # row 11: vertical z
            f.write(f'{swim_speed[0]:.2f} {swim_speed[1]:.2f}\n')  # row 12: swim speed

            # row 13: Time stamp for first sample in sound_i, first_arr_time
            for cc in chan_id:
                row = df_one_call[df_one_call['Channel new'] == cc + 1]
                first_arr_time = stream_starttime + pd.DateOffset(seconds=row['first_arr_time'].values[0])
                write_time(f, first_arr_time)

            # row 13+nrec_sf: Time stamp for start of sound of interest
            for cc in chan_id:
                row = df_one_call[df_one_call['Channel new'] == cc + 1]
                begin_time_call = stream_starttime + pd.DateOffset(seconds=row['Begin Time (s)'].values[0])
                write_time(f, begin_time_call)

            # row 13+nrec_sf*2: Time stamp for end of sound of interest
            for cc in chan_id:
                row = df_one_call[df_one_call['Channel new'] == cc + 1]
                end_time_call = stream_starttime + pd.DateOffset(seconds=row['End Time (s)'].values[0])
                write_time(f, end_time_call)

            for jj in range(nrec_sf):
                f.write(f'{sound_speed[0]:.2f} {sound_speed[1]:.2f}\n')  # row xx: sound speed
            f.write(f'{same_object:d}')  # row xx: same_object parameter

            # output the sound file of selected channels
            # samples_sel = sample_multi_chan[:, chan_id]
            # samples_sel = samples[:, :]
            print(file_count)
            print(sample_rate) 
            #librosa.output.write_wav(os.path.join(output_path, "sound_"+str(file_count)+".wav"), sample_multi_chan.astype('Float32'), sample_rate)
            sf.write(os.path.join(output_path, "sound_"+str(file_count)+".wav"), sample_multi_chan, sample_rate) #yash samat 1/22/2025: used soundfile instead of librosa
            print('Mission accomplished once more.')
            df_one_call_list.append(df_one_call)
    file_count += 1

# output new data frame
# how to add file_id as the new column
print(df_one_call_list) #debug line added by yash 1/21/2025: print list to check contents of list
df_new_calls = pd.concat(df_one_call_list)
#output_new_seltab = os.path.join(output_path, r'__seltab', os.path.basename(first_arrival_seltab)+'_cleaned.txt')
output_new_seltab = os.path.join(output_path, os.path.basename(first_arrival_seltab)+'_cleaned.txt') #yash samat 1/22/2025: changed directory name
df_new_calls.to_csv(output_new_seltab, sep='\t', index=False)