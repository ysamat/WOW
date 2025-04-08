# Read contents of data_prepare_inputs.txt

def read_inputs_from_file(lines):

    inputs = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):  # Ignore comments and empty lines
            inputs.append(line)
    
    return inputs

def get_data_selection_path(inputs):
    """Returns the first path (data selection table file)."""
    return inputs[0] if len(inputs) > 0 else None


def get_folder_path(inputs):
    """Returns the second path (folder path)."""
    return inputs[1] if len(inputs) > 1 else None

def get_output_path(inputs):
    return inputs[2] if len(inputs) > 1 else None

def get_num_channels(inputs):
    return inputs[3] if len(inputs) > 1 else None

def get_old2new(inputs):
    return inputs[4] if len(inputs) > 1 else None

def get_new2old(inputs):
    return inputs[5] if len(inputs) > 1 else None

def get_channel_ref(inputs):
    return inputs[6] if len(inputs) > 1 else None

def get_max_dis(inputs):
    return inputs[7] if len(inputs) > 1 else None

def get_min_z(inputs):
    return inputs[8] if len(inputs) > 1 else None

def get_max_z(inputs):
    return inputs[9] if len(inputs) > 1 else None

def get_min_speed(inputs):
    return inputs[10] if len(inputs) > 1 else None

def get_max_speed(inputs):
    return inputs[11] if len(inputs) > 1 else None

def get_min_sound_speed(inputs):
    return inputs[12] if len(inputs) > 1 else None

def get_max_sound_speed(inputs):
    return inputs[13] if len(inputs) > 1 else None

def get_same_object(inputs):
    return inputs[14] if len(inputs) > 1 else None

def get_max_dur(inputs):
    return inputs[15] if len(inputs) > 1 else None