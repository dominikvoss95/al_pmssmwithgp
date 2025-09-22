#!/usr/bin/env python3
import sys

input_file = sys.argv[1]
model_name = sys.argv[2]
output_file = sys.argv[3]

# From Kaito Sugizaki repo -> prospino.in.les_houches
# ALLOWED_BLOCKS = {
#     'SPINFO', 'SPHENOINFO', 'MODSEL', 'MINPAR',
#     'SMINPUTS', 'GAUGE', 'YU', 'YD', 'YE',
#     'AU', 'AD', 'AE', 'MSOFT', 'MASS', 'ALPHA',
#     'HMIX', 'STOPMIX', 'SBOTMIX', 'STAUMIX',
#     'NMIX', 'UMIX', 'VMIX', 'SPHENOLOWENERGY',
#     'FWCOEF', 'IMFWCOEF'
# }

# Only keep necessary blocks for prospino.in.les_houches
ALLOWED_BLOCKS = {
    'SPINFO', 'MODSEL', 'SMINPUTS', 'MINPAR',
    'MASS', 'ALPHA', 'STOPMIX', 'SBOTMIX',
    'STAUMIX', 'NMIX', 'UMIX', 'VMIX',
    'GAUGE', 'YU', 'YD', 'YE','HMIX', 'MSOFT', 
    'AU', 'AD', 'AE'
}

with open(input_file, 'r') as infile:
    lines = infile.readlines()

# Creates stripped prospino.in.les_houches file used as input for prospino
filtered_lines = []
write_block = False
for line in lines:
    stripped = line.strip()
    if stripped.upper().startswith("BLOCK"):
        parts = stripped.split()
        block_name = parts[1].upper()
        write_block = block_name in ALLOWED_BLOCKS
        if write_block:
            filtered_lines.append(line)
    elif stripped.upper().startswith("DECAY"):
        write_block = False
    elif write_block and stripped: 
        filtered_lines.append(line)

with open(output_file, "w") as f:
    f.write(f"# Filtered SLHA file for model {model_name}\n")
    for line in filtered_lines:
        if line.strip(): 
            f.write(line)
