#!/usr/bin/env python3

import sys
import csv
import os

def read_slha_params(slha_path):
    """Function to parse the 19 pMSSM parameters from SLHA input file."""
    slha = {}

    if not os.path.isfile(slha_path):
        print(f"[WARNING] SLHA file not found: {slha_path}")
        return slha

    block = None
    with open(slha_path) as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines or lines beginning with #
            if not line or line.startswith('#'):
                continue
            
            # Exctract parts with BLOCK
            if line.startswith("BLOCK"):
                parts = line.split()
                if len(parts) > 1:
                    block = parts[1].upper()
                continue
            
            # Extract first part from line as index and second as value
            tokens = line.split()
            try:
                idx = int(tokens[0])
                val = float(tokens[1])
            except (ValueError, IndexError):
                continue

            # Extract 15 pMSSM parameters from EXTPAR block
            if block == "EXTPAR":
                if idx == 1:  slha["M_1"] = val
                elif idx == 2:  slha["M_2"] = val
                elif idx == 3:  slha["M_3"] = val
                elif idx == 23: slha["mu"] = val
                elif idx == 26: slha["mA"] = val
                elif idx == 31: slha["meL"] = val
                elif idx == 32: slha["mtauL"] = val
                elif idx == 34: slha["meR"] = val
                elif idx == 36: slha["mtauR"] = val
                elif idx == 41: slha["mqL1"] = val
                elif idx == 43: slha["mqL3"] = val
                elif idx == 44: slha["muR"] = val
                elif idx == 46: slha["mtR"] = val
                elif idx == 47: slha["mdR"] = val
                elif idx == 49: slha["mbR"] = val

            # Extract tan_beta
            elif block == "MINPAR" and idx == 3:
                slha["tan_beta"] = val

            # Extract Trilinear couplings
            elif block == "AU" and tokens[:2] == ['3', '3']:
                slha["At"] = float(tokens[2])
            elif block == "AD" and tokens[:2] == ['3', '3']:
                slha["Ab"] = float(tokens[2])
            elif block == "AE" and tokens[:2] == ['3', '3']:
                slha["Atau"] = float(tokens[2])

    return slha


def sum_cross_sections(xsec_file, out_csv, model="model", iteration="0", slha_file=None):
    '''Function to sum the cross sections and write them in a csv file'''
    if not os.path.isfile(xsec_file):
        print(f"[WARNING] Missing: {xsec_file}")
        return

    total = 0.0
    n = 0

    # EW and QCD crosssections 
    valid = {"sg", "gg", "ss", "sb", "tb", "bb", "ll", "nn", "ng", "ns"}
    
    # Take the 12th (NLO) or 9th(LO - if NLO 0) column of the prospino.dat as cross section and sum it up
    # LO (Leading Order) - lowest perturbation, NLO(Next Leading Order) = with loop and real emissions
    # Prospino gives NLO for strong SUSY channels (gg, sg, ss, sb) but only LO for EW channels (nn, ll, ns, ng)
    with open(xsec_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].lower() not in valid:
                continue
            lo  = float(parts[9])  # σ_LO
            nlo = float(parts[11]) # σ_NLO
            if nlo is not None and nlo > 0.0:
                total += nlo
                n += 1
            elif lo is not None:
                total += lo
                n += 1
    if n == 0:
        print(f"[WARNING] No xsecs found in {xsec_file}")
        return

    inputs = read_slha_params(slha_file) if slha_file else {}
    
    # Warn if SLHA file missing
    if not inputs or sum(v is not None for v in inputs.values()) < 10:
        print(f"[WARNING] Incomplete or missing SLHA info for model {model}")
        return

    # Write headers for csv file
    headers = [
        "model", "xsec_TOTAL",
        "M_1", "M_2", "tan_beta", "mu", "M_3", "At", "Ab", "Atau", 
        "mA", "mqL3", "mtR", "mbR", "meL", "mtauL", "meR", "mtauR", 
        "mqL1", "muR", "mdR"
    ]
    row = {key: inputs.get(key, "") for key in headers if key not in {"model", "xsec_TOTAL"}}
    row = {"model": model, "xsec_TOTAL": f"{total:.6e}", **row}

    write_header = not os.path.isfile(out_csv)

    with open(out_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[INFO] total xsec: {total:.2e} pb written to {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("[INFO] Usage: python3 xsec_summary.py <prospino.dat> <out.csv> [model] [iter] [slha_file]")
        sys.exit(1)

    dat_file = sys.argv[1]
    out_csv = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "model"
    iteration = sys.argv[4] if len(sys.argv) > 4 else "0"
    slha = sys.argv[5] if len(sys.argv) > 5 else None

    sum_cross_sections(dat_file, out_csv, model, iteration, slha)
