import pandas as pd

Il_0_00 = pd.DataFrame(pd.read_csv('./IR data/Il_0-00.CSV', header=None))
Il_0_25 = pd.DataFrame(pd.read_csv('./IR data/Il_0-25.CSv', header=None))
Il_0_50 = pd.DataFrame(pd.read_csv('./IR data/Il_0-50.CSv', header=None))
Il_0_75 = pd.DataFrame(pd.read_csv('./IR data/Il_0-75.CSv', header=None))
Il_1_00 = pd.DataFrame(pd.read_csv('./IR data/Il_1-00.CSv', header=None))
Il_1_25 = pd.DataFrame(pd.read_csv('./IR data/Il_1-25.CSv', header=None))
Il_1_50 = pd.DataFrame(pd.read_csv('./IR data/Il_1-50.CSv', header=None))
Il_1_75 = pd.DataFrame(pd.read_csv('./IR data/Il_1-75.CSv', header=None))
Il_2_00 = pd.DataFrame(pd.read_csv('./IR data/Il_2-00.CSv', header=None))
Il_2_25 = pd.DataFrame(pd.read_csv('./IR data/Il_2-25.CSv', header=None))
Il_2_50 = pd.DataFrame(pd.read_csv('./IR data/Il_2-50.CSv', header=None))
Il_2_75 = pd.DataFrame(pd.read_csv('./IR data/Il_2-75.CSv', header=None))
Il_3_00 = pd.DataFrame(pd.read_csv('./IR data/Il_3-00.CSv', header=None))
Il_3_25 = pd.DataFrame(pd.read_csv('./IR data/Il_3-25.CSv', header=None))
Il_3_50 = pd.DataFrame(pd.read_csv('./IR data/Il_3-50.CSv', header=None))
Il_3_75 = pd.DataFrame(pd.read_csv('./IR data/Il_3-75.CSv', header=None))
Il_4_00 = pd.DataFrame(pd.read_csv('./IR data/Il_4-00.CSv', header=None))
Il_4_25 = pd.DataFrame(pd.read_csv('./IR data/Il_4-25.CSv', header=None))
Il_4_50 = pd.DataFrame(pd.read_csv('./IR data/Il_4-50.CSv', header=None))
Il_4_75 = pd.DataFrame(pd.read_csv('./IR data/Il_4-75.CSv', header=None))
Il_5_00 = pd.DataFrame(pd.read_csv('./IR data/Il_5-00.CSv', header=None))
Il_5_25 = pd.DataFrame(pd.read_csv('./IR data/Il_5-25.CSv', header=None))
Il_5_50 = pd.DataFrame(pd.read_csv('./IR data/Il_5-500.CSv', header=None))
Il_5_75 = pd.DataFrame(pd.read_csv('./IR data/Il_5-75.CSv', header=None))
Il_6_00 = pd.DataFrame(pd.read_csv('./IR data/Il_6-00.CSv', header=None))
Il_6_25 = pd.DataFrame(pd.read_csv('./IR data/Il_6-25.CSv', header=None))
Il_6_50 = pd.DataFrame(pd.read_csv('./IR data/Il_6-50.CSv', header=None))
Il_6_75 = pd.DataFrame(pd.read_csv('./IR data/Il_6-75.CSv', header=None))
Il_7_00 = pd.DataFrame(pd.read_csv('./IR data/Il_7-00.CSv', header=None))
Il_7_25 = pd.DataFrame(pd.read_csv('./IR data/Il_7-25.CSv', header=None))
Il_7_50 = pd.DataFrame(pd.read_csv('./IR data/Il_7-50.CSv', header=None))
Il_7_75 = pd.DataFrame(pd.read_csv('./IR data/Il_7-75.CSv', header=None))
Il_3_00 = pd.DataFrame(pd.read_csv('./IR data/Il_3-00.CSv', header=None))
Il_3_25 = pd.DataFrame(pd.read_csv('./IR data/Il_3-25.CSv', header=None))
Il_3_50 = pd.DataFrame(pd.read_csv('./IR data/Il_3-50.CSv', header=None))
Il_3_75 = pd.DataFrame(pd.read_csv('./IR data/Il_3-75.CSv', header=None))
Il_8_00 = pd.DataFrame(pd.read_csv('./IR data/Il_8-00.CSv', header=None))


def combine_spectrum():

    dfs = [Il_0_00, Il_0_25, Il_0_50, Il_0_75, Il_1_00, Il_1_25, Il_1_50, Il_1_75, Il_2_00, Il_2_25, Il_2_50, Il_2_75, Il_3_00, Il_3_25, Il_3_50, Il_3_75, Il_4_00, Il_4_25, Il_4_50, Il_4_75, Il_5_00, Il_5_25, Il_5_50, Il_5_75, Il_6_00, Il_6_25, Il_6_50, Il_6_75, Il_7_00, Il_7_25, Il_7_50, Il_7_75, Il_8_00]

    # Set the 1st column (wavelength) as '0' and the sample concentration in 2nd column
    suffix = 0
    for df in dfs:
        df.columns = [0, f"{suffix:.2f}"] 
        suffix += 0.25
    
    # Merge the spectrum DataFrames
    merged_df = dfs[0]    
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on=0, how="outer")
    
    # Set datatype of columns
    merged_df.columns = [float(s) for s in merged_df.columns]   
    merged_df.columns = ['Wavelength'] + list(merged_df.columns[1:]) # Renmae wavelength column

    # Export for furter use
    merged_df.to_csv('./IR data/consolidated_spectrum_raw.csv')

    return merged_df


# Merge the spectrums into one consoldated source
combined_spectrum = combine_spectrum()