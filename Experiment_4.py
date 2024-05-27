import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks
from isotope_reference import alpha_isotope_reference, gamma_isotope_reference  # Import the isotope references

def read_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

def subtract_background(sample_df, background_df):
    corrected_df = sample_df.copy()
    corrected_df['Counts'] = corrected_df['Counts'] - background_df['Counts']
    corrected_df['Counts'] = corrected_df['Counts'].apply(lambda x: max(x, 0))
    return corrected_df

def match_isotopes(peak_energies, isotope_reference, tolerance=0.01):
    matches = []
    for energy in peak_energies:
        for isotope, energies in isotope_reference.items():
            for iso_energy in energies:
                if abs(energy - iso_energy) <= tolerance:
                    confidence = max(0, 1 - (abs(energy - iso_energy) / tolerance))  # Linear confidence
                    matches.append((energy, isotope, iso_energy, confidence))
    return matches

def plot_spectra_interactive(corrected_df, sample_name, energy_unit, energy_min, energy_max, output_dir):
    zoomed_df = corrected_df[(corrected_df[energy_unit] >= energy_min) & (corrected_df[energy_unit] <= energy_max)]

    # Determine height threshold dynamically as 10% of the maximum count value in the zoomed range
    height_threshold = zoomed_df['Counts'].max() * 0.1
    peaks, _ = find_peaks(zoomed_df['Counts'], height=height_threshold)

    peak_energies = zoomed_df[energy_unit].iloc[peaks]
    peak_counts = zoomed_df['Counts'].iloc[peaks]

    # Get the indices of the top 5 peaks
    top_peaks_idx = peak_counts.nlargest(5).index

    # Plot only the top 5 peaks
    top_peaks_energies = peak_energies.loc[top_peaks_idx]
    top_peaks_counts = peak_counts.loc[top_peaks_idx]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=corrected_df[energy_unit], y=corrected_df['Counts'],
                             mode='lines', name=sample_name))

    fig.add_trace(go.Scatter(x=top_peaks_energies, y=top_peaks_counts,
                             mode='markers+text',
                             text=[f'{energy:.2f} {energy_unit}' for energy in top_peaks_energies],
                             textposition="top center",
                             name='Top Peaks',
                             marker=dict(color='red', size=10)))

    fig.update_layout(title=f'Spectrum and Top 5 Peak Labeling ({sample_name}, {energy_min}-{energy_max} {energy_unit})',
                      xaxis_title=f'Energy ({energy_unit})',
                      yaxis_title='Counts')

    fig.write_html(f'{output_dir}\\{sample_name}_Zoomed_Spectrum_{energy_min}_{energy_max}{energy_unit}.html')
    fig.show()

def save_top_peaks_report(corrected_df, sample_name, isotope_reference, output_dir, energy_unit, T_s, T_b=None):
    # Filter peaks for the specified range (1-6 MeV for alpha samples)
    if energy_unit == 'MeV':
        filtered_df = corrected_df[(corrected_df[energy_unit] >= 1.0) & (corrected_df[energy_unit] <= 6.0)]
    else:
        filtered_df = corrected_df

    # Determine height threshold dynamically as 10% of the maximum count value
    height_threshold = filtered_df['Counts'].max() * 0.1
    peaks, _ = find_peaks(filtered_df['Counts'], height=height_threshold)

    peak_energies = filtered_df[energy_unit].iloc[peaks]
    peak_counts = filtered_df['Counts'].iloc[peaks]

    # Get the top 25 peaks
    top_peaks_idx = peak_counts.nlargest(50).index

    top_peaks_energies = peak_energies.loc[top_peaks_idx]
    top_peaks_counts = peak_counts.loc[top_peaks_idx]

    matches = match_isotopes(top_peaks_energies, isotope_reference, tolerance=0.01 if energy_unit == 'MeV' else 1.0)

    # Calculate confidence intervals for counts using radiation counting statistics
    std_devs = top_peaks_counts ** 0.5
    confidence_intervals = 1.96 * std_devs  # 95% confidence interval

    # Calculate net count rates and their standard deviations
    net_count_rates = top_peaks_counts / T_s
    if T_b is not None:
        background_count_rates = peak_counts.mean() / T_b
        net_count_rate_stddev = ((net_count_rates / T_s) + (background_count_rates / T_b)) ** 0.5
    else:
        net_count_rate_stddev = (net_count_rates / T_s) ** 0.5

    report = pd.DataFrame({
        f'Energy ({energy_unit})': top_peaks_energies,
        'Counts +/- Confidence Interval': [f"{count} +/- {ci:.2f}" for count, ci in zip(top_peaks_counts, confidence_intervals)],
        'Count Rate +/- Confidence Interval': [f"{net_count_rate:.5f} +/- {net_count_rate_stddev:.5f}" for net_count_rate, net_count_rate_stddev in zip(net_count_rates, net_count_rate_stddev)]
    })

    # Format the report output for better alignment
    formatted_report = report.to_string(index=False).split('\n')
    energy_header = f'Energy ({energy_unit})'
    counts_header = 'Counts +/- Confidence Interval'
    rate_header = 'Count Rate +/- Confidence Interval'
    formatted_report[0] = f"{energy_header:^18} {counts_header:^32} {rate_header:^40}"
    formatted_report = "\n".join(formatted_report)

    with open(f'{output_dir}\\{sample_name}_Top_50_Peaks_Report.txt', 'w') as f:
        f.write("Top 50 Peaks Report\n")
        f.write(f"Sample: {sample_name}\n\n")
        f.write(formatted_report)
        f.write("\n\nLikely Isotopes:\n")
        for match in matches:
            f.write(f"Energy: {match[0]:.2f} {energy_unit}, Isotope: {match[1]}, Reference Energy: {match[2]:.2f} {energy_unit}, Confidence: {match[3]*100:.2f}%\n")

def process_spectra_interactive(file_path, sheet_name, sample_names, energy_min, energy_max, output_dir, isotope_reference, energy_unit, T_s, T_b=None):
    df = read_data(file_path, sheet_name)
    
    for sample in sample_names:
        if energy_unit == 'MeV':
            sample_df = df[['MeV', sample]].rename(columns={sample: 'Counts'})
            corrected_df = sample_df  # No background subtraction for Alpha tab
        else:
            sample_df = df[['keV', sample]].rename(columns={sample: 'Counts'})
            background_df = df[['keV', 'Background']].rename(columns={'Background': 'Counts'})
            corrected_df = subtract_background(sample_df, background_df)
        
        plot_spectra_interactive(corrected_df, f"{sample}_Alpha" if energy_unit == 'MeV' else f"{sample}_Gamma", energy_unit, energy_min, energy_max, output_dir)
        save_top_peaks_report(corrected_df, f"{sample}_Alpha" if energy_unit == 'MeV' else f"{sample}_Gamma", isotope_reference, output_dir, energy_unit, T_s, T_b)

def main(file_path):
    output_dir = r'C:\Users\17244\OneDrive\Grad School\NUCE 597\Exp4\Output'
    gamma_samples = ['Monazite', 'White Sand', 'LANL']
    alpha_samples = ['Monazite', 'White Sand', 'LANL']

    # Gamma measurement times
    T_s_gamma = 300  # seconds
    T_b_gamma = 300  # seconds

    # Alpha measurement times
    T_s_alpha = 24 * 3600  # 24 hours in seconds

    process_spectra_interactive(file_path, 'Gamma Spec', gamma_samples, 200, 500, output_dir, gamma_isotope_reference, 'keV', T_s_gamma, T_b_gamma)
    process_spectra_interactive(file_path, 'Alpha', alpha_samples, 1, 6, output_dir, alpha_isotope_reference, 'MeV', T_s_alpha)

main(r"C:\Users\17244\OneDrive\Grad School\NUCE 597\Exp4\Data.xlsx")
