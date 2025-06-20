# synthetic_sonar_data.py

import numpy as np
import pandas as pd

def generate_submarine_signature(duration, sample_rate, base_frequency, harmonics, amplitude, phase=0):
    """Generates a synthetic submarine acoustic signature."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * base_frequency * t + phase)
    for h_freq, h_amp in harmonics:
        signal += h_amp * np.sin(2 * np.pi * h_freq * t + phase)
    return signal

def generate_ocean_noise(duration, sample_rate, noise_level):
    """Generates synthetic ocean ambient noise (white noise for simplicity)."""
    return noise_level * np.random.randn(int(sample_rate * duration))

def generate_synthetic_data(num_samples, duration, sample_rate):
    """Generates a dataset of synthetic sonar signals (submarine or noise)."""
    data = []
    labels = []

    # Submarine signature parameters
    sub_base_freq = 50  # Hz
    sub_harmonics = [(100, 0.3), (150, 0.2)]  # (frequency, amplitude_multiplier)
    sub_amplitude = 1.0

    # Noise parameters
    noise_level = 0.5

    for i in range(num_samples):
        if i % 2 == 0:  # Generate submarine signal
            signal = generate_submarine_signature(duration, sample_rate, sub_base_freq, sub_harmonics, sub_amplitude)
            label = "submarine"
        else:  # Generate noise
            signal = generate_ocean_noise(duration, sample_rate, noise_level)
            label = "noise"
        
        # Add some background noise to all signals
        signal += generate_ocean_noise(duration, sample_rate, noise_level * 0.2)

        data.append(signal)
        labels.append(label)

    return np.array(data), np.array(labels)

if __name__ == "__main__":
    duration = 2  # seconds
    sample_rate = 1000  # Hz
    num_samples = 100  # Total number of synthetic signals

    print(f"Generating {num_samples} synthetic sonar signals...")
    synthetic_signals, synthetic_labels = generate_synthetic_data(num_samples, duration, sample_rate)
    print("Synthetic data generation complete.")

    # Save to a pandas DataFrame and then to a CSV for easy use
    df_data = pd.DataFrame(synthetic_signals)
    df_data["label"] = synthetic_labels
    
    output_filename = "synthetic_sonar_dataset.csv"
    df_data.to_csv(output_filename, index=False)
    print(f"Synthetic dataset saved to {output_filename}")

    print("\nFirst 5 rows of the generated dataset (features are signal values over time):\n")
    print(df_data.head())


