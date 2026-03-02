import matplotlib.pyplot as plt

# Too many plots so I put them all here.

# big black dotted line that marks when intervention starts
def sim_intervention_line(warmup_time, max_val, line_color='black', line_style=':', label_text='INTERVENTION'):
    plt.axvline(x=warmup_time, color=line_color, linestyle=line_style, linewidth=2)
    plt.text(
        warmup_time + 0.5,        # x-position slightly right of line
        0.5,                       # y-axis coordinate (0=center, 1=top)
        label_text,
        rotation=90,
        color=line_color,
        va='center',
        ha='left',
        fontsize=14,
        fontweight='bold',
        transform=plt.gca().get_yaxis_transform()  # y in axis coords, x in data coords
    )
# mainly good if you want to know pressure range:
def plot_pressure_waveform(t, P, warmup_time, figsize=(12,4), line_color='black', line_style=':'):
    plt.figure(figsize=figsize)
    plt.plot(t, P, label="Arterial Pressure")
    sim_intervention_line(warmup_time, max(P), line_color, line_style)
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (mmHg)")
    plt.title("Pressure Waveform")
    plt.grid(True)
    plt.legend()
    plt.show()
# less info but a lot easier to interpret:
def plot_map_response(beat_times, MAP_beats, target_map, warmup_time, figsize=(10,5), line_color='black', line_style=':'):
    plt.figure(figsize=figsize)
    plt.plot(beat_times, MAP_beats, 'o-', label="Estimated MAP")
    plt.axhline(target_map, linestyle='--', color='black', label="Target MAP")
    sim_intervention_line(warmup_time, max(MAP_beats), line_color, line_style)
    plt.xlabel("Time (s)")
    plt.ylabel("Mean Arterial Pressure (mmHg)")
    plt.title("Beat-Averaged MAP")
    plt.legend()
    plt.grid(True)
    plt.show()
# only includes first compartment
def plot_drug_concentrations(t, C1_phe, C1_nic, warmup_time, figsize=(10,5), line_color='black', line_style=':'):
    plt.figure(figsize=figsize)
    plt.plot(t, C1_phe, label="C1_phe (Phenylephrine)")
    plt.plot(t, C1_nic, label="C1_nic (Nicardipine)")
    sim_intervention_line(warmup_time, max(max(C1_phe), max(C1_nic)), line_color, line_style)
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration (ug/L)")
    plt.title("Effect-Site Concentrations")
    plt.legend()
    plt.grid(True)
    plt.show()
# phenylephrine and nicardipine infusion in ug/s
def plot_infusion(t, u_phe, u_nic, warmup_time, figsize=(10,5), line_color='black', line_style=':'):
    plt.figure(figsize=figsize)
    plt.plot(t, u_phe, label="u_phe (Phenylephrine infusion)")
    plt.plot(t, u_nic, label="u_nic (Nicardipine infusion)")
    sim_intervention_line(warmup_time, max(max(u_phe), max(u_nic)), line_color, line_style)
    plt.xlabel("Time (s)")
    plt.ylabel("Infusion Rate (ug/s)")
    plt.title("Drug Infusion Commands")
    plt.legend()
    plt.grid(True)
    plt.show()
# peripheral resistance is basically a proxy for map (MAP ~ CO * R)
def plot_resistance(t, R, warmup_time, figsize=(10,5), line_color='black', line_style=':'):
    plt.figure(figsize=figsize)
    plt.plot(t[:-1], R[:-1], linewidth=2)
    sim_intervention_line(warmup_time, max(R[:-1]), line_color, line_style)
    plt.xlabel("Time (s)")
    plt.ylabel("mmHg*s/L")
    plt.title("Peripheral Resistance R Over Time")
    plt.grid(True)
    plt.show()