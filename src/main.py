from config import *
import plots
from sim_engine import BPSimEngine

def main():
    sim = BPSimEngine(N=N, target_map=target_map)
    sim.run()

    # comment out whichever plots are not wanted
    plots.plot_pressure_waveform(
        t=t,
        P=sim.P,
        warmup_time=warmup_time
    )
    plots.plot_map_response(
        beat_times=[bi * dt for bi in sim.beat_indices],
        MAP_beats=sim.MAP_beats,
        target_map=target_map,
        warmup_time=warmup_time
    )
    plots.plot_drug_concentrations(
        t=t,
        C1_phe=sim.C1_phe,
        C1_nic=sim.C1_nic,
        warmup_time=warmup_time
    )
    plots.plot_infusion(
        t=t,
        u_phe=sim.u_phe,
        u_nic=sim.u_nic,
        warmup_time=warmup_time
    )
    plots.plot_resistance(
        t=t,
        R=sim.R,
        warmup_time=warmup_time
    )

if __name__ == "__main__":
    main()