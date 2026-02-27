Control systems are ubiqitous and 'digital twins' are becoming common. The medical versions of these are defective for a few reasons. Human bodies (and animals in general) do
not come with datasheets and are not made to be probed by test kids, reverse engineered, or taken apart for maintenance; most medicine is risky patchwork. This project starts off
with the notion that the missing element in medical automation is physiologically accurate models with parameters that can be reliabily estimated and controlled.

Anaesthesiologists watch over the homeostasis of several different body systems at once during operations and intensive care stays, often including the respiratory and cardiovascular
systems of patients on top of their central nervous systems (anaesthesia). Typically they have to delegate, and in any case manual oversight is prone to error. 

It turns out that there are a variety of 'Windkessel' circuit-analogue models for the cardiovascular system, which may be accurate enough for regulation. There are a couple problems, 
however. First, these circuits are not like familiar manufactured circuits in that their 'component' values are unknown and prone to drift. The other problem is that it is hard to
locate or 'measure' these 'components' because what they represent is inherently distributed. A minor issue is that the analogy between bloodflow and electricity is a bit dubious.

In point of fact, what actually matters for hospitals is that the models work. There are no circuits involved, really, only difference equations and digital signal processing to be 
handled by computers. What is needed for that is trials to show the physiological accuracy of different models and the effect of cardiac drugs on model parameters. Unfortunately, that 
is out of the scope of a school project, but it is possible to show that standard control approaches (i.e., LQR) would work fine on realistic inputs provided known relationships.

There are a few puzzle pieces to this control system:
## Major BP Regulation Model Components

| Project Component      | Approach                                      |
|------------------------|-----------------------------------------------|
| Input                  | Blood flow waveform                           |
| Output                 | Blood pressure waveform                       |
| Live State Estimation  | Signal processing & analysis (blood pressure & flow) |
| Actuators              | Cardiac drugs (simulated)                     |

The system as a whole has more parts (sensors, database, data relay etc.), these are handled in other repository.

## Project Structure

| File               | Purpose                                                                                                                 |
|--------------------|-------------------------------------------------------------------------------------------------------------------------|
| `config.py`        | Simulation setup, parameters/constants to be tuned, imported in all other files                                         |
| `pump.py`          | Generates input blood flow                                                                                              |
| `windkessel.py`    | Calculates BP waveform based on blood flow and a three‑element Windkessel (Z, R, C)                                     |
| `signal_process.py`| Cleans BP waveform and estimates MAP                                                                                    |
| `control.py`       | Controller – changes simulated drug infusion every heart beat based on MAP and main compartment drug concentrations               |
| `pk.py`            | Pharmacokinetic model – calculates drug concentration in plasma over time with a two‑compartment model based on infusion|
| `pd.py`            | Pharmacodynamic model – calculates the effect of drug concentration levels on Windkessel parameters                     |
| `main.py`          | Gathers, prints/plots, and sends results to database and dashboard (outside of this repository)                         |                                                       


  

  
