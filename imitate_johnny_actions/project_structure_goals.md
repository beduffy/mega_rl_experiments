# Johnny Robot Imitation Project Structure Goals

## Objectives
- Train ACT policy for 24-DOF robot control (ref: `imitate_johnny_action_act.py` 117-328)
- Synthesize realistic joint angle sequences (ref: `imitate_johnny_action_act.py` 38-54)
- Enable PyBullet-based simulation evaluation (ref: `imitate_johnny_action_act.py` 329-347)

## Core Components
1. **Data Handling**
   - `ServoDataset` for joint angle sequences (ref: `imitate_johnny_action_act.py` 57-113)
   - Hardcoded demonstration data (ref: `imitate_johnny_action_act.py` 38-54)

2. **Policy Architecture**
   - ACT-based control policy (ref: `imitate_johnny_action_act.py` 402-416)
   - Joint error weighted loss function (ref: `imitate_johnny_action_act.py` 121-127)

3. **Evaluation**
   - PyBullet simulation environment (ref: `imitate_johnny_action_act.py` 329-347)
   - Greeting sequence validation (ref: `imitate_johnny_action_act.py` 312-328)

## File Structure

## Implementation Goals
- [ ] Windowed sequence prediction (ref: `imitate_johnny_action_act.py` 57-113)
- [ ] Joint-space error weighting (ref: `imitate_johnny_action_act.py` 121-127)
- [ ] Mixed precision training (ref: `imitate_johnny_action_act.py` 118)
- [ ] Simulation-based validation (ref: `imitate_johnny_action_act.py` 329-347)

## Training Pipeline
1. Synthetic dataset generation (ref: `imitate_johnny_action_act.py` 373-380)
2. ACT policy configuration (ref: `imitate_johnny_action_act.py` 383-400)
3. Training with joint error prioritization (ref: `imitate_johnny_action_act.py` 117-328)
4. Regular PyBullet simulation checks (ref: `imitate_johnny_action_act.py` 329-347)
5. Model checkpointing and deployment
