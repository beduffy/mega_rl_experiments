# Mouse Imitation Learning Project Structure Goals

## Objectives
- Implement ACT-based policy for mouse movement imitation (ref: `imitate_mouse.py` 137-391)
- Support both real and synthetic mouse trajectory generation (ref: `imitate_mouse.py` 28-104)
- Enable visual state representation through screen capture (ref: `imitate_mouse.py` 107-135)

## Core Components
1. **Data Collection**
   - `MouseRecorder` class for capturing screen/mouse data (ref: `imitate_mouse.py` 28-104)
   - Synthetic circular motion generator (ref: `imitate_mouse.py` 156-191)

2. **Training Infrastructure**
   - `MouseACTDataset` for handling image+position data (ref: `imitate_mouse.py` 107-135)
   - WandB integration for experiment tracking (ref: `imitate_mouse.py` 137-153)

3. **Policy Architecture**
   - ACT-based neural policy with ResNet backbone (ref: `imitate_mouse.py` 207-224)
   - 2D position prediction head (ref: `imitate_mouse.py` 207-222)

## File Structure

## Implementation Goals
- [ ] Real-time screen capture with history buffering (ref: `imitate_mouse.py` 28-50)
- [ ] Data augmentation pipeline for mouse images (ref: TODO - current implementation uses dummy images)
- [ ] Multi-modal input handling (images + normalized positions)
- [ ] Deployment pipeline for real-time mouse control

## Training Pipeline
1. Data collection via `MouseRecorder` or scripted patterns
2. Dataset preprocessing with `MouseACTDataset` (ref: 107-135)
3. ACT policy configuration (ref: 207-222)
4. Training with mixed precision and LR scheduling (ref: 229-391)
5. Visualization of position predictions (ref: 394-420)
