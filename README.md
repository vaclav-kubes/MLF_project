# Machine Learning Fundamemts Final project: 5G Base Station Classification

## Project Overview
This project focuses on the classification of 5G base stations to detect the presence of false base stations (FBS), also known as rogue base stations (RBS) or IMSI catchers. These malicious devices attempt to impersonate legitimate network towers, posing security risks by intercepting user data.

The goal is to develop a machine learning model that can classify whether a given signal comes from a legitimate base station or a rogue one, based on channel frequency response data extracted from LTE/4G signals.

## Theoretical Background
### Mobile Network Base Stations
Cellular communication relies on base stations (eNodeB/gNodeB) to connect mobile devices to the network. These stations broadcast synchronization sequences, including the Primary Synchronization Sequence (PSS) and the Secondary Synchronization Sequence (SSS), which allow devices to identify and connect to the network.

### False Base Stations and Their Risks
A false base station mimics a legitimate one by broadcasting similar synchronization sequences. Once a mobile device connects to it, the attacker can:
- Intercept user communications.
- Send malicious messages.
- Track user locations.

Detecting these rogue stations requires analyzing subtle differences in the transmitted signals.

## Dataset Description
The dataset consists of channel frequency responses derived from PSS/SSS signals, represented as 72×48 matrices. The classification task is to distinguish between:
- **Class 0**: Only a legitimate base station is present.
- **Class 1**: A rogue base station is in the first attacker location.
- **Class 2**: A rogue base station is in the second attacker location.

## Implementation Steps
1. **Data Preprocessing**: Load the dataset, normalize values, and explore different representations (e.g., 2D matrices, 1D vectors).
2. **Model Selection**: Experiment with different machine learning models such as CNNs, LSTMs, and traditional classifiers.
3. **Training & Tuning**: Train models and apply techniques like data augmentation, batch normalization, and hyperparameter tuning.
4. **Evaluation**: Measure model accuracy using test data and optimize performance.
5. **Submission**: Submit predictions to Kaggle and document results.

## Dependencies
- Python (3.x)
- NumPy
- Keras

## Usage
Clone this repository and run the preprocessing and training scripts:
```bash
 git clone https://github.com/vaclav-kubes/MLF_project.git
```

## License
This project is for educational purposes and follows the guidelines of the university course.

