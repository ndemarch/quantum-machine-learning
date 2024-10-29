#!/bin/bash

# Execute preprocess.py
python preprocess.py

# Wait for 5 seconds before proceeding to the next script
sleep 2

# Execute stats_analysis.py
python3 stats_analysis.py

# Wait for 5 seconds before proceeding to the next script
sleep 2

# Execute model.py
python model.py
