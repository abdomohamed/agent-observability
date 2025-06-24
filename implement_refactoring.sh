#!/bin/bash
# Script to implement the refactoring

# Create a backup of the original app.py
cp app.py app.py.bak
cp README.md README.md.bak

# Rename the new files
mv app.py.new app.py
mv README.md.new README.md

# Run the app to test it
echo "Refactoring completed. Run the app with: streamlit run app.py"
