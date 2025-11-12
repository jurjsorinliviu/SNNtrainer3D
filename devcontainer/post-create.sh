#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Setting up SNNtrainer3D Environment...${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
python -m pip install --upgrade pip

# Install requirements
echo -e "\n${GREEN}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Create necessary directories if they don't exist
echo -e "\n${GREEN}Creating required directories...${NC}"
mkdir -p /tmp/data/mnist
mkdir -p experiments
mkdir -p evaluation_metrics_experiments
mkdir -p comparisons

# Set git safe directory (for Codespaces)
echo -e "\n${GREEN}Configuring git...${NC}"
git config --global --add safe.directory /workspaces/$(basename "$PWD")

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${GREEN}To start the application, run:${NC}"
echo -e "  ${BLUE}python app.py${NC}\n"

echo -e "${GREEN}The Flask server will be available at:${NC}"
echo -e "  ${BLUE}http://localhost:5000${NC}\n"

echo -e "${GREEN}For more information, see README.md${NC}\n"