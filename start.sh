#!/bin/bash
source venv/bin/activate
python main.py > lisa.log 2>&1 &
