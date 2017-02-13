@echo off
for /l %%a in (1,1,20) do (
python data_prepare.py
python data_sa.py R
)