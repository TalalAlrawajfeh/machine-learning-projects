@echo off

call python -m digits.sequence_generator
call python -m model.icr digits
