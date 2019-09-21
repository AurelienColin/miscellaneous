import sys
from datetime import datetime

def print_header(title):
    print(f'---------\n---------\n{title}\n\n')

def carriage_returned_print(s):
    sys.stdout.write(f'\r{s}')

def print_remaining_time(begin, i, n):
    mean_time = (datetime.now() - begin) / (i + 1)
    ending = (mean_time * n + begin).strftime('%D-%H:%M')
    carriage_returned_print(f"\r{round(i/n*100, 1)}% | {ending} | {mean_time}")
