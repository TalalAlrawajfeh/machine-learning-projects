#!/usr/bin/python3.6

import json


def read_settings_file(settings_file_path):
    with open(settings_file_path, 'r', encoding='utf-8') as f:
        settings = f.read()
    return json.loads(settings, encoding='utf-8')


SETTINGS = read_settings_file('digits/settings.json')
