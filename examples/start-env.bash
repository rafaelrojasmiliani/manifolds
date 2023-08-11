#!/bin/bash
current_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export PYTHONPATH=$current_dir/../python:$current_dir/../cmake-build-Release:$PYTHONPATH
bash
