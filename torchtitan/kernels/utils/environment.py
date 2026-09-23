# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import os


def get_boolean_env_variable(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ["1", "true"]
