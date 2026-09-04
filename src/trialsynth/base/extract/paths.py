"""Centralized pystow path constants for the extract pipeline."""
import pystow

# todo: Consider coordinating with HOME_DIR in trialsynth/base/config.py
TRIALSYNTH_BASE = pystow.module("trialsynth")
CONTENT_TXT_DIR = TRIALSYNTH_BASE.module("content", "txt")
RESULTS_DIR = TRIALSYNTH_BASE.module("results")
RESULTS_GROUNDED_DIR = RESULTS_DIR.module("grounded")
CLINICALTRIALS_DIR = TRIALSYNTH_BASE.module("clinicaltrials")
XML_DIR = CLINICALTRIALS_DIR.module("xml")
RESOURCES_DIR = TRIALSYNTH_BASE.module("resources")
