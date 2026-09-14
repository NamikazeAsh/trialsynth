"""Centralized pystow path constants for the extract pipeline."""
import pystow

# todo: Consider coordinating with HOME_DIR in trialsynth/base/config.py
TRIALSYNTH_BASE = pystow.module("trialsynth")
CONTENT_DIR = TRIALSYNTH_BASE.module("content")
CONTENT_TXT_DIR = CONTENT_DIR.module("txt")
PMID_TEXTS_CACHE = CONTENT_DIR.join(name="pmid_texts.json.gz")
RESULTS_DIR = TRIALSYNTH_BASE.module("results")
RESULTS_GROUNDED_DIR = RESULTS_DIR.module("grounded")
CLINICALTRIALS_DIR = TRIALSYNTH_BASE.module("clinicaltrials")
XML_DIR = CLINICALTRIALS_DIR.module("xml")
RESOURCES_DIR = TRIALSYNTH_BASE.module("resources")
