"""Language sets used by the VibeASR EU benchmark.

FLEURS config names for the 24 official languages of the European Union. The default
sweep uses EU19 -- the 24 official languages minus the five with no FLEURS test audio
worth benchmarking at this scale or no meaningful coverage in the model's training mix
(see README in this directory). Pass --langs to override either list.
"""

# FLEURS config -> (ISO code, English name)
EU24 = {
    "bg_bg": ("bg", "Bulgarian"),
    "cs_cz": ("cs", "Czech"),
    "da_dk": ("da", "Danish"),
    "de_de": ("de", "German"),
    "el_gr": ("el", "Greek"),
    "en_us": ("en", "English"),
    "es_419": ("es", "Spanish"),
    "et_ee": ("et", "Estonian"),
    "fi_fi": ("fi", "Finnish"),
    "fr_fr": ("fr", "French"),
    "ga_ie": ("ga", "Irish"),
    "hr_hr": ("hr", "Croatian"),
    "hu_hu": ("hu", "Hungarian"),
    "it_it": ("it", "Italian"),
    "lt_lt": ("lt", "Lithuanian"),
    "lv_lv": ("lv", "Latvian"),
    "mt_mt": ("mt", "Maltese"),
    "nl_nl": ("nl", "Dutch"),
    "pl_pl": ("pl", "Polish"),
    "pt_br": ("pt", "Portuguese"),
    "ro_ro": ("ro", "Romanian"),
    "sk_sk": ("sk", "Slovak"),
    "sl_si": ("sl", "Slovenian"),
    "sv_se": ("sv", "Swedish"),
}

# The 19 largest EU official languages by number of EU speakers. Irish, Maltese,
# Estonian, Latvian and Slovenian are dropped; they stay available via --langs.
EU19 = [
    "de_de", "fr_fr", "it_it", "es_419", "pl_pl", "ro_ro", "nl_nl", "el_gr",
    "hu_hu", "pt_br", "cs_cz", "sv_se", "bg_bg", "da_dk", "fi_fi", "sk_sk",
    "hr_hr", "lt_lt", "en_us",
]

ALL = list(EU24.keys())


def resolve(spec):
    """Turn a --langs value into a list of FLEURS config names."""
    if not spec or spec == "eu19":
        return list(EU19)
    if spec == "eu24" or spec == "all":
        return list(ALL)
    return [s.strip() for s in spec.split(",") if s.strip()]
