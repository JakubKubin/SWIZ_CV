# logging_setup.py
"""Centralna konfiguracja logowania dla calego systemu stereo.

Konsola pokazuje poziom z LOG_LEVEL (domyslnie INFO) - czyli to, co istotne
podczas normalnej pracy. Do pliku logs/stereo.log trafia ZAWSZE pelny DEBUG,
zeby po fakcie mozna bylo przesledzic kazdy etap pipeline i wartosci posrednie.

Uzycie (raz, na starcie procesu - CLI lub backend):
    from logging_setup import setup_logging
    setup_logging()
"""
import logging
import logging.handlers
import os
from pathlib import Path

# Folder i nazwa pliku logow - mozna nadpisac zmiennymi srodowiskowymi.
LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
LOG_FILE = os.environ.get("LOG_FILE", "stereo.log")

_configured = False


def setup_logging(
    console_level: int | None = None,
    file_level: int = logging.DEBUG,
) -> Path:
    """Konfiguruje root logger: handler konsoli + rotujacy plik DEBUG w logs/.

    Idempotentna - kolejne wywolania nie dokladaja duplikatow handlerow
    (kazdy modul moze ja wywolac bez obawy o powielone linie w logach).

    Args:
        console_level: poziom logow na konsoli; domyslnie z LOG_LEVEL (INFO)
        file_level:    poziom logow w pliku; domyslnie DEBUG (pelna diagnostyka)

    Returns:
        Sciezka do pliku logow.
    """
    global _configured

    log_path = LOG_DIR / LOG_FILE
    if _configured:
        return log_path

    if console_level is None:
        # LOG_LEVEL np. DEBUG/INFO/WARNING; nieznana wartosc -> INFO
        level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        console_level = getattr(logging, level_name, logging.INFO)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    # Root musi przepuszczac najnizszy z poziomow, inaczej plik nie dostanie DEBUG
    root.setLevel(min(console_level, file_level))

    # Konsola - zwiezly format, tylko to co wazne w danym uruchomieniu
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    # Plik - pelny DEBUG z timestampem i nazwa modulu, rotacja po 5 MB (x3 kopie)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Usuwamy ewentualne handlery z basicConfig(), zeby nie dublowac linii konsoli
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(console)
    root.addHandler(file_handler)

    _configured = True
    logging.getLogger(__name__).debug("Logowanie skonfigurowane -> %s (konsola=%s, plik=%s)",
                                       log_path, logging.getLevelName(console_level),
                                       logging.getLevelName(file_level))
    return log_path
