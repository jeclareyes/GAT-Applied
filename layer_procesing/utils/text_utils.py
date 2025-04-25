# utils/text_utils.py
import re
import unicodedata
from difflib import get_close_matches


def normalize_text(text):
    """
    Normaliza texto eliminando tildes y caracteres especiales,
    pasando a minúsculas y eliminando espacios redundantes.
    """
    if not isinstance(text, str):
        return text
    # Descomponer caracteres unicode (NFD) y eliminar diacríticos
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    # Reemplazar caracteres suecos específicos
    replacements = {'ä': 'a', 'Ä': 'A', 'ö': 'o', 'Ö': 'O', 'å': 'a', 'Å': 'A'}
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    # Minúsculas y recortar
    text = text.lower().strip()
    # Collapsar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    return text


def find_best_match(target, candidates, cutoff=0.6):
    """
    Busca la mejor coincidencia aproximada de 'target' en la lista 'candidates'.
    Usa difflib.get_close_matches sobre texto normalizado.
    Devuelve la cadena coincidente o None si no hay coincidencias suficientes.
    """
    if not isinstance(target, str) or not candidates:
        return None
    norm_target = normalize_text(target)
    # Normalizar candidatos
    norm_map = {normalize_text(c): c for c in candidates if isinstance(c, str)}
    matches = get_close_matches(norm_target, norm_map.keys(), n=1, cutoff=cutoff)
    if matches:
        return norm_map[matches[0]]
    return None

