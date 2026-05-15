"""Compatibilité: package `src.visualization` réexportant `src.visualisation`.

Ce package existe pour les imports utilisant l'orthographe américaine
`visualization` alors que le code principal utilise `visualisation`.
"""

from .visualize import *  # noqa: F401,F403
