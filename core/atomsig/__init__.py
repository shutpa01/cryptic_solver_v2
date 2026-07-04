"""Atom-map signature model (the redesign's new core).

A clue is recorded as a map from clue atoms to answer atoms: every answer atom is
sourced by exactly one clue piece, and the OPERATION on each piece is read off the
SHAPE of that atom-map (stay -> identity, reverse -> reversal, scramble -> anagram,
a dropped source letter -> deletion, a split answer-span -> container). Indicators do
not initiate an operation; they VALIDATE one the atom-map already implies.

This package is deliberately separate from the per-type engine cascade. It starts
with a HARVESTER that reads the cascade's existing passes (each Parse is already an
atom-map) and converts them into atom-signatures, so we have a real catalogue to
build the verifier against. See core/atomsig/harvest.py.
"""
