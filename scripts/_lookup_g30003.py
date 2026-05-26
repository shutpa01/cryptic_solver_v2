"""DB lookups to find alternatives for missing coverage pieces."""
import sqlite3, sys
sys.stdout.reconfigure(encoding='utf-8')
ref = sqlite3.connect('data/cryptic_new.db')
ref.row_factory = sqlite3.Row

def show(label, sql, params=()):
    print(f"--- {label} ---")
    rows = ref.execute(sql, params).fetchall()
    for r in rows[:25]:
        print(' ', tuple(r))
    if len(rows) > 25:
        print(f"  ... ({len(rows)} total)")
    print()

# MUGGLE — what definitions does it have?
show("MUGGLE defs", "SELECT definition FROM definition_answers_augmented WHERE UPPER(answer)='MUGGLE'")
show("MUGGLE syn", "SELECT word, synonym FROM synonyms_pairs WHERE UPPER(synonym)='MUGGLE' OR UPPER(word)='MUGGLE' LIMIT 30")

# SPOON
show("SPOON defs", "SELECT definition FROM definition_answers_augmented WHERE UPPER(answer)='SPOON'")
show("SPOON syn", "SELECT word, synonym FROM synonyms_pairs WHERE UPPER(synonym)='SPOON' OR UPPER(word)='SPOON' LIMIT 30")

# tea ~ T homophone
show("homophones for tea", "SELECT * FROM homophones WHERE LOWER(word)='tea' OR LOWER(homophone)='tea'")
show("homophones for t", "SELECT * FROM homophones WHERE LOWER(word)='t' OR LOWER(homophone)='t' LIMIT 20")
show("abbreviations of tea", "SELECT * FROM wordplay WHERE LOWER(indicator)='tea'")
show("syn tea/T", "SELECT word, synonym FROM synonyms_pairs WHERE (LOWER(word)='tea' AND synonym='T') OR (LOWER(synonym)='t' AND LOWER(word)='tea')")

# 'coming in' container indicator
show("'coming in' indicators", "SELECT word, wordplay_type, subtype FROM indicators WHERE LOWER(word)='coming in'")
show("'coming' indicators", "SELECT word, wordplay_type, subtype FROM indicators WHERE LOWER(word)='coming'")
show("'in' as container?", "SELECT word, wordplay_type, subtype FROM indicators WHERE LOWER(word)='in' AND wordplay_type IN ('container','insertion')")

# pelting anagram
show("'pelting' indicators", "SELECT word, wordplay_type, subtype FROM indicators WHERE LOWER(word) LIKE 'pelt%'")

# MARINADE defs
show("MARINADE defs", "SELECT definition FROM definition_answers_augmented WHERE UPPER(answer)='MARINADE'")

# DEER
show("DEER syn", "SELECT word, synonym FROM synonyms_pairs WHERE UPPER(synonym)='DEER' OR UPPER(word)='DEER' LIMIT 30")
show("venison syn", "SELECT word, synonym FROM synonyms_pairs WHERE LOWER(word) LIKE '%venison%' OR LOWER(synonym) LIKE '%venison%'")

# RENDER for DD
show("RENDER defs", "SELECT definition FROM definition_answers_augmented WHERE UPPER(answer)='RENDER'")
show("RENDER syn", "SELECT word, synonym FROM synonyms_pairs WHERE UPPER(synonym)='RENDER' OR UPPER(word)='RENDER' LIMIT 30")

# BARREL for DD
show("BARREL defs", "SELECT definition FROM definition_answers_augmented WHERE UPPER(answer)='BARREL'")
show("BARREL syn", "SELECT word, synonym FROM synonyms_pairs WHERE UPPER(synonym)='BARREL' OR UPPER(word)='BARREL' LIMIT 30")

# WOLVERINE
show("WOLVERINE defs", "SELECT definition FROM definition_answers_augmented WHERE UPPER(answer)='WOLVERINE'")

# IN as synonym for 'part of'
show("'part of' syn", "SELECT word, synonym FROM synonyms_pairs WHERE LOWER(word)='part of' OR LOWER(word)='in' AND UPPER(synonym)='IN' LIMIT 20")
show("'in' syn", "SELECT word, synonym FROM synonyms_pairs WHERE LOWER(word)='in' LIMIT 30")
# 'first off' deletion indicator
show("'first off' indicator", "SELECT word, wordplay_type, subtype FROM indicators WHERE LOWER(word) LIKE 'first%'")

# DAM
show("DAM syn", "SELECT word, synonym FROM synonyms_pairs WHERE UPPER(synonym)='DAM' OR UPPER(word)='DAM' LIMIT 30")
# AGE
show("AGE syn", "SELECT word, synonym FROM synonyms_pairs WHERE UPPER(synonym)='AGE' OR (UPPER(word)='AGE' AND LENGTH(synonym)<25) LIMIT 30")

# MOT
show("MOT abbr", "SELECT * FROM wordplay WHERE UPPER(substitution)='MOT' LIMIT 20")
show("MOT syn", "SELECT word, synonym FROM synonyms_pairs WHERE UPPER(synonym)='MOT' LIMIT 20")

# CID
show("CID abbr", "SELECT * FROM wordplay WHERE UPPER(substitution)='CID' LIMIT 20")
show("CID syn", "SELECT word, synonym FROM synonyms_pairs WHERE UPPER(synonym)='CID' LIMIT 20")

# left hand / right hand → L/R
show("L abbr", "SELECT * FROM wordplay WHERE UPPER(substitution)='L' AND LOWER(indicator) LIKE '%hand%'")
show("R abbr", "SELECT * FROM wordplay WHERE UPPER(substitution)='R' AND LOWER(indicator) LIKE '%hand%'")
show("hand→L?", "SELECT word, synonym FROM synonyms_pairs WHERE LOWER(word) LIKE '%hand%' AND (synonym='L' OR synonym='R') LIMIT 20")

# MARE defs
show("MARE defs", "SELECT definition FROM definition_answers_augmented WHERE UPPER(answer)='MARE'")
show("MARE syn", "SELECT word, synonym FROM synonyms_pairs WHERE UPPER(synonym)='MARE' OR UPPER(word)='MARE' LIMIT 30")

# EL SALVADOR
show("ELSALVADOR defs", "SELECT definition FROM definition_answers_augmented WHERE UPPER(answer)='ELSALVADOR'")
show("ELSALVADOR alt defs (EL SALVADOR)", "SELECT definition FROM definition_answers_augmented WHERE answer='EL SALVADOR'")

# HOTTUB
show("'installs' indicators", "SELECT word, wordplay_type, subtype FROM indicators WHERE LOWER(word)='installs'")
show("'install' indicators", "SELECT word, wordplay_type, subtype FROM indicators WHERE LOWER(word)='install' OR LOWER(word) LIKE 'install%'")

# RUMBA defs
show("RUMBA defs", "SELECT definition FROM definition_answers_augmented WHERE UPPER(answer)='RUMBA'")
show("BA airline", "SELECT * FROM wordplay WHERE UPPER(substitution)='BA' LIMIT 20")

# MEWS — homophone
show("muse homophones", "SELECT * FROM homophones WHERE LOWER(word)='muse' OR LOWER(homophone)='muse' OR LOWER(word)='mews' OR LOWER(homophone)='mews'")
show("MEWS defs", "SELECT definition FROM definition_answers_augmented WHERE UPPER(answer)='MEWS'")
show("MEWS syn", "SELECT word, synonym FROM synonyms_pairs WHERE UPPER(synonym)='MEWS' OR UPPER(word)='MEWS' LIMIT 30")
show("ponder/MUSE syn", "SELECT word, synonym FROM synonyms_pairs WHERE LOWER(word)='ponder' OR LOWER(synonym)='muse' OR UPPER(synonym)='MUSE' LIMIT 30")
show("stables → mews", "SELECT word, synonym FROM synonyms_pairs WHERE LOWER(word) LIKE '%stables%' OR LOWER(word) LIKE '%mews%' OR LOWER(synonym) LIKE '%stables%' LIMIT 20")

# 'not entirely' deletion indicator
show("'not entirely' indicator", "SELECT word, wordplay_type FROM indicators WHERE LOWER(word)='not entirely' OR LOWER(word) LIKE 'not%' LIMIT 30")
show("'entirely' indicator", "SELECT word, wordplay_type FROM indicators WHERE LOWER(word)='entirely' OR LOWER(word) LIKE '%entirely%' LIMIT 30")

# FLYTIP definition variant
show("FLYTIP defs", "SELECT definition FROM definition_answers_augmented WHERE UPPER(answer)='FLYTIP' OR answer='FLY-TIP'")

# 21a MARSH — 'On' indicator? Charade order. Indicator not needed actually.
# 28a CRISES — no MISS.
# 7d Shivering anagram OK.

# 'BARREL' for 'powerless' meaning
show("powerless / over a barrel", "SELECT word, synonym FROM synonyms_pairs WHERE LOWER(synonym) LIKE '%powerless%' OR (LOWER(word) LIKE 'over%' AND UPPER(synonym)='BARREL')")
