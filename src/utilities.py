def clean_mda_narrative(raw_text):
    if not raw_text:
        return ""

    # 1. SPLIT INTO BLOCKS: Preserve the paragraph structure
    blocks = raw_text.split('\n')
    clean_blocks = []

    # 2. DEFINE DEBRIS PATTERNS (Common table headers/metadata)
    debris_keywords = [
        r'Three Months Ended', r'Six Months Ended', r'Percentage Change',
        r'In millions', r'In billions', r'Unaudited', r'Ended December',
        r'Fiscal Year', r'Consolidated Statements', r'Note [0-9]+'
    ]
    debris_regex = "|".join(debris_keywords)

    for block in blocks:
        # Normalize whitespace within the block
        block = re.sub(r'\s+', ' ', block).strip()
        
        # SKIP CRITERIA:
        # A. Block is too short (likely a stray header or date)
        if len(block) < 40:
            continue
            
        # B. Block matches common table debris
        if re.search(debris_regex, block, re.IGNORECASE):
            continue
            
        # C. Block is "Digit-Dense" (Even without numbers, tables have many symbols)
        symbol_count = len(re.findall(r'[\$\%\(\)\-\,]', block))
        if symbol_count > 5: # High symbol count usually indicates a table row
            continue

        # 3. NARRATIVE RECOVERY: Strip the numbers but keep the words
        # This preserves the "39% Azure growth" as "Azure growth"
        financial_pattern = r'\$?\b\(?\d{1,3}(?:,\d{3})*(?:\.\d+)?\b%?\)?'
        block = re.sub(financial_pattern, '', block)
        
        # 4. LEFTOVER CLEANUP: Remove empty brackets () and orphaned symbols
        block = re.sub(r'\(\s*\)', '', block) # Removes ()
        block = re.sub(r'\s[\-\,]\s', ' ', block) # Removes stray dashes/commas
        
        # 5. FINAL VALIDATION: Only keep blocks that look like actual sentences
        if len(block) > 40 and any(c.isalpha() for c in block):
            clean_blocks.append(block)

    # REJOIN: Use double newlines to keep paragraphs distinct for your RAG system
    return "\n\n".join(clean_blocks)