
def get_savings_per_level(level):
    """
    Returns the savings for a given GroupX level (1 to 10).
    Level 1 = Exact (Group)
    Level 2 = GroupX2
    ...
    Level 10 = GroupX10
    
    Based on the column height model for an 8x4 multiplication:
    Col Heights: 1, 2, 3, 4, 4, 4, 4, 3, 2, 1, 0
    Savings per Col (Height - 1): 0, 1, 2, 3, 3, 3, 3, 2, 1, 0
    """
    # Cumulative savings map
    # Level 1 means 0 columns approximated -> Savings 0
    # Level 2 means cols 0,1 approximated -> Savings 0 + 1 = 1
    savings_map = {
        1: 0,
        2: 1,   # 0+1
        3: 3,   # 1+2
        4: 6,   # 3+3
        5: 9,   # 6+3
        6: 12,  # 9+3
        7: 15,  # 12+3
        8: 18,  # 15+3
        9: 20,  # 18+2
        10: 21, # 20+1 (Max for 8x4 structure)
    }
    return savings_map.get(level, 0)

def decode_gene_savings(gene_index):
    """
    Calculates the savings for a single gene index (0-100).
    Index 0-99: Encodes two 8x4 levels.
    Index 100: Zero Multiplier (Max Savings).
    """
    MAX_SAVINGS_8x4 = 21
    MAX_SAVINGS_8x8 = MAX_SAVINGS_8x4 * 2

    if gene_index == 100:
        return MAX_SAVINGS_8x8
    
    if not (0 <= gene_index <= 99):
        raise ValueError(f"Invalid gene index: {gene_index}")

    # Decode index (XY)
    # a = Tens digit, b = Ones digit
    # Level = Digit + 1
    a = gene_index // 10
    b = gene_index % 10
    
    level_lower = a + 1
    level_upper = b + 1
    
    savings_lower = get_savings_per_level(level_lower)
    savings_upper = get_savings_per_level(level_upper)
    
    return savings_lower + savings_upper

def calculate_chromosome_savings(chromosome):
    """
    Calculates the total savings for a full chromosome (list of 9 gene indices).
    """
    total_savings = 0
    for gene_index in chromosome:
        total_savings += decode_gene_savings(gene_index)
    return total_savings

if __name__ == "__main__":
    # Small test
    test_gene = 46 # Should be Level 5 + Level 7 = 9 + 15 = 24
    print(f"Savings for Gene 46: {decode_gene_savings(test_gene)}")
    
    test_paa = [100, 100, 100, 99, 99, 99, 0, 0, 0]
    print(f"Savings for Paa: {calculate_chromosome_savings(test_paa)}")
