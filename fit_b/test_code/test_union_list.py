def find_connected_components(lists):
    # Convert lists to sets for easier manipulation
    sets = [set(lst) for lst in lists]
    
    # Keep track of which sets have been merged
    merged = []
    used = set()

    # Check each set against other sets
    for i, set1 in enumerate(sets):
        if i in used:
            continue
            
        current = set1.copy()
        used.add(i)
        
        # Keep checking for connections until no more are found
        changed = True
        while changed:
            changed = False
            for j, set2 in enumerate(sets):
                if j in used:
                    continue
                    
                # If sets share any elements, merge them
                if current & set2:
                    current |= set2
                    used.add(j)
                    changed = True
        
        merged.append(sorted(list(current)))
    
    return merged

# Your input lists
lists = [[1], [2,3], [3,4], [4,5], [6,10], [7], [8,10], [9], [10,6,8]]

# lists = [[1], [2], [3], [4], [5], [6,14], [7], [8, 14], [9], [10], [11], [12], [13], [14,6,8]]

# Get the result
result = find_connected_components(lists)

# Print the result
print(result)

