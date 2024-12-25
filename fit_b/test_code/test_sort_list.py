def reduce_lists(lists):
    # Sort lists by length in descending order to prioritize longer lists
    sorted_lists = sorted(lists, key=len, reverse=True)
    
    # Store the final result
    result = []
    
    for current_list in sorted_lists:
        # Check if this list is already a subset of any list in the result
        if not any(set(current_list).issubset(set(existing_list)) for existing_list in result):
            result.append(current_list)
    
    return result

# Example usage
input_lists = [[1], [2], [3], [4,9], [5], [6,9], [7], [8], [9,6,4], [10], [11], [12], [13, 16], [14], [15,16,17], [16,15,17], [17,16,15], [18],[13,15,16,17]]
reduced_lists = reduce_lists(input_lists)
print(reduced_lists)
