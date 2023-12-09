import re

def process_text(data, max_length=1500):
    """
    Process a list of sentences into units, each containing sentences with a combined length not exceeding max_length.

    Parameters:
    - data (list): A list of sentences.
    - max_length (int): Maximum allowed length for combined sentences in a unit.

    Returns:
    - list: A list of units, each containing combined sentences with lengths not exceeding max_length.
    """
    combined_units = []
    current_unit = []

    for sentence in data:
        # Check if adding the current sentence exceeds the maximum length
        if sum(len(s) for s in current_unit) + len(sentence) <= max_length:
            current_unit.append(sentence)
        else:
            combined_units.append(" ".join(current_unit))
            current_unit = [sentence]

    # Add the last unit if it's not empty
    if current_unit:
        combined_units.append(" ".join(current_unit))

    # Split each unit if it exceeds max_length
    final_units = []
    for unit in combined_units:
        final_units.extend([unit[i:i+max_length] for i in range(0, len(unit), max_length)])

    return final_units

def filter_scores(input_list):
    cumulative_sum = 0
    result_list = []

    # Iterate through the sorted list and filter based on cumulative sum and element count
    for element in input_list:
        if cumulative_sum + element[1] < 0.85 and len(result_list) < 9:
            cumulative_sum += element[1]
            result_list.append(element)
    if len(result_list) <= 4:
        return input_list[:5]
    return result_list

def clean_strings(strings):
    cleaned_strings = []
    for string in strings:
        # Remove 'keyflix_' and hyperlinks with 'cloud.deydoo.club'
        cleaned_string = re.sub(r'keyflix_|https://[^ ]*cloud\.deydoo\.club[^ ]*', '', string)
        cleaned_strings.append(cleaned_string.strip())

    return cleaned_strings


