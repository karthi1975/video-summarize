import re
import json
from utils import parse_json_output

# Function to read the invalid JSON string from a file
def read_quiz_text_from_file(file_path):
    """
    Reads the quiz text (JSON string) from a file.
    """
    with open(file_path, 'r') as file:
        return file.read()

# Function to write cleaned JSON to a file
def write_cleaned_json_to_file(output_file_path, data):
    """
    Writes cleaned JSON data to a file.
    """
    with open(output_file_path, 'w') as file:
        formatted_json = json.dumps(data, indent=4)  # Pretty-print JSON with indentation
        file.write(formatted_json)  # Write the formatted JSON string to the file
        print(f"Cleaned JSON has been saved to {output_file_path}")


# Main function to read, process, and write the cleaned output
def process_quiz_file(input_file_path, output_file_path):
    """
    Reads quiz text from the input file, processes it, and writes the cleaned output to the output file.
    """
    # Read quiz text from file
    quiz_text = read_quiz_text_from_file(input_file_path)

    # Parse the quiz text and clean it
    cleaned_json = parse_json_output(quiz_text)

    # Write the cleaned JSON output to the output file
    write_cleaned_json_to_file(output_file_path, cleaned_json)

# Example usage
if __name__ == "__main__":
    input_file_path = 'original-string.json'  # Replace with the actual input file path
    output_file_path = 'cleaned-quiz-output.json'  # Replace with the desired output file path
    
    # Process the quiz text file
    process_quiz_file(input_file_path, output_file_path)