import re
import json

# Function to extract key-value pairs from JSON text
def extract_key_value_pairs(json_string):
    # Adjusted regex to capture all content including nested quotes
    role_match = re.search(r"'role':\s*'([^']*)'", json_string)
    content_match = re.search(r"'content':\s*'(.*)'", json_string, re.DOTALL)  # re.DOTALL to capture all lines
    
    if role_match and content_match:
        role_value = role_match.group(1)
        content_value = content_match.group(1)
        return role_value, content_value
    else:
        return None, None

# Function to clean up JSON content by fixing quotes and escaping
# Step 2: Clean the extracted content
def clean_content(content):
    # Replace single quotes with double quotes and handle escape sequences
    content = content.replace("'", '"')
    content = re.sub(r'\\\'', "'", content)  # Replace escaped single quotes
    content = re.sub(r'\\"', '"', content)  # Remove unnecessary backslashes before double quotes
    content = content.replace('\\n', '\n')  # Handle newlines
    return content


# Function to extract Q&A from JSON text
def extract_qanda(json_string):
    try:
        qanda_match = re.search(r"\[(.*?)\]", json_string, re.DOTALL)  # Capture Q&A section enclosed in []
        if qanda_match:
            qanda_text = qanda_match.group(0)  # Extract the entire Q&A section
            qanda_json = json.loads(qanda_text)  # Convert Q&A text to a Python list
            return qanda_json
        else:
            print("No Q&A section found.")
            return []
    except json.JSONDecodeError as e:
        print(f"Failed to parse Q&A: {e}")
        return []

def parse_json_output(invalid_json):
   
    
    # Step 2: Extract key-value pairs
    role, content = extract_key_value_pairs(invalid_json)

    # Step 3: Clean the content
    if content:
        content = clean_content(content)
    else:
        print("Failed to extract 'content' field.")
        return
    # Step 4: Extract Q&A if present
    qanda = extract_qanda(invalid_json)


    # Step 4: Reconstruct the JSON object with the cleaned fields
    if role and content:
        cleaned_json = {
            "role": role,
            "content": content, 
            "qanda": qanda
        }
        
        # Step 5: Attempt to parse and print the final JSON structure
        try:
            print("Final valid JSON structure:")
            print(json.dumps(cleaned_json, indent=4))  # Pretty-print the valid JSON
            return cleaned_json
        except json.JSONDecodeError as e:
            print(f"Failed to parse cleaned JSON: {e}")
    else:
        print("Failed to extract 'role' or 'content' key-value pairs.")
   
    
        
        
# Function for chunking transcription into tokenized chunks
def chunk_transcription(tokenizer, transcription, max_chunk_size):
    """
    Chunks and tokenizes each word in the given transcription.
    Returns a list of chunks where each chunk is a list of tokens.
    """
    print("[DEBUG] Chunking transcription.")
    splitted_transcription = transcription.split(" ")
    queue = [[]]  # Initialize with one empty list to start chunking
    index = 0

    for token in splitted_transcription:
        # If adding this token keeps the chunk size within the limit
        current_chunk = ' '.join(queue[index] + [token])  # Temporary chunk with the new token
        tokenized_chunk = tokenizer.tokenize(current_chunk)

        if len(tokenized_chunk) < max_chunk_size:
            queue[index].append(token)
        else:
            # Start a new chunk if adding the token exceeds max_chunk_size
            queue.append([token])
            index += 1

    print(f"[DEBUG] Finished chunking. Total chunks: {len(queue)}")
    return queue

# Example usage for quiz_text input
if __name__ == "__main__":
    # Sample quiz_text input (replace this with actual quiz text)
    quiz_text = '{"role": "user", "content": "Example content here with nested quotes and special characters."}'
    
    # Parse the quiz text
    parsed_output = parse_json_output(quiz_text, retries=3)

    # Output the parsed result
    print(parsed_output)