import json
import logging

def load_json(filename):
    """Load JSON data from a file."""
    with open(filename, 'r') as file:
        return json.load(file)

def save_json(data, filename):
    """Save data as JSON format to a file."""
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def configure_logging():
    """Configure the logging format and level."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Example usage
if __name__ == "__main__":
    configure_logging()
    logging.info("Logging is configured.")