import os

def diagnose_csv_file(file_path):
    """
    Comprehensive CSV file diagnostics
    """
    print("File Diagnostics:")
    print("-" * 50)
    
    # Check file existence
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        return
    
    # File basic information
    print(f"File Path: {file_path}")
    print(f"File Size: {os.path.getsize(file_path)} bytes")
    
    # Read first few bytes
    with open(file_path, 'rb') as file:
        first_bytes = file.read(100)
        print("First 100 bytes (hex):", first_bytes.hex())
    
    # Use chardet for encoding detection
    import chardet
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        print(f"Detected Encoding: {result}")

# Usage
file_path = 'ml_model\data\dataset.csv'  
diagnose_csv_file(file_path)