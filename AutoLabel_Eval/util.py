def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    
def load_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()