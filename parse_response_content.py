def extract_h1_tags(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if '<h1>' in line or '</h1>' in line:
                print(line.strip())

if __name__ == "__main__":
    extract_h1_tags('/home/ubuntu/response_content.txt')
