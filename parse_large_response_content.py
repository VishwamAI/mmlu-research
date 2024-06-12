import re

def extract_h1_tags(file_path, num_examples=10):
    examples_found = 0
    buffer_size = 1024 * 1024  # 1MB buffer size
    leftover = ""
    h1_pattern = re.compile(r'<h1[^>]*>(.*?)</h1>', re.IGNORECASE)
    seen_titles = set()

    with open(file_path, 'rb') as file:
        while examples_found < num_examples:
            chunk = file.read(buffer_size)
            if not chunk:
                break
            chunk = chunk.decode('utf-8', errors='ignore')
            chunk = leftover + chunk
            lines = chunk.split('\n')
            leftover = lines.pop()  # Save the last line in case it is incomplete
            for line in lines:
                match = h1_pattern.search(line)
                if match:
                    title = match.group(0).strip()
                    if title not in seen_titles:
                        print(title)
                        seen_titles.add(title)
                        examples_found += 1
                        if examples_found >= num_examples:
                            break
            print(f"Processed {buffer_size} bytes, file pointer at position: {file.tell()}")

if __name__ == "__main__":
    extract_h1_tags('/home/ubuntu/full_outputs/python3_parse_respon_1718136858.3741724.txt')
