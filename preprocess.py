import os
import xmltodict
import pandas as pd

def get_text(value):
    if isinstance(value, str):
        return value.strip()
    elif isinstance(value, dict):
        return value.get('#text', '').strip()
    return ''

def xml_to_csv(xml_file):
    with open(xml_file, 'r', encoding='utf-8') as file:
        data = xmltodict.parse(file.read())

    qa_pairs = []

    # Safely get QAPairs
    document = data.get('Document')
    if not document:
        return qa_pairs  # Skip files that don't have 'Document'

    qapairs_data = document.get('QAPairs')
    if not qapairs_data:
        return qa_pairs

    qapairs = qapairs_data.get('QAPair', [])
    if isinstance(qapairs, dict):
        qapairs = [qapairs]

    for qa in qapairs:
        question = get_text(qa.get('Question', ''))
        answer = get_text(qa.get('Answer', ''))
        if question and answer:
            qa_pairs.append({'question': question, 'answer': answer})

    return qa_pairs

# Directory containing XML files
xml_dir = 'MedQuAD/'

all_qa_pairs = []

for root, dirs, files in os.walk(xml_dir):
    for xml_file in files:
        if xml_file.endswith('.xml'):
            full_path = os.path.join(root, xml_file)
            qa_pairs = xml_to_csv(full_path)
            all_qa_pairs.extend(qa_pairs)

# Save CSV
output_csv = 'medquad_qa_pairs.csv'
df = pd.DataFrame(all_qa_pairs)
df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"CSV file created successfully with {len(df)} rows")
