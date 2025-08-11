import os
import json
import re
from typing import Dict

def clean_and_group_names_by_decade(data_dir: str, output_json: str) -> None:
    grouped: Dict[int, Dict[str, int]] = {}

    for fname in os.listdir(data_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in ('.csv', '.txt'):
            continue

        year_match = re.search(r'\d{4}', fname)
        if not year_match:
            print(f"[SKIP] {fname}: cannot parse year")
            continue
        year = int(year_match.group())
        decade = (year // 10) * 10
        grouped.setdefault(decade, {})

        path = os.path.join(data_dir, fname)
        with open(path, encoding='utf-8') as f:
            raw_lines = f.readlines()

        seen_names = set()

        # Fixed-width .txt (no commas)
        if ext.lower() == '.txt' and not any(',' in line for line in raw_lines):
            for line in raw_lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                name = re.sub(r'[^a-z]', '', parts[0].lower())
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                try:
                    value = int(parts[-1])
                except ValueError:
                    continue
                grouped[decade][name] = grouped[decade].get(name, 0) + value

        # CSV or whitespace-normalized .txt
        else:
            if ext.lower() == '.txt':
                raw_lines = [','.join(line.strip().split()) for line in raw_lines]

            lines = [line.strip().lower().split(',') for line in raw_lines if line.strip()]
            if not lines:
                continue

            sample_lines = lines[:10]
            num_cols = max(len(row) for row in sample_lines)
            rank_col_index = -1

            for i in range(1, num_cols):
                values = [row[i] for row in sample_lines if len(row) > i]
                if all(re.fullmatch(r'\d+', val) for val in values):
                    rank_col_index = i
                    break
            if rank_col_index == -1:
                rank_col_index = num_cols - 1

            for row in lines:
                if len(row) <= rank_col_index:
                    continue
                name = re.sub(r'[^a-z]', '', row[0].lower())
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                try:
                    value = int(row[rank_col_index])
                except ValueError:
                    continue
                grouped[decade][name] = grouped[decade].get(name, 0) + value

    with open(output_json, 'w', encoding='utf-8') as jf:
        json.dump(grouped, jf, indent=2)

    print(f"[SAVED] grouped name counts by decade → {output_json}")


# Run it
clean_and_group_names_by_decade(
    data_dir="name_data/first",
    output_json="name_data/first_names_by_decade.json"
)
