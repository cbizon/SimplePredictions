#!/usr/bin/env python
"""
Clean indications CSV file by detecting and fixing split rows.

The file has unquoted fields containing newlines, which causes CSV parsers
to split one logical row into multiple parsed rows. This script detects
these cases and merges them back together.

Expected structure:
- Valid rows start with a drug ID (contains ':')
- Invalid rows (fragments) start with text and should be merged with previous row
- The fragments are typically in column 5 (indications text)
"""

import argparse
import csv
from pathlib import Path


def is_valid_drug_id(drug_id: str) -> bool:
    """Check if drug_id has a valid format (contains colon separator)."""
    if not drug_id:
        return False
    return ':' in drug_id


def merge_split_rows(input_file):
    """
    Read CSV and merge split rows back together.

    Returns a list of complete, valid rows as dictionaries.
    """
    rows = []
    current_row = None
    fragments_merged = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for parsed_row in reader:
            drug_id = (parsed_row.get('final normalized drug id') or '').strip()

            if is_valid_drug_id(drug_id):
                # This is a valid row start
                if current_row is not None:
                    # Save the previous complete row
                    rows.append(current_row)

                # Start a new row
                current_row = parsed_row
            else:
                # This is a fragment that should be merged with current_row
                if current_row is None:
                    print(f"Warning: Found fragment before any valid row: {drug_id[:50]}")
                    continue

                fragments_merged += 1

                # The fragment represents a continuation where the newline occurred
                # in the middle of a field (typically 'indications text')
                # The fragment's first column contains the continuation of that field
                # And subsequent columns in the fragment are actually the NEXT fields

                # Append the fragment's first field to 'indications text'
                indications_text = current_row.get('indications text', '')
                fragment_text = drug_id  # The fragment appears in the drug_id column

                if indications_text:
                    current_row['indications text'] = indications_text + '\n' + fragment_text
                else:
                    current_row['indications text'] = fragment_text

                # The remaining non-empty fields in the fragment are the ACTUAL
                # values for fields 5-10 that were shifted due to the split
                # We need to copy them to the current_row if they're empty there

                # Map the fragment columns to actual columns
                # Fragment has: drug_label -> disease name, disease_id -> drug|disease, etc.
                if parsed_row.get('final normalized drug label'):
                    current_row['disease name'] = parsed_row.get('final normalized drug label')
                if parsed_row.get('final normalized disease id'):
                    current_row['drug|disease'] = parsed_row.get('final normalized disease id')
                if parsed_row.get('final normalized disease label'):
                    current_row['FDA'] = parsed_row.get('final normalized disease label')
                if parsed_row.get('indications text'):
                    current_row['EMA'] = parsed_row.get('indications text')
                if parsed_row.get('disease name'):
                    current_row['PMDA'] = parsed_row.get('disease name')
                if parsed_row.get('drug|disease'):
                    current_row['hyperrelations'] = parsed_row.get('drug|disease')

        # Don't forget the last row
        if current_row is not None:
            rows.append(current_row)

    print(f"Merged {fragments_merged} row fragments")

    return rows, fieldnames


def clean_csv(input_file, output_file):
    """
    Read the CSV, merge split rows, and write cleaned version.
    """
    print("Reading and merging split rows...")
    rows, fieldnames = merge_split_rows(input_file)

    print(f"\nWriting {len(rows):,} complete rows to output file...")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary:")
    print(f"  Complete rows written: {len(rows):,}")

    # Validate output
    valid_count = sum(1 for row in rows if is_valid_drug_id(row.get('final normalized drug id', '')))
    print(f"  Rows with valid drug IDs: {valid_count:,}")
    if valid_count != len(rows):
        print(f"  WARNING: {len(rows) - valid_count} rows still have invalid drug IDs!")


def main():
    parser = argparse.ArgumentParser(
        description='Clean indications CSV by merging split rows'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to write cleaned CSV file'
    )

    args = parser.parse_args()

    print(f"Reading: {args.input}")
    print(f"Writing: {args.output}")
    print()

    clean_csv(args.input, args.output)

    print("\nDone!")


if __name__ == '__main__':
    main()
