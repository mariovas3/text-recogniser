#!/bin/bash


# expected format in input file is:
# file_name1 'the first string' 'the second string'
# file_name2 'the first string' 'the second string'
# there can be multiple lines for the same file name;

# Check if the input file exists
if [ ! -f "$1" ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

# Read the file containing file names and string pairs
while IFS= read -r line; do
    # Split the line into file name and string pair
    file=$(echo "$line" | cut -d' ' -f1)
    strings=$(echo "$line" | cut -d' ' -f2-)

	# set the field separator to \'
	# and ignore the first, third and fifth token after separation;
	IFS=\' read -r _ first_string _ second_string _ <<< "$line"

    # Substitute all occurrences of the first string with the second string in the file
    sed -i "s/$first_string/$second_string/g" "$file"

    echo "Replaced '$first_string' with '$second_string' in $file"
done < "$1"
