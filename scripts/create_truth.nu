#!/snap/bin/nu
# Create the truth mechanization data sets

print "Creating truth mechanization data sets"

def main [input_dir: string, output_dir: string] {
    print $"Reading data from: ($input_dir)"
    print $"Writing results to: ($output_dir)"
    # Ensure output directory exists (no error if already exists)
    mkdir $output_dir
    # Find all .csv files in input_dir
    ls $input_dir | where type == 'file' and name =~ ".csv$" | each {|file|
        let input_file = $file.name
        let base = ($input_file | path basename | str replace ".csv" "")
        print $"Processing: ($input_file)"
        let output_file = ($output_dir | path join ($base + "_truth.csv"))
        try {
            strapdown --mode closed-loop --input $input_file --output $output_file
        } catch {|err|
            print $"Skipping ($input_file) due to error: ($err)"
        }
    }
    print "Truth mechanization data sets created."
}