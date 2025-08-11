#!/snap/bin/nu
# Create the truth mechanization data sets

print "Creating truth mechanization data sets"

def main [input_dir: string, 
          output_dir: string, 
          gps_interval: int = 10, 
          gps_accuracy: float = 1.0,
          gps_spoofing: float = 0.0
          ] {
    print $"Reading data from: ($input_dir)"
    print $"Writing results to: ($output_dir)"
    print $"GPS interval set to: ($gps_interval) seconds"
    print $"GPS accuracy set to: ($gps_accuracy) meters"
    print $"GPS spoofing set to: ($gps_spoofing) meters"
    # Determine config string if any gps_* value is not default
    let config_str = if ($gps_interval != 10) {
        $"($gps_interval)s"
    } else {
        ''
    }
    let config_str = $config_str + if ($gps_accuracy != 1.0) {
        $"_($gps_accuracy)m"
    } else {
        ''
    }
    let config_str = $config_str + if ($gps_spoofing != 0.0) {
        $"_($gps_spoofing)m"
    } else {
        ''
    }
    # Use config string in output directory if not default
    let final_output_dir = if ($config_str != '') {
        $output_dir | path join $config_str
    } else {
        $output_dir
    }
    # Ensure output directory exists (no error if already exists)
    mkdir $final_output_dir
    # Find all .csv files in input_dir
    glob ($input_dir | path join "**/*.csv") | par-each {|file|
        let input_file = $file
        let base = ($input_file | path basename | str replace ".csv" "")
        print $"Processing: ($input_file) from ($base)"

        # let output_file = ($final_output_dir | path join ($base + "_" + $config_str + ".csv"))
        # print $"Writing results to: ($output_file)"
        # try {
        #     strapdown --mode closed-loop --input $input_file --output $output_file --gps-interval $gps_interval
        # } catch {|err|
        #     print $"Skipping ($input_file) due to error: ($err.msg)"
        # }
    }
    print "Truth mechanization data sets created."
}